"""Snowflake destination (the warehouse used by the official ELT-Bench).

Layout follows ELT-Bench: one database per task with raw and model tables in
``AIRBYTE_SCHEMA``. For isolation each rollout gets its own database
``<TASK>_<ROLLOUT>`` owned by the Airbyte role; the rewritten ``config.yaml``
points Airbyte, dbt and the agent's ``sql`` tool at it, and teardown drops it.

Two connections are used:

* the grader/admin connection (host-side credential JSON, never shown to the
  agent) provisions, grades and drops namespaces;
* the agent connection uses the task's Airbyte user/role from ``config.yaml``,
  i.e. the same privileges the pipeline itself runs with.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from elt_rl.destinations.base import Destination, Namespace, QueryResult, safe_identifier
from elt_rl.task import ELTTask


class SnowflakeDestination(Destination):
    name = "snowflake"
    display_name = "Snowflake"

    def __init__(
        self,
        admin_credential: dict[str, Any],
        schema: str = "AIRBYTE_SCHEMA",
        agent_role: str = "AIRBYTE_ROLE",
        database_prefix: str = "RL",
    ):
        self.admin_credential = dict(admin_credential)
        self.schema = schema
        self.agent_role = agent_role
        self.database_prefix = database_prefix
        self._admin = None
        self._agent_conns: dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- connections ----------------------------------------------------------
    def _admin_conn(self):
        if self._admin is None:
            import snowflake.connector

            self._admin = snowflake.connector.connect(**self.admin_credential)
        return self._admin

    def _agent_conn(self, ns: Namespace):
        conn = self._agent_conns.get(ns.rollout_id)
        if conn is None:
            import snowflake.connector

            cfg = ns.extra["agent_credential"]
            conn = snowflake.connector.connect(
                account=self.admin_credential["account"],
                user=cfg["username"],
                password=cfg["password"],
                role=cfg.get("role", self.agent_role),
                warehouse=cfg.get("warehouse"),
                database=ns.database,
                schema=ns.schema,
            )
            self._agent_conns[ns.rollout_id] = conn
        return conn

    def _admin_exec(self, *statements: str) -> None:
        with self._lock:
            cur = self._admin_conn().cursor()
            try:
                for stmt in statements:
                    cur.execute(stmt)
            finally:
                cur.close()

    # -- lifecycle ------------------------------------------------------------
    def _database_name(self, task: ELTTask, rollout_id: str) -> str:
        return safe_identifier(self.database_prefix, task.name, rollout_id).upper()

    def _agent_credential(self, task: ELTTask) -> dict[str, Any]:
        config = yaml.safe_load((task.input_dir / "config.yaml").read_text(encoding="utf-8"))
        return dict(config[self.name]["config"])

    def _grant_statements(self, db: str) -> list[str]:
        return [
            f"GRANT OWNERSHIP ON DATABASE {db} TO ROLE {self.agent_role} COPY CURRENT GRANTS",
            f"GRANT OWNERSHIP ON SCHEMA {db}.{self.schema} TO ROLE {self.agent_role} "
            "COPY CURRENT GRANTS",
        ]

    def _namespace(self, task: ELTTask, rollout_id: str, workspace: Path, db: str) -> Namespace:
        return Namespace(
            rollout_id=rollout_id,
            database=db,
            schema=self.schema,
            workspace=workspace,
            extra={"agent_credential": self._agent_credential(task)},
        )

    def _provision(self, task: ELTTask, rollout_id: str, workspace: Path) -> Namespace:
        db = self._database_name(task, rollout_id)
        self._admin_exec(
            f"CREATE OR REPLACE DATABASE {db}",
            f"CREATE SCHEMA {db}.{self.schema}",
            *self._grant_statements(db),
        )
        return self._namespace(task, rollout_id, workspace, db)

    def _clone_from_template(
        self, task: ELTTask, template: str, rollout_id: str, workspace: Path
    ) -> Namespace:
        # Zero-copy clone: metadata-only, so preloading raw tables is ~instant
        # regardless of data size (vs. minutes per Airbyte sync).
        db = self._database_name(task, rollout_id)
        self._admin_exec(f"CREATE OR REPLACE DATABASE {db} CLONE {template}", *self._grant_statements(db))
        return self._namespace(task, rollout_id, workspace, db)

    def _teardown(self, ns: Namespace) -> None:
        conn = self._agent_conns.pop(ns.rollout_id, None)
        if conn is not None:
            conn.close()
        self._admin_exec(f"DROP DATABASE IF EXISTS {ns.database}")

    def close(self) -> None:
        for conn in self._agent_conns.values():
            conn.close()
        self._agent_conns.clear()
        if self._admin is not None:
            self._admin.close()
            self._admin = None

    # -- SQL ------------------------------------------------------------------
    def _execute(self, ns: Namespace, sql: str, *, as_agent: bool) -> QueryResult:
        if as_agent:
            cur = self._agent_conn(ns).cursor()
            try:
                cur.execute(sql)
                if cur.description is None:
                    return QueryResult([], [])
                return QueryResult([d[0] for d in cur.description], cur.fetchall())
            finally:
                cur.close()
        with self._lock:
            cur = self._admin_conn().cursor()
            try:
                cur.execute(f"USE SCHEMA {ns.database}.{ns.schema}")
                cur.execute(sql)
                if cur.description is None:
                    return QueryResult([], [])
                return QueryResult([d[0] for d in cur.description], cur.fetchall())
            finally:
                cur.close()

    def qualify(self, ns: Namespace, table: str) -> str:
        return f"{ns.database}.{ns.schema}.{table}"

    def _list_tables_sql(self, ns: Namespace) -> str:
        return (
            f"SELECT table_name FROM {ns.database}.information_schema.tables "
            f"WHERE table_schema = '{ns.schema.upper()}' AND table_type = 'BASE TABLE'"
        )

    def _perturb_sql(self, ns: Namespace, table: str) -> str:
        # Snowflake only allows HASH(*) in a SELECT list (not in DELETE/WHERE), so
        # rebuild the table without a deterministic ~20% of its rows.
        fqn = self.qualify(ns, table)
        return (
            f"CREATE OR REPLACE TABLE {fqn} AS SELECT * EXCLUDE (_elt_rl_h) FROM "
            f"(SELECT *, MOD(ABS(HASH(*)), 5) AS _elt_rl_h FROM {fqn}) WHERE _elt_rl_h <> 0"
        )
