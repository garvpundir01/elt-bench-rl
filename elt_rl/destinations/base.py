"""Warehouse-agnostic destination interface.

The env, tools and grader only talk to a ``Destination``; supporting a new
warehouse means implementing one subclass (connection, identifier rules,
namespace lifecycle) and registering it in ``destinations/__init__.py``.

Every rollout gets its own ``Namespace`` (a database, file, or schema that no
other rollout touches), because an ELT environment is stateful: the agent's
actions accumulate in the warehouse, and G concurrent rollouts of the same task
must not see or overwrite each other's tables.
"""

from __future__ import annotations

import asyncio
import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from elt_rl.task import ELTTask

AGENT_RESULT_MAX_ROWS = 50


@dataclass(frozen=True)
class Namespace:
    """Where one rollout's raw and model tables live."""

    rollout_id: str
    database: str
    schema: str
    workspace: Path
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


def safe_identifier(*parts: str, max_len: int = 200) -> str:
    """Join parts into an identifier that is valid unquoted in every backend."""
    ident = "_".join(re.sub(r"[^A-Za-z0-9_]", "_", p) for p in parts if p)
    if not ident or ident[0].isdigit():
        ident = f"t_{ident}"
    return ident[:max_len]


class Destination(ABC):
    """A data warehouse that ELT pipelines load into and transform within."""

    #: Key of this destination's block in ``config.yaml`` (e.g. ``snowflake``).
    name: str
    #: Human-readable name used in prompts.
    display_name: str

    # -- namespace lifecycle (sync; wrapped as async below) -------------------
    @abstractmethod
    def _provision(self, task: ELTTask, rollout_id: str, workspace: Path) -> Namespace:
        """Create an empty, isolated namespace for one rollout."""

    @abstractmethod
    def _clone_from_template(
        self, task: ELTTask, template: str, rollout_id: str, workspace: Path
    ) -> Namespace:
        """Create a namespace pre-populated with already-loaded raw tables.

        Used by ``el_mode="preloaded"`` to skip slow extraction/loading syncs
        (e.g. a zero-copy ``CREATE DATABASE ... CLONE`` on Snowflake).
        """

    @abstractmethod
    def _teardown(self, ns: Namespace) -> None:
        """Drop everything the rollout created. Must be idempotent."""

    # -- SQL execution --------------------------------------------------------
    @abstractmethod
    def _execute(self, ns: Namespace, sql: str, *, as_agent: bool) -> QueryResult:
        """Run one statement with the namespace as the default context.

        ``as_agent=True`` runs with the agent's (restricted) privileges;
        ``False`` uses the grader's privileges.
        """

    @abstractmethod
    def qualify(self, ns: Namespace, table: str) -> str:
        """Fully qualified name of ``table`` inside the namespace."""

    @abstractmethod
    def _list_tables_sql(self, ns: Namespace) -> str:
        """Query returning one column of table/view names in the namespace."""

    @abstractmethod
    def _perturb_sql(self, ns: Namespace, table: str) -> str:
        """Statement deleting a deterministic ~20% of ``table``'s rows."""

    def agent_config(self, ns: Namespace, task_config: dict) -> dict:
        """Rewrite the task's ``config.yaml`` so the agent targets ``ns``."""
        config = copy.deepcopy(task_config)
        block = config.setdefault(self.name, {}).setdefault("config", {})
        block["database"] = ns.database
        block["schema"] = ns.schema
        return config

    def close(self) -> None:
        """Release any client-side connection state."""

    # -- async API used by the env, tools and grader --------------------------
    async def provision(self, task: ELTTask, rollout_id: str, workspace: Path) -> Namespace:
        return await asyncio.to_thread(self._provision, task, rollout_id, workspace)

    async def clone_from_template(
        self, task: ELTTask, template: str, rollout_id: str, workspace: Path
    ) -> Namespace:
        return await asyncio.to_thread(
            self._clone_from_template, task, template, rollout_id, workspace
        )

    async def teardown(self, ns: Namespace) -> None:
        await asyncio.to_thread(self._teardown, ns)

    async def execute(self, ns: Namespace, sql: str, *, as_agent: bool) -> QueryResult:
        return await asyncio.to_thread(self._execute, ns, sql, as_agent=as_agent)

    async def list_tables(self, ns: Namespace) -> list[str]:
        result = await self.execute(ns, self._list_tables_sql(ns), as_agent=False)
        return [str(row[0]) for row in result.rows]

    async def row_count(self, ns: Namespace, table: str) -> int:
        result = await self.execute(
            ns, f"SELECT COUNT(*) FROM {self.qualify(ns, table)}", as_agent=False
        )
        return int(result.rows[0][0])

    async def fetch_table(self, ns: Namespace, table: str) -> pd.DataFrame:
        result = await self.execute(ns, f"SELECT * FROM {self.qualify(ns, table)}", as_agent=False)
        return result.to_frame()

    async def drop_relation(self, ns: Namespace, name: str) -> None:
        fqn = self.qualify(ns, name)
        for kind in ("VIEW", "TABLE"):
            try:
                await self.execute(ns, f"DROP {kind} IF EXISTS {fqn}", as_agent=False)
            except Exception:
                # Dropping a TABLE as VIEW (or vice versa) errors on some backends.
                pass

    async def materialize(self, ns: Namespace, name: str, select_sql: str) -> None:
        """Execute a compiled model and store its result as a table."""
        await self.execute(
            ns, f"CREATE OR REPLACE TABLE {self.qualify(ns, name)} AS\n{select_sql}", as_agent=False
        )

    async def perturb(self, ns: Namespace, table: str) -> None:
        await self.execute(ns, self._perturb_sql(ns, table), as_agent=False)


def format_result(result: QueryResult, max_rows: int = AGENT_RESULT_MAX_ROWS) -> str:
    """Render a query result as compact text for the agent."""
    if not result.columns:
        return "OK (statement returned no rows)"
    df = result.to_frame()
    shown = df.head(max_rows)
    text = shown.to_string(index=False, max_colwidth=60)
    if len(df) > max_rows:
        text += f"\n... ({len(df)} rows total, showing first {max_rows})"
    else:
        text += f"\n({len(df)} rows)"
    return text
