"""DuckDB destination: a local, credential-free warehouse.

Each rollout gets its own database file. The connection is opened with
external access disabled except for the rollout's workspace, and the
configuration is locked, so SQL issued by the agent cannot read files outside
its workspace (e.g. ground-truth CSVs) or re-enable access.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import duckdb
import yaml

from elt_rl.destinations.base import Destination, Namespace, QueryResult, safe_identifier
from elt_rl.task import ELTTask


class DuckDBDestination(Destination):
    name = "duckdb"
    display_name = "DuckDB"

    def __init__(self, root: str | Path | None = None, schema: str = "elt"):
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="elt_rl_duckdb_"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self._conns: dict[str, duckdb.DuckDBPyConnection] = {}
        self._locks: dict[str, threading.Lock] = {}

    # -- lifecycle ------------------------------------------------------------
    def _db_path(self, task: ELTTask, rollout_id: str) -> Path:
        return self.root / f"{safe_identifier(task.name, rollout_id)}.duckdb"

    def _open(self, db_path: Path, workspace: Path) -> duckdb.DuckDBPyConnection:
        workspace = workspace.resolve().as_posix()
        conn = duckdb.connect(
            str(db_path),
            config={"autoinstall_known_extensions": False, "autoload_known_extensions": False},
        )
        # Order matters: allowed_directories can only be set once the database
        # is running and before external access is disabled. Only absolute
        # paths are checked against it (relative paths are always refused).
        conn.execute(f"SET allowed_directories = ['{workspace}']")
        conn.execute("SET enable_external_access = false")
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        conn.execute("SET lock_configuration = true")
        return conn

    def _register(self, task: ELTTask, rollout_id: str, workspace: Path, db_path: Path) -> Namespace:
        ns = Namespace(
            rollout_id=rollout_id,
            database=str(db_path),
            schema=self.schema,
            workspace=workspace,
        )
        self._conns[rollout_id] = self._open(db_path, workspace)
        self._locks[rollout_id] = threading.Lock()
        return ns

    def _provision(self, task: ELTTask, rollout_id: str, workspace: Path) -> Namespace:
        db_path = self._db_path(task, rollout_id)
        db_path.unlink(missing_ok=True)
        return self._register(task, rollout_id, workspace, db_path)

    def _clone_from_template(
        self, task: ELTTask, template: str, rollout_id: str, workspace: Path
    ) -> Namespace:
        db_path = self._db_path(task, rollout_id)
        shutil.copyfile(template, db_path)
        return self._register(task, rollout_id, workspace, db_path)

    def _teardown(self, ns: Namespace) -> None:
        conn = self._conns.pop(ns.rollout_id, None)
        self._locks.pop(ns.rollout_id, None)
        if conn is not None:
            conn.close()
        Path(ns.database).unlink(missing_ok=True)
        Path(ns.database + ".wal").unlink(missing_ok=True)

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()

    # -- SQL ------------------------------------------------------------------
    def _execute(self, ns: Namespace, sql: str, *, as_agent: bool) -> QueryResult:
        # Agent and grader share the locked-down connection: DuckDB allows one
        # configuration per open file, and the grader never needs file access.
        with self._locks[ns.rollout_id]:
            cur = self._conns[ns.rollout_id].cursor()
            try:
                cur.execute(f"SET schema = '{ns.schema}'")
                cur.execute(sql)
                if cur.description is None:
                    return QueryResult([], [])
                columns = [d[0] for d in cur.description]
                return QueryResult(columns, cur.fetchall())
            finally:
                cur.close()

    def qualify(self, ns: Namespace, table: str) -> str:
        return f"{ns.schema}.{table}"

    def _list_tables_sql(self, ns: Namespace) -> str:
        return (
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{ns.schema}' AND table_type = 'BASE TABLE'"
        )

    def _perturb_sql(self, ns: Namespace, table: str) -> str:
        return f"DELETE FROM {self.qualify(ns, table)} WHERE rowid % 5 = 0"


def build_flat_file_template(task: ELTTask, out_path: str | Path, schema: str = "elt") -> Path:
    """Load a local task's flat-file sources into a template database.

    This is the reference extraction/loading for local tasks; the result is
    cloned per rollout when ``el_mode="preloaded"`` (transform-only curriculum).
    """
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    config = yaml.safe_load((task.input_dir / "config.yaml").read_text(encoding="utf-8"))
    readers = {"csv": "read_csv_auto", "jsonl": "read_json_auto", "json": "read_json_auto",
               "parquet": "read_parquet"}
    conn = duckdb.connect(str(out_path))
    try:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for source in config.get("flat_files", []):
            path = (task.input_dir / source["path"]).resolve().as_posix()
            reader = readers[source["format"]]
            conn.execute(
                f"CREATE TABLE {schema}.{source['table']} AS SELECT * FROM {reader}('{path}')"
            )
    finally:
        conn.close()
    return out_path
