"""Destination registry.

``DestinationSpec`` is a small picklable description of a destination;
``EnvGroupBuilder``s carry the spec (they must be picklable) and build the
live ``Destination`` (which holds connections) inside ``make_envs``.

To add a warehouse: subclass ``Destination`` and add a branch to ``build``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from elt_rl.destinations.base import Destination, Namespace, QueryResult, format_result

DESTINATION_NAMES = ("duckdb", "snowflake")


@dataclass(frozen=True)
class DestinationSpec:
    name: str = "duckdb"
    # Host-side admin credential JSON (snowflake); never copied to a workspace.
    credential_path: str | None = None
    # Directory for per-rollout database files (duckdb).
    root: str | None = None

    def build(self) -> Destination:
        if self.name == "duckdb":
            from elt_rl.destinations.duckdb_dest import DuckDBDestination

            return DuckDBDestination(root=self.root)
        if self.name == "snowflake":
            from elt_rl.destinations.snowflake_dest import SnowflakeDestination

            if not self.credential_path:
                raise ValueError("snowflake destination requires credential_path")
            credential = json.loads(Path(self.credential_path).read_text(encoding="utf-8"))
            return SnowflakeDestination(admin_credential=credential)
        raise ValueError(f"unknown destination {self.name!r}; expected one of {DESTINATION_NAMES}")


__all__ = [
    "DESTINATION_NAMES",
    "Destination",
    "DestinationSpec",
    "Namespace",
    "QueryResult",
    "format_result",
]
