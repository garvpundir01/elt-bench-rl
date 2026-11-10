"""Discover and compile the transformation models an agent submits.

The submitted pipeline's transformation stage is a set of dbt-style model
files: one ``SELECT`` per data model in a ``models/`` directory, optionally
using ``{{ ref('m') }}``, ``{{ source('s', 't') }}`` and ``{{ config(...) }}``.
The grader compiles and executes these itself (instead of trusting whatever
tables the agent left behind), so the reward is a function of the pipeline.

Only the jinja subset that ELT-Bench models actually use is supported; a full
dbt project can instead be compiled with ``dbt compile`` and its
``target/compiled`` output passed through unchanged (no jinja left to render).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRS = {"target", "dbt_packages", "logs", ".venv", "node_modules"}
_CONFIG_RE = re.compile(r"\{\{\s*config\s*\(.*?\)\s*\}\}", re.DOTALL)
_REF_RE = re.compile(r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")
_SOURCE_RE = re.compile(
    r"\{\{\s*source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"
)
_JINJA_RE = re.compile(r"\{\{|\{%")
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


@dataclass(frozen=True)
class ModelFile:
    name: str
    path: Path
    raw_sql: str

    @property
    def refs(self) -> list[str]:
        return _REF_RE.findall(self.raw_sql)


def find_model_files(workspace: Path) -> dict[str, ModelFile]:
    """All ``*.sql`` files under any ``models/`` directory in the workspace.

    If two files share a name, the shallower path wins (deterministically).
    """
    found: dict[str, ModelFile] = {}
    candidates = sorted(
        (p for p in workspace.rglob("*.sql") if "models" in p.relative_to(workspace).parts),
        key=lambda p: (len(p.parts), str(p)),
    )
    for path in candidates:
        rel_parts = path.relative_to(workspace).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        name = path.stem.lower()
        if name not in found:
            found[name] = ModelFile(name, path, path.read_text(encoding="utf-8"))
    return found


def render(sql: str, qualify: Callable[[str], str]) -> str:
    """Render the supported jinja subset; ``qualify(table)`` gives a FQN."""
    sql = _CONFIG_RE.sub("", sql)
    sql = _REF_RE.sub(lambda m: qualify(m.group(1)), sql)
    sql = _SOURCE_RE.sub(lambda m: qualify(m.group(2)), sql)
    if _JINJA_RE.search(sql):
        raise ValueError("unsupported jinja in model (only ref/source/config are supported)")
    return sql.strip().rstrip(";").strip()


def compile_model(workspace: Path, model: ModelFile, qualify: Callable[[str], str]) -> str:
    """Render a model, falling back to dbt's own compiled SQL for richer jinja."""
    try:
        return render(model.raw_sql, qualify)
    except ValueError:
        compiled = sorted(workspace.rglob(f"target/compiled/**/{model.path.name}"))
        if not compiled:
            raise
        return compiled[0].read_text(encoding="utf-8").strip().rstrip(";")


def execution_order(models: dict[str, ModelFile], targets: list[str]) -> list[str]:
    """Targets plus their ``ref`` dependencies, dependencies first."""
    order: list[str] = []
    state: dict[str, str] = {}

    def visit(name: str) -> None:
        if state.get(name) == "done" or name not in models:
            return
        if state.get(name) == "visiting":
            raise ValueError(f"cycle in model refs involving {name!r}")
        state[name] = "visiting"
        for dep in models[name].refs:
            visit(dep.lower())
        state[name] = "done"
        order.append(name)

    for target in targets:
        visit(target)
    return order


def references_any(sql: str, tables: list[str]) -> bool:
    """True if the (comment-stripped) SQL mentions any of the given tables."""
    body = _COMMENT_RE.sub(" ", sql).lower()
    return any(re.search(rf"\b{re.escape(t.lower())}\b", body) for t in tables)
