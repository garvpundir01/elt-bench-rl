"""Execution-based grading and reward for ELT rollouts.

Correctness follows the official ELT-Bench evaluator
(``evaluation/eva_stage1.py`` and ``eva_stage2.py``):

* Stage 1 (extract & load): every expected raw table exists as a base table
  in the namespace with exactly the expected row count.
* Stage 2 (transform): each data model matches its ground-truth CSV. Both
  sides are serialized to CSV and re-read as strings, sorted by the model's
  unique key (remaining columns as tie-breakers), and every gold column must
  match (case-insensitive name; numeric within rtol=1e-2; strings trimmed and
  case-insensitive; NULL only matches NULL). Extra columns are allowed.

On top of that, the reward adds partial credit (RL needs reward variance
within a GRPO group, and full-pipeline success is rare for small models) and
two checks against outputs that are not computed from the loaded data.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from elt_rl.destinations import Destination, Namespace
from elt_rl.models import compile_model, execution_order, find_model_files, references_any
from elt_rl.task import ELTTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparator (ported from ELT-Bench evaluation/eva_stage2.py)
# ---------------------------------------------------------------------------
def _sort_series_key(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    n_non_null = col.notna().sum()
    if n_non_null > 0 and int(num.notna().sum()) == int(n_non_null):
        return num
    return col.astype("string").str.strip().str.lower()


def sort_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    lower = {c.lower(): c for c in df.columns}
    resolved: list[str] = []
    for k in keys:
        actual = lower.get(k.lower())
        if actual is not None and actual not in resolved:
            resolved.append(actual)
    ordered = resolved + [c for c in df.columns if c not in resolved]
    aux = pd.DataFrame({c: _sort_series_key(df[c]) for c in ordered})
    order = aux.sort_values(by=ordered, kind="mergesort", na_position="last").index
    return df.loc[order].reset_index(drop=True)


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced if coerced.isna().sum() == s.isna().sum() else s


def vectors_match(v1: pd.Series, v2: pd.Series, rtol: float = 1e-2, atol: float = 1e-9) -> bool:
    if len(v1) != len(v2):
        return False
    n1 = _to_numeric_if_possible(v1.reset_index(drop=True))
    n2 = _to_numeric_if_possible(v2.reset_index(drop=True))
    if pd.api.types.is_numeric_dtype(n1) and pd.api.types.is_numeric_dtype(n2):
        a, b = n1.to_numpy(dtype=float), n2.to_numpy(dtype=float)
        nan_match = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
        return bool(np.all(nan_match | close))
    for a, b in zip(v1, v2):
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            return False
        if isinstance(a, str) and isinstance(b, str):
            if a.strip().lower() != b.strip().lower():
                return False
        elif a != b:
            return False
    return True


def as_csv_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV exactly like the official evaluator does."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    if not buf.getvalue().strip():
        return pd.DataFrame()
    return pd.read_csv(buf, dtype=str, keep_default_na=True)


@dataclass
class ModelResult:
    name: str
    gold_columns: int = 0
    exists: bool = False
    match: bool = False
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    error: str | None = None
    flagged: str | None = None  # reason credit was revoked by an anti-hacking check


def compare_model(gt: pd.DataFrame, df: pd.DataFrame, keys: list[str], result: ModelResult) -> None:
    gt, df = sort_by_keys(gt, keys), sort_by_keys(df, keys)
    actual = {c.lower(): c for c in df.columns}
    for gold_col in gt.columns:
        col = actual.get(gold_col.lower())
        if col is None:
            result.missed.append(gold_col)
        elif vectors_match(gt[gold_col], df[col]):
            result.matched.append(gold_col)
        else:
            result.unmatched.append(gold_col)
    # A model with no rows only "matches" if the ground truth is empty too.
    nonempty_ok = len(df) > 0 or len(gt) == 0
    result.match = nonempty_ok and not result.unmatched and not result.missed


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardWeights:
    stage1: float = 0.2  # fraction of raw tables loaded with exact row counts
    models: float = 0.6  # fraction of data models fully correct (ELT-Bench SRDT)
    columns: float = 0.2  # fraction of gold columns correct (dense shaping)


@dataclass(frozen=True)
class GraderConfig:
    # "pipeline": re-execute the submitted model files, then grade (training default).
    # "warehouse": grade whatever tables the agent left (official-evaluator parity).
    mode: str = "pipeline"
    # Metamorphic check: perturb the raw tables and re-run the models; outputs
    # that do not change at all were not computed from the loaded data.
    perturbation_check: bool = True
    weights: RewardWeights = RewardWeights()


@dataclass
class GradeResult:
    reward: float
    stage1_tables_ok: int
    stage1_tables_total: int
    models: list[ModelResult]
    notes: list[str] = field(default_factory=list)

    @property
    def stage1_pass(self) -> bool:
        return self.stage1_tables_ok == self.stage1_tables_total

    def metrics(self) -> dict[str, float]:
        n_models = max(len(self.models), 1)
        gold_cols = sum(m.gold_columns for m in self.models)
        ok_cols = sum(len(m.matched) for m in self.models if not m.flagged)
        return {
            "elt/reward": self.reward,
            "elt/stage1_frac": self.stage1_tables_ok / max(self.stage1_tables_total, 1),
            "elt/stage1_pass": float(self.stage1_pass),
            "elt/models_correct_frac": sum(m.match and not m.flagged for m in self.models)
            / n_models,
            "elt/columns_correct_frac": ok_cols / max(gold_cols, 1),
            "elt/models_missing": float(sum(not m.exists for m in self.models)),
            "elt/models_errored": float(sum(m.error is not None for m in self.models)),
            "elt/models_flagged": float(sum(m.flagged is not None for m in self.models)),
        }

    def summary(self) -> str:
        lines = [
            f"reward={self.reward:.3f} stage1={self.stage1_tables_ok}/{self.stage1_tables_total}"
        ]
        for m in self.models:
            status = "PASS" if m.match and not m.flagged else "FAIL"
            detail = m.flagged or m.error or (
                f"unmatched={m.unmatched} missed={m.missed}" if not m.match else ""
            )
            lines.append(f"  {m.name}: {status} {detail}".rstrip())
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


class ELTGrader:
    """Grades one rollout's namespace (+ workspace) against a task's ground truth."""

    def __init__(self, task: ELTTask, destination: Destination, config: GraderConfig):
        self.task = task
        self.destination = destination
        self.config = config
        self._gt = {
            name: pd.read_csv(task.gt_dir / f"{name}.csv", dtype=str, keep_default_na=True)
            for name in task.model_names
        }

    async def grade(self, ns: Namespace) -> GradeResult:
        ok, total = await self._stage1(ns)
        results = [
            ModelResult(name, gold_columns=len(self._gt[name].columns))
            for name in self.task.model_names
        ]
        notes: list[str] = []

        if self.config.mode == "pipeline":
            await self._run_pipeline_models(ns, results, notes)

        outputs: dict[str, pd.DataFrame] = {}
        existing = {t.lower() for t in await self.destination.list_tables(ns)}
        for res in results:
            if res.error is not None:
                continue
            if res.name not in existing:
                res.error = "model table not found"
                continue
            res.exists = True
            try:
                outputs[res.name] = await self.destination.fetch_table(ns, res.name)
                compare_model(
                    self._gt[res.name],
                    as_csv_strings(outputs[res.name]),
                    self.task.sort_keys.get(res.name, []),
                    res,
                )
            except Exception as e:  # noqa: BLE001 - any failure is a failed model
                res.error = f"comparison failed: {e}"

        if self.config.mode == "pipeline" and self.config.perturbation_check:
            await self._perturbation_check(ns, results, outputs, notes)

        return GradeResult(
            reward=self._reward(ok, total, results),
            stage1_tables_ok=ok,
            stage1_tables_total=total,
            models=results,
            notes=notes,
        )

    async def _stage1(self, ns: Namespace) -> tuple[int, int]:
        existing = {t.lower() for t in await self.destination.list_tables(ns)}
        ok = 0
        for table, expected in self.task.expected_rows.items():
            if table.lower() in existing and await self.destination.row_count(ns, table) == expected:
                ok += 1
        return ok, len(self.task.expected_rows)

    async def _run_pipeline_models(
        self, ns: Namespace, results: list[ModelResult], notes: list[str]
    ) -> None:
        """Compile and execute the submitted model files into the namespace."""
        by_name = {r.name: r for r in results}
        # Anything the agent created by hand under a model's name is discarded:
        # only tables produced by the submitted pipeline are graded.
        for name in by_name:
            await self.destination.drop_relation(ns, name)

        models = find_model_files(ns.workspace)
        for name, res in by_name.items():
            if name not in models:
                res.error = "no model file submitted"
        try:
            order = execution_order(models, [n for n in by_name if n in models])
        except ValueError as e:
            notes.append(str(e))
            for res in results:
                res.error = res.error or str(e)
            return

        upstream = list(self.task.expected_rows) + list(models)
        for name in order:
            model = models[name]
            res = by_name.get(name)
            if res is not None and not references_any(model.raw_sql, upstream):
                res.error = "model does not read from any source table or model"
                res.flagged = "constant model (no source reference)"
                continue
            try:
                sql = compile_model(ns.workspace, model, lambda t: self.destination.qualify(ns, t))
                await self.destination.materialize(ns, name, sql)
            except Exception as e:  # noqa: BLE001
                msg = f"model failed to execute: {str(e).splitlines()[0][:300]}"
                if res is not None:
                    res.error = msg
                else:
                    notes.append(f"{name}: {msg}")

    async def _perturbation_check(
        self,
        ns: Namespace,
        results: list[ModelResult],
        outputs: dict[str, pd.DataFrame],
        notes: list[str],
    ) -> None:
        """Delete ~20% of every raw table, re-run the models, and revoke credit
        from any model whose output is byte-identical (it ignores its inputs).

        Runs last because it mutates the namespace, which is torn down anyway.
        """
        credited = [r for r in results if r.name in outputs and r.matched and not r.flagged]
        if not credited:
            return
        existing = {t.lower() for t in await self.destination.list_tables(ns)}
        for table in self.task.expected_rows:
            if table.lower() in existing:
                await self.destination.perturb(ns, table)
        await self._run_pipeline_models(ns, [ModelResult(r.name) for r in credited], notes)
        for res in credited:
            try:
                after = await self.destination.fetch_table(ns, res.name)
            except Exception:  # noqa: BLE001 - failing on smaller input is not hardcoding
                continue
            before = as_csv_strings(outputs[res.name])
            if len(before) and as_csv_strings(after).equals(before):
                res.flagged = "output unchanged when source data was perturbed"
        if any(r.flagged for r in credited):
            notes.append("perturbation check revoked credit from input-insensitive models")

    def _reward(self, ok: int, total: int, results: list[ModelResult]) -> float:
        w = self.config.weights
        n_models = max(len(results), 1)
        gold_cols = sum(r.gold_columns for r in results)
        ok_cols = sum(len(r.matched) for r in results if not r.flagged)
        models_ok = sum(r.match and not r.flagged for r in results)
        reward = (
            w.stage1 * ok / max(total, 1)
            + w.models * models_ok / n_models
            + w.columns * ok_cols / max(gold_cols, 1)
        )
        return float(min(max(reward, 0.0), 1.0))
