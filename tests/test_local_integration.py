"""Credential-free end-to-end tests: toy task + DuckDB + scripted policies.

Each test plays a scripted trajectory through the real cookbook
``AgentToolMessageEnv`` and checks the execution-derived reward.
"""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO, assistant, gold_sql, load_calls, oracle_calls, run_episode, source_path
from elt_rl.destinations import DestinationSpec
from elt_rl.env import ELTEnv, ELTEnvConfig, ELTEnvGroupBuilder
from elt_rl.grading import GraderConfig


async def test_oracle_pipeline_gets_full_reward(make_rollout, config):
    rollout = await make_rollout()
    result = await run_episode(rollout, config, oracle_calls(rollout))
    assert result.reward == pytest.approx(1.0)
    assert result.metrics["elt/stage1_pass"] == 1.0
    assert result.metrics["elt/models_correct_frac"] == 1.0
    assert result.metrics["elt/submitted"] == 1.0
    assert "customer_summary: OK (8 rows)" in result.tool_outputs()[-2]


async def test_partial_pipeline_gets_partial_credit(make_rollout, config):
    rollout = await make_rollout()
    wrong = gold_sql("category_sales").replace("ROUND(SUM(pr.rev), 2) AS revenue", "SUM(pr.units) AS revenue")
    calls = [
        *load_calls(rollout),
        ("write_file", {"path": "models/customer_summary.sql", "content": gold_sql("customer_summary")}),
        ("write_file", {"path": "models/category_sales.sql", "content": wrong}),
        ("submit", {}),
    ]
    result = await run_episode(rollout, config, calls)
    # stage1 0.2 + 1/2 models * 0.6 + 11/12 columns * 0.2
    assert result.reward == pytest.approx(0.2 + 0.3 + 0.2 * 11 / 12)
    grade = rollout.last_grade
    by_name = {m.name: m for m in grade.models}
    assert by_name["category_sales"].unmatched == ["revenue"]


async def test_no_work_gets_zero(make_rollout, config):
    rollout = await make_rollout()
    result = await run_episode(rollout, config, [("submit", {})])
    assert result.reward == 0.0
    assert result.metrics["elt/models_missing"] == 2.0


async def test_loading_twice_breaks_stage1(make_rollout, config):
    """full_refresh_append semantics: re-running a sync duplicates rows (paper §4.2)."""
    rollout = await make_rollout()
    calls = [
        *load_calls(rollout),
        ("sql", {"query": f"INSERT INTO elt.orders SELECT * FROM read_json_auto('{source_path(rollout, 'orders.jsonl')}')"}),
        ("submit", {}),
    ]
    await run_episode(rollout, config, calls)
    assert rollout.last_grade.stage1_tables_ok == 2


async def test_hand_built_tables_do_not_count_in_pipeline_mode(make_rollout, config):
    """Only tables produced by submitted model files are graded."""
    hand_built = "CREATE TABLE elt.customer_summary AS " + gold_sql("customer_summary").replace(
        "{{ source('elt', 'orders') }}", "elt.orders").replace(
        "{{ source('elt', 'products') }}", "elt.products").replace(
        "{{ source('elt', 'customers') }}", "elt.customers")

    def calls(rollout):
        return [*load_calls(rollout), ("sql", {"query": hand_built}), ("submit", {})]

    rollout = await make_rollout()
    result = await run_episode(rollout, config, calls(rollout))
    assert result.metrics["elt/models_correct_frac"] == 0.0

    # The official evaluator grades leftover warehouse state, so it would count it.
    parity_cfg = ELTEnvConfig(workspace_root=config.workspace_root, grade_mode="warehouse")
    rollout2 = await make_rollout(parity_cfg)
    result2 = await run_episode(rollout2, parity_cfg, calls(rollout2))
    assert result2.metrics["elt/models_correct_frac"] == 0.5


def _literal_select(gt_csv: Path) -> str:
    df = pd.read_csv(gt_csv)
    rows = []
    for _, r in df.iterrows():
        vals = ["NULL" if pd.isna(v) else (repr(v) if isinstance(v, str) else str(v)) for v in r]
        rows.append("(" + ", ".join(vals) + ")")
    cols = ", ".join(df.columns)
    return f"SELECT * FROM (VALUES {', '.join(rows)}) AS t({cols})"


async def test_hardcoded_outputs_are_caught(make_rollout, config):
    """Copying the expected answer as literals earns no transform credit."""
    gt_dir = REPO / "tasks" / "toy_shop" / "grader" / "gt"
    constant = _literal_select(gt_dir / "customer_summary.csv")
    # Disguised: mentions a source table but ignores its contents.
    disguised = _literal_select(gt_dir / "category_sales.csv") + " WHERE (SELECT COUNT(*) FROM elt.products) >= 0"
    def calls(rollout):
        return [
            *load_calls(rollout),
            ("write_file", {"path": "models/customer_summary.sql", "content": constant}),
            ("write_file", {"path": "models/category_sales.sql", "content": disguised}),
            ("submit", {}),
        ]

    rollout = await make_rollout()
    result = await run_episode(rollout, config, calls(rollout))
    flags = {m.name: m.flagged for m in rollout.last_grade.models}
    assert flags["customer_summary"] == "constant model (no source reference)"
    assert flags["category_sales"] == "output unchanged when source data was perturbed"
    assert result.reward == pytest.approx(0.2)  # stage-1 credit only

    # Without the perturbation check, the disguised model would have been paid.
    rollout2 = await make_rollout(grader_config=GraderConfig(perturbation_check=False))
    result2 = await run_episode(rollout2, config, calls(rollout2))
    assert result2.reward > 0.5


async def test_agent_cannot_reach_ground_truth_or_escape_workspace(make_rollout, config):
    gt = (REPO / "tasks" / "toy_shop" / "grader" / "gt" / "customer_summary.csv").resolve().as_posix()
    rollout = await make_rollout()
    # Starts inside the allowed workspace directory, then climbs out of it.
    escape = source_path(rollout, "customers.csv").rsplit("/", 1)[0] + "/" + "../" * 12 + gt.lstrip("/")
    calls = [
        ("sql", {"query": f"SELECT * FROM read_csv_auto('{gt}')"}),
        ("sql", {"query": f"SELECT * FROM read_csv_auto('{escape}')"}),
        ("sql", {"query": "SET enable_external_access = true"}),
        ("sql", {"query": f"ATTACH '{gt}.duckdb' AS other"}),
        ("read_file", {"path": "../../../grader/gt/customer_summary.csv"}),
        ("read_file", {"path": gt}),
        ("write_file", {"path": "config.yaml", "content": "hacked"}),
        ("write_file", {"path": "models/../config.yaml", "content": "hacked"}),
        ("submit", {}),
    ]
    result = await run_episode(rollout, config, calls)
    outputs = result.tool_outputs()
    assert all(("error" in o.lower()) for o in outputs[:8]), outputs
    assert not list(rollout.ns.workspace.rglob("gt"))  # grader assets never copied in
    assert "hacked" not in (rollout.ns.workspace / "config.yaml").read_text()


async def test_concurrent_rollouts_are_isolated(make_rollout, config):
    a, b = await asyncio.gather(make_rollout(), make_rollout())
    assert a.ns.database != b.ns.database and a.ns.workspace != b.ns.workspace
    ra, rb = await asyncio.gather(
        run_episode(a, config, oracle_calls(a)), run_episode(b, config, [("submit", {})])
    )
    assert ra.reward == pytest.approx(1.0)
    assert rb.reward == 0.0


async def test_preloaded_mode_skips_extraction(make_rollout, tmp_path):
    cfg = ELTEnvConfig(workspace_root=str(tmp_path / "ws"), el_mode="preloaded")
    rollout = await make_rollout(cfg)
    assert "already been completed" in rollout.instruction
    calls = [
        ("write_file", {"path": "models/customer_summary.sql", "content": gold_sql("customer_summary")}),
        ("write_file", {"path": "models/category_sales.sql", "content": gold_sql("category_sales")}),
        ("submit", {}),
    ]
    result = await run_episode(rollout, cfg, calls)
    assert result.reward == pytest.approx(1.0)


async def test_close_releases_namespace_and_workspace(make_rollout, config):
    rollout = await make_rollout()
    await run_episode(rollout, config, load_calls(rollout))
    db, ws = Path(rollout.ns.database), rollout.ns.workspace
    assert db.exists() and ws.exists()
    await rollout.close()
    assert not db.exists() and not ws.exists()


async def test_turn_without_tool_call_is_nudged_not_terminal(make_rollout, config):
    """Planning text without an action must not end (and grade) the episode."""
    rollout = await make_rollout()
    env = rollout.message_env(rollout.initial_messages(None), config)
    await env.initial_observation()
    step = await env.step({"role": "assistant", "content": "Let me think about the plan..."})
    assert not step.episode_done and not rollout.graded
    assert step.metrics["elt/no_tool_call_turn"] == 1.0
    assert "did not call a tool" in step.next_messages[-1]["content"]
    for i, (name, args) in enumerate(oracle_calls(rollout)):
        step = await env.step(assistant(name, args, i))
    assert step.episode_done and step.reward == pytest.approx(1.0)


class _StubRenderer:
    def get_stop_sequences(self):
        return ["<stop>"]


async def test_token_limit_stop_is_still_graded_by_execution(make_rollout, config):
    """A turn hitting max_tokens ends the episode (when terminate_on_length=True),
    but the reward must still come from grading the warehouse, not only the penalty."""
    rollout = await make_rollout()
    for _, args in load_calls(rollout)[:2]:  # 2 of 3 raw tables loaded mid-episode
        await rollout.destination.execute(rollout.ns, args["query"], as_agent=True)
    assert not rollout.graded
    env = ELTEnv(
        rollout,
        renderer=_StubRenderer(),
        message_env=rollout.message_env(rollout.initial_messages(None), config),
        context_overflow_reward=-0.1,
        terminate_on_length=True,
    )
    result = await env.step([], extra={"stop_reason": "length"})
    assert result.episode_done and rollout.graded
    assert result.metrics["elt/graded_after_limit"] == 1.0
    assert result.reward == pytest.approx(-0.1 + 0.2 * 2 / 3)


def test_group_builder_is_picklable(task):
    builder = ELTEnvGroupBuilder(
        task, DestinationSpec("duckdb"), ELTEnvConfig(), "Qwen/Qwen3-4B-Instruct-2507", None, 4
    )
    clone = pickle.loads(pickle.dumps(builder))
    assert clone.task.name == "toy_shop" and clone.group_size == 4
