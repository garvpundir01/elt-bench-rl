"""Shared fixtures: a scripted policy that drives the real cookbook env loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tinker_cookbook.renderers.base import Message, ToolCall

from elt_rl.destinations.duckdb_dest import DuckDBDestination
from elt_rl.env import ELTEnvConfig, ELTRollout
from elt_rl.grading import ELTGrader, GraderConfig
from elt_rl.task import ELTTask, load_local_tasks

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "tasks" / "toy_shop" / "grader" / "gold"

def source_path(rollout: ELTRollout, file: str) -> str:
    return (rollout.ns.workspace / "sources" / file).resolve().as_posix()


def load_calls(rollout: ELTRollout) -> list[tuple[str, dict]]:
    """Stage 1, as a policy would do it from the paths in config.yaml."""
    return [
        ("sql", {"query": f"CREATE TABLE elt.customers AS SELECT * FROM read_csv_auto('{source_path(rollout, 'customers.csv')}')"}),
        ("sql", {"query": f"CREATE TABLE elt.products AS SELECT * FROM read_csv_auto('{source_path(rollout, 'products.csv')}')"}),
        ("sql", {"query": f"CREATE TABLE elt.orders AS SELECT * FROM read_json_auto('{source_path(rollout, 'orders.jsonl')}')"}),
    ]


def gold_sql(model: str) -> str:
    return (GOLD / f"{model}.sql").read_text()


def oracle_calls(rollout: ELTRollout) -> list[tuple[str, dict]]:
    return [
        ("read_file", {"path": "config.yaml"}),
        *load_calls(rollout),
        ("write_file", {"path": "models/customer_summary.sql", "content": gold_sql("customer_summary")}),
        ("write_file", {"path": "models/category_sales.sql", "content": gold_sql("category_sales")}),
        ("run_models", {}),
        ("submit", {"summary": "done"}),
    ]


def assistant(name: str, args: dict, i: int) -> Message:
    return {
        "role": "assistant",
        "content": f"Calling {name}.",
        "tool_calls": [
            ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)), id=f"call_{i}")
        ],
    }


class EpisodeResult:
    def __init__(self, reward: float, metrics: dict, history: list[Message], rollout: ELTRollout):
        self.reward, self.metrics, self.history, self.rollout = reward, metrics, history, rollout

    def tool_outputs(self) -> list[str]:
        return [str(m["content"]) for m in self.history if m["role"] == "tool"]


async def run_episode(rollout: ELTRollout, config: ELTEnvConfig, calls: list[tuple[str, dict]]) -> EpisodeResult:
    """Play scripted tool calls through AgentToolMessageEnv until it ends."""
    env = rollout.message_env(rollout.initial_messages(renderer=None), config)
    await env.initial_observation()
    total, metrics = 0.0, {}
    for i, (name, args) in enumerate(calls):
        step = await env.step(assistant(name, args, i))
        total += step.reward
        metrics.update(step.metrics)
        if step.episode_done:
            break
    else:
        step = await env.step({"role": "assistant", "content": "I am done."})
        total += step.reward
        metrics.update(step.metrics)
    return EpisodeResult(total, metrics, env.history, rollout)


@pytest.fixture
def task() -> ELTTask:
    return load_local_tasks(REPO / "tasks", only=["toy_shop"])[0]


@pytest.fixture
def config(tmp_path: Path) -> ELTEnvConfig:
    return ELTEnvConfig(workspace_root=str(tmp_path / "ws"))


@pytest.fixture
async def destination(tmp_path: Path):
    dest = DuckDBDestination(root=tmp_path / "db")
    yield dest
    dest.close()


@pytest.fixture
async def make_rollout(task, destination, config):
    opened: list[ELTRollout] = []

    async def _make(cfg: ELTEnvConfig | None = None, grader_config: GraderConfig | None = None) -> ELTRollout:
        cfg = cfg or config
        grader = ELTGrader(task, destination, grader_config or GraderConfig(mode=cfg.grade_mode))
        rollout = await ELTRollout.open(task, destination, grader, cfg)
        opened.append(rollout)
        return rollout

    yield _make
    for rollout in opened:
        await rollout.close()
