"""Run one group of rollouts with a Tinker-hosted policy and report rewards.

Credentialed run on an official ELT-Bench task (Snowflake + docker sandbox):

    python scripts/run_rollout.py task_source=eltbench only=books \
        eltbench_root=/path/ELT-Bench gt_root=/path/ELT-Bench/ground_truth/gt_snowflake \
        destination=snowflake snowflake_credential=/path/snowflake_admin.json \
        sandbox=docker group_size=1 max_turns=100 max_trajectory_tokens=131072

Local toy task (only TINKER_API_KEY needed):

    python scripts/run_rollout.py group_size=2

Prints each trajectory's reward and grading metrics and writes the decoded
transcripts to ``out_dir``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

from elt_rl.destinations import DestinationSpec
from elt_rl.env import ELTEnvConfig, ELTEnvGroupBuilder
from elt_rl.train import CLIConfig, load_tasks


@chz.chz
class RolloutConfig(CLIConfig):
    group_size: int = 1
    # Sample from a trained checkpoint (tinker://...) instead of the base model.
    sampler_path: str | None = None
    out_dir: str = "rollouts"
    # The model's sampling context window; caps max_tokens per request so that
    # prompt + max_tokens never exceeds it. Defaults to max_trajectory_tokens.
    context_window: int | None = None


async def run(cfg: RolloutConfig) -> None:
    task = load_tasks(cfg)[0]
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    builder = ELTEnvGroupBuilder(
        task=task,
        destination=DestinationSpec(cfg.destination, credential_path=cfg.snowflake_credential),
        config=ELTEnvConfig(
            sandbox=cfg.sandbox,
            el_mode=cfg.el_mode,
            template=cfg.template,
            grade_mode=cfg.grade_mode,
            perturbation_check=cfg.perturbation_check,
            max_turns=cfg.max_turns,
            max_trajectory_tokens=cfg.max_trajectory_tokens,
            max_generation_tokens=cfg.max_tokens,
        ),
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        group_size=cfg.group_size,
    )

    service = tinker.ServiceClient()
    if cfg.sampler_path:
        client = service.create_sampling_client(model_path=cfg.sampler_path)
    else:
        client = service.create_sampling_client(base_model=cfg.model_name)
    client = client.result() if hasattr(client, "result") else client
    policy = TinkerTokenCompleter(
        client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        context_window=cfg.context_window or cfg.max_trajectory_tokens,
    )

    group = await do_group_rollout(builder, policy)

    tokenizer = tokenizer_utils.get_tokenizer(cfg.model_name)
    out_dir = Path(cfg.out_dir) / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = group.get_total_rewards()
    for i, traj in enumerate(group.trajectories_G):
        metrics: dict[str, float] = {}
        for t in traj.transitions:
            metrics.update(t.metrics)
        last = traj.transitions[-1]
        transcript = tokenizer.decode(last.ob.to_ints() + list(last.ac.tokens))
        print(f"[{task.name} #{i}] reward={rewards[i]:.3f} turns={len(traj.transitions)} "
              f"stop={traj.stop_reason}")
        print(json.dumps({k: v for k, v in metrics.items() if k.startswith("elt/")}, indent=2))
        (out_dir / f"rollout_{i}.json").write_text(
            json.dumps(
                {"task": task.name, "reward": rewards[i], "metrics": metrics,
                 "turns": len(traj.transitions), "stop_reason": traj.stop_reason,
                 "transcript": transcript},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
    print(f"transcripts written to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(chz.entrypoint(RolloutConfig)))
