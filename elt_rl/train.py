"""RL training on ELT tasks with Tinker (GRPO-style group-centered advantages).

Local toy task, DuckDB, one optimization step (needs only TINKER_API_KEY):

    python -m elt_rl.train max_steps=1 group_size=4 groups_per_batch=1

Official ELT-Bench task on Snowflake in the ELT-Bench docker network:

    python -m elt_rl.train task_source=eltbench only=books destination=snowflake \
        snowflake_credential=/path/snowflake_admin.json sandbox=docker \
        eltbench_root=/path/ELT-Bench gt_root=/path/gt_snowflake max_steps=1
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import Config, main

from elt_rl.destinations import DestinationSpec
from elt_rl.env import ELTDatasetBuilder, ELTEnvConfig
from elt_rl.task import ELTTask, load_eltbench_tasks, load_local_tasks

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # Tasks
    task_source: str = "local"  # "local" | "eltbench"
    tasks_root: str = "tasks"
    eltbench_root: str | None = None  # ELT-Bench checkout (inputs/ and evaluation/)
    gt_root: str | None = None  # e.g. <ELT-Bench>/ground_truth/gt_snowflake
    only: str | None = None  # comma-separated task names

    # Environment
    destination: str = "duckdb"
    snowflake_credential: str | None = None
    sandbox: str = "local"
    el_mode: str = "agent"
    template: str | None = None
    grade_mode: str = "pipeline"
    perturbation_check: bool = True
    max_turns: int = 30
    max_trajectory_tokens: int = 32 * 1024

    # Model / optimization
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str | None = None
    lora_rank: int = 32
    learning_rate: float = 4e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    group_size: int = 4
    groups_per_batch: int = 2
    repeats: int = 1
    kl_penalty_coef: float = 0.0
    max_steps: int | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 10
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def load_tasks(cfg: CLIConfig) -> list[ELTTask]:
    only = cfg.only.split(",") if cfg.only else None
    if cfg.task_source == "local":
        tasks = load_local_tasks(cfg.tasks_root, only=only)
    elif cfg.task_source == "eltbench":
        if not cfg.eltbench_root or not cfg.gt_root:
            raise ValueError("task_source=eltbench needs eltbench_root and gt_root")
        tasks = load_eltbench_tasks(
            f"{cfg.eltbench_root}/inputs", f"{cfg.eltbench_root}/evaluation", cfg.gt_root, only=only
        )
    else:
        raise ValueError(f"unknown task_source {cfg.task_source!r}")
    if not tasks:
        raise ValueError("no tasks loaded")
    return tasks


def build_config(cfg: CLIConfig) -> Config:
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    run_name = (
        f"elt-{cfg.task_source}-{cfg.destination}-{cfg.model_name.replace('/', '-')}-"
        f"g{cfg.group_size}b{cfg.groups_per_batch}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    dataset_builder = ELTDatasetBuilder(
        tasks=load_tasks(cfg),
        destination=DestinationSpec(cfg.destination, credential_path=cfg.snowflake_credential),
        env_config=ELTEnvConfig(
            sandbox=cfg.sandbox,
            el_mode=cfg.el_mode,
            template=cfg.template,
            grade_mode=cfg.grade_mode,
            perturbation_check=cfg.perturbation_check,
            max_turns=cfg.max_turns,
            max_trajectory_tokens=cfg.max_trajectory_tokens,
            max_generation_tokens=cfg.max_tokens,
        ),
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        repeats=cfg.repeats,
    )
    log_path = cfg.log_path or cli_utils.runs_path("elt_rl", run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    return Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        recipe_name="elt_bench_rl",
        renderer_name=renderer_name,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        log_path=log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        save_every=cfg.save_every,
        eval_every=0,
        max_steps=cfg.max_steps,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(build_config(chz.entrypoint(CLIConfig))))
