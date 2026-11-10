"""ELT-Bench RL environment on top of tinker-cookbook.

Structure (mirrors ``tinker_cookbook/recipes/harbor_rl``):

* ``ELTRollout``: every resource one episode owns (workspace copy of the
  project base, isolated warehouse namespace, sandbox, tools, grader binding).
* ``AgentToolMessageEnv`` (cookbook): the multi-turn tool loop; calls our
  ``reward_fn`` once at episode end, which grades the warehouse by execution.
* ``ELTEnvGroupBuilder``: G rollouts of the same task (one GRPO group).
  Rewards are centered within the group by the cookbook training loop.
* ``ELTDataset`` / ``ELTDatasetBuilder``: batches of groups for ``rl.train``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chz
import yaml
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import AgentToolMessageEnv

from elt_rl.airbyte import AirbyteWorkspaces
from elt_rl.destinations import Destination, DestinationSpec, Namespace
from elt_rl.grading import ELTGrader, GradeResult, GraderConfig
from elt_rl.prompts import SYSTEM_PROMPT, initial_instruction
from elt_rl.sandbox import DockerSandbox, WorkspaceSandbox
from elt_rl.task import ELTTask, materialize_workspace
from elt_rl.tools import ELTTools

logger = logging.getLogger(__name__)

PRELOADED_NOTE = (
    "\nNOTE: Stage 1 has already been completed for you: every source table is loaded into the "
    "destination schema. Do not load them again; go straight to Stage 2."
)


@dataclass(frozen=True)
class ELTEnvConfig:
    """Everything about an ELT episode except the task and the warehouse."""

    sandbox: str = "local"  # "local" | "docker"
    # "agent": the policy must extract & load. "preloaded": raw tables are cloned
    # from a template (transform-only curriculum, and much cheaper rollouts).
    el_mode: str = "agent"
    # Template for el_mode="preloaded": duckdb file path or Snowflake database
    # name; "{task}" is substituted. Built automatically for local duckdb tasks.
    template: str | None = None
    grade_mode: str = "pipeline"  # "pipeline" | "warehouse"
    perturbation_check: bool = True
    max_turns: int = 30
    allow_local_bash: bool = False
    docker_image: str = "elt-swe"
    docker_network: str = "elt-docker_elt_network"
    command_timeout: int = 600
    workspace_root: str | None = None
    # Malformed tool calls: retry up to N times with a corrective message.
    parse_retries: int = 2
    parse_error_penalty: float = 0.05
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    # A turn that hits max_generation_tokens is kept (truncated) and the episode
    # continues, instead of ending it; long "thinking" turns are common here.
    terminate_on_length: bool = False
    # Airbyte API root reachable from the host. ELT-Bench rollouts get their own
    # Airbyte workspace through it (None: share the task's workspace, no cleanup).
    airbyte_api_url: str | None = "http://localhost:8000/api"
    # Before grading, wait up to this long for the rollout's in-flight Airbyte
    # syncs, so a correct pipeline is not scored on Airbyte's copy speed.
    sync_settle_seconds: int = 900
    # Local Terraform provider mirror for docker sandboxes (used if it exists).
    terraform_mirror: str | None = "~/.cache/elt_rl/terraform-mirror"


_TEMPLATE_CACHE: dict[tuple[str, str], str] = {}


def _local_template(task: ELTTask, root: Path) -> str:
    """Build (once per process) the reference raw-table database for a local task."""
    from elt_rl.destinations.duckdb_dest import build_flat_file_template

    key = (task.name, str(root))
    if key not in _TEMPLATE_CACHE:
        path = root / f"_template_{task.name}.duckdb"
        _TEMPLATE_CACHE[key] = str(build_flat_file_template(task, path))
    return _TEMPLATE_CACHE[key]


class ELTRollout:
    """Resources for one episode; ``close()`` releases all of them."""

    def __init__(
        self,
        task: ELTTask,
        destination: Destination,
        grader: ELTGrader,
        ns: Namespace,
        sandbox: SandboxInterface,
        tools: ELTTools,
        instruction: str,
        airbyte: tuple[AirbyteWorkspaces, str] | None = None,
    ):
        self.task = task
        self.destination = destination
        self.grader = grader
        self.ns = ns
        self.sandbox = sandbox
        self.tools = tools
        self.instruction = instruction
        self.airbyte = airbyte  # (client, this rollout's workspace id)
        self.sync_settle_seconds = 0
        self.last_grade: GradeResult | None = None
        self.graded = False
        self._closed = False

    @classmethod
    async def open(
        cls,
        task: ELTTask,
        destination: Destination,
        grader: ELTGrader,
        config: ELTEnvConfig,
        rollout_id: str | None = None,
    ) -> ELTRollout:
        rollout_id = rollout_id or uuid.uuid4().hex[:10]
        root = Path(config.workspace_root or tempfile.gettempdir()) / "elt_rl_workspaces"
        root.mkdir(parents=True, exist_ok=True)
        workspace = materialize_workspace(task, root / f"{task.name}_{rollout_id}")

        if config.el_mode == "preloaded":
            if config.template:
                template = config.template.format(task=task.name)
            elif destination.name == "duckdb" and task.kind == "local":
                template = _local_template(task, root)
            else:
                raise ValueError("el_mode='preloaded' needs ELTEnvConfig.template for this destination")
            ns = await destination.clone_from_template(task, template, rollout_id, workspace)
        else:
            ns = await destination.provision(task, rollout_id, workspace)

        airbyte: tuple[AirbyteWorkspaces, str] | None = None
        try:
            # Point every config consumer (agent, Airbyte, dbt) at this rollout's namespace.
            config_path = workspace / "config.yaml"
            task_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            airbyte_cfg = (task_config.get("Airbyte") or {}).get("config")
            if task.kind == "eltbench" and config.airbyte_api_url and airbyte_cfg:
                client = AirbyteWorkspaces(
                    config.airbyte_api_url, airbyte_cfg["client_id"], airbyte_cfg["client_secret"]
                )
                grants = [v for k, v in airbyte_cfg.items() if k.endswith("definition_id")]
                workspace_id = await client.create(f"eltrl-{task.name}-{rollout_id}", grants)
                airbyte = (client, workspace_id)
                airbyte_cfg["workspace_id"] = workspace_id
            if task.kind == "local":
                # Local sources are read by the warehouse itself, which only accepts
                # absolute paths inside the workspace.
                for source in task_config.get("flat_files", []):
                    source["path"] = (workspace / source["path"]).resolve().as_posix()
            config_path.write_text(
                yaml.safe_dump(destination.agent_config(ns, task_config), sort_keys=False),
                encoding="utf-8",
            )

            if config.sandbox == "docker":
                sandbox: SandboxInterface = await DockerSandbox.create(
                    workspace,
                    name=f"eltrl-{task.name}-{rollout_id}",
                    image=config.docker_image,
                    network=config.docker_network,
                    teardown_commands=[
                        "cd /workspace/elt && terraform destroy -parallelism=1 -auto-approve"
                        " >/dev/null 2>&1 || true"
                    ],
                    terraform_mirror=(
                        Path(config.terraform_mirror).expanduser() if config.terraform_mirror else None
                    ),
                )
                tools = ELTTools(
                    sandbox, destination, ns,
                    writable_dirs=("elt",), allow_bash=True, allow_run_models=False,
                    command_timeout=config.command_timeout,
                )
            else:
                sandbox = WorkspaceSandbox(workspace)
                tools = ELTTools(
                    sandbox, destination, ns,
                    writable_dirs=("models",), allow_bash=config.allow_local_bash,
                    allow_run_models=True, command_timeout=config.command_timeout,
                )
        except Exception:
            if airbyte is not None:
                await airbyte[0].delete(airbyte[1])
            await destination.teardown(ns)
            shutil.rmtree(workspace, ignore_errors=True)
            raise

        instruction = initial_instruction(task, destination.display_name, destination.name, ns.database)
        if config.el_mode == "preloaded":
            instruction += PRELOADED_NOTE
        rollout = cls(task, destination, grader, ns, sandbox, tools, instruction, airbyte)
        rollout.sync_settle_seconds = config.sync_settle_seconds
        return rollout

    def initial_messages(self, renderer: Renderer | None) -> list[Message]:
        user: Message = {"role": "user", "content": self.instruction}
        if renderer is None:
            return [{"role": "system", "content": SYSTEM_PROMPT}, user]
        specs = [t.to_spec() for t in self.tools.tool_list()]
        return renderer.create_conversation_prefix_with_tools(tools=specs, system_prompt=SYSTEM_PROMPT) + [user]

    def message_env(self, initial_messages: list[Message], config: ELTEnvConfig) -> AgentToolMessageEnv:
        return ELTMessageEnv(
            tools=self.tools.tool_list(),
            initial_messages=initial_messages,
            max_turns=config.max_turns,
            reward_fn=self.reward_fn,
            tool_execution="sequential",  # tools share the warehouse namespace
            parse_error_policy=parse_error_policy(config),
        )

    async def reward_fn(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the final warehouse state (called once, at episode end)."""
        self.graded = True
        settle_metrics: dict[str, float] = {}
        if self.airbyte is not None and self.sync_settle_seconds > 0:
            client, workspace_id = self.airbyte
            try:
                unsettled = await client.wait_for_syncs(workspace_id, self.sync_settle_seconds)
                settle_metrics["elt/syncs_unsettled_at_grading"] = float(unsettled)
            except Exception as e:  # noqa: BLE001 - grade anyway
                logger.warning("could not wait for syncs in %s: %s", workspace_id, e)
        try:
            grade = await self.grader.grade(self.ns)
        except Exception as e:  # noqa: BLE001 - infrastructure failure, not policy failure
            logger.exception("grading failed for %s: %s", self.task.name, e)
            return 0.0, {"elt/grading_error": 1.0}
        self.last_grade = grade
        logger.info("[%s/%s] %s", self.task.name, self.ns.rollout_id, grade.summary())
        return grade.reward, {**grade.metrics(), **settle_metrics, "elt/submitted": float(self.tools.submitted)}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        steps = [self.sandbox.cleanup]
        if self.airbyte is not None:
            client, workspace_id = self.airbyte
            steps.append(lambda: client.delete(workspace_id))
        steps.append(lambda: self.destination.teardown(self.ns))
        for step in steps:
            try:
                await step()
            except Exception as e:  # noqa: BLE001 - cleanup must not mask results
                logger.warning("cleanup step failed for %s: %s", self.ns.rollout_id, e)


NO_TOOL_CALL_NUDGE = (
    "You did not call a tool. Continue building the pipeline with the available tools, "
    "or call `submit` if you are finished."
)


@dataclass
class ELTMessageEnv(AgentToolMessageEnv):
    """Tool loop that ends only on ``submit`` (or a limit), not on a turn without a tool call.

    The cookbook default treats a message without tool calls as "done". Long-horizon
    agents often end a turn with planning text and no action; ending the episode
    there would grade an unfinished pipeline. Like the official ELT-Bench harness
    (which stops on an explicit terminate action), the policy is nudged instead.
    """

    async def step(self, message: Message) -> MessageStepResult:
        no_call = not message.get("tool_calls") and not message.get("unparsed_tool_calls")
        if no_call and self._turn_count + 1 < self.max_turns:
            self._turn_count += 1
            self.history.append(message)
            self.history.append({"role": "user", "content": NO_TOOL_CALL_NUDGE})
            return MessageStepResult(
                reward=0.0, episode_done=False, next_messages=self.history,
                metrics={"elt/no_tool_call_turn": 1.0},
            )
        return await super().step(message)


class ELTEnv(EnvFromMessageEnv):
    """Token-level env that grades the warehouse however the episode ends.

    The cookbook's limit paths (generation hits max_tokens, context overflow,
    parse-error budget exhausted) end the episode with a flat penalty and never
    call ``reward_fn``. Here the execution-based grade is always added, so every
    episode's reward reflects the warehouse state the policy produced.
    """

    def __init__(self, rollout: ELTRollout, **kwargs):
        super().__init__(**kwargs)
        self.rollout = rollout

    async def step(self, action, *, extra=None):
        result = await super().step(action, extra=extra)
        if result.episode_done and not self.rollout.graded:
            reward, metrics = await self.rollout.reward_fn([])
            result.reward += reward
            result.metrics = {**result.metrics, **metrics, "elt/graded_after_limit": 1.0}
        return result


def parse_error_policy(config: ELTEnvConfig) -> ParseErrorPolicy:
    return ParseErrorPolicy(
        max_consecutive=config.parse_retries,
        penalty_per_error=config.parse_error_penalty,
        terminal_reward=0.0,
    )


class ELTEnvGroupBuilder(EnvGroupBuilder):
    """G independent rollouts of one task (each with its own namespace)."""

    def __init__(
        self,
        task: ELTTask,
        destination: DestinationSpec,
        config: ELTEnvConfig,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
    ):
        self.task = task
        self.destination_spec = destination
        self.config = config
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self._destination: Destination | None = None
        self._rollouts: list[ELTRollout] = []

    def __getstate__(self) -> dict:
        # Live connections/sandboxes are created in make_envs; never pickle them.
        state = self.__dict__.copy()
        state["_destination"] = None
        state["_rollouts"] = []
        return state

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        self._destination = self.destination_spec.build()
        grader = ELTGrader(
            self.task,
            self._destination,
            GraderConfig(mode=self.config.grade_mode, perturbation_check=self.config.perturbation_check),
        )
        opened = await asyncio.gather(
            *[
                ELTRollout.open(self.task, self._destination, grader, self.config)
                for _ in range(self.group_size)
            ],
            return_exceptions=True,
        )
        self._rollouts = [r for r in opened if isinstance(r, ELTRollout)]
        errors = [r for r in opened if isinstance(r, BaseException)]
        if errors:
            await self.cleanup()
            raise errors[0]
        envs: list[Env] = []
        for rollout in self._rollouts:
            message_env = rollout.message_env(rollout.initial_messages(renderer), self.config)
            env = ELTEnv(
                rollout,
                renderer=renderer,
                message_env=message_env,
                max_trajectory_tokens=self.config.max_trajectory_tokens,
                max_generation_tokens=self.config.max_generation_tokens,
                context_overflow_reward=self.config.context_overflow_reward,
                terminate_on_length=self.config.terminate_on_length,
            )
            env.set_parse_error_policy(parse_error_policy(self.config))
            envs.append(env)
        return envs

    async def cleanup(self) -> None:
        await asyncio.gather(*[r.close() for r in self._rollouts], return_exceptions=True)
        self._rollouts = []
        if self._destination is not None:
            self._destination.close()
            self._destination = None

    def logging_tags(self) -> list[str]:
        return ["elt", self.task.name]


class ELTDataset(RLDataset):
    def __init__(self, builders: list[ELTEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.builders[index * self.batch_size : (index + 1) * self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    tasks: list[ELTTask]
    destination: DestinationSpec
    env_config: ELTEnvConfig
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str | None = None
    # Repeat the task list so small task sets still give several training steps.
    repeats: int = 1

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        builders = [
            ELTEnvGroupBuilder(
                task=task,
                destination=self.destination,
                config=self.env_config,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
            )
            for _ in range(self.repeats)
            for task in self.tasks
        ]
        return ELTDataset(builders, self.batch_size), None
