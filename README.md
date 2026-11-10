# elt-bench-rl

An end-to-end RLVR environment for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench)
built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).
A policy inspects the task, extracts and loads the sources (Airbyte via Terraform), writes the
transformations (dbt), iterates on execution feedback and submits. The warehouse state produced by
the submitted pipeline is graded by execution against ELT-Bench ground truth, and the grade is the
RL reward.

**Design, results and the environment bugs we fixed: [docs/DESIGN.md](docs/DESIGN.md).**

*Implemented with the help of an AI coding assistant (Claude Code). I directed the design and the
experiments, set up and debugged the credentialed environment, and reviewed the results.*

```
elt_rl/
  task.py                 ELTTask + loaders (local tasks, official ELT-Bench bundles)
  destinations/           Destination interface; duckdb (local) and snowflake backends
  airbyte.py              per-rollout Airbyte workspaces (isolation + guaranteed cleanup)
  sandbox.py              WorkspaceSandbox (local) and DockerSandbox (ELT-Bench agent container)
  tools.py                list_files/read_file/write_file/sql/run_models/bash/submit (@tool)
  models.py               discover + compile dbt-style model files submitted by the policy
  grading.py              ELT-Bench stage-1/stage-2 checks, partial-credit reward, anti-hacking checks
  env.py                  ELTRollout, ELTEnv, ELTEnvGroupBuilder, ELTDataset(Builder)
  prompts.py              system prompt + task instructions
  train.py                Tinker RL training entrypoint (cookbook rl.train)
scripts/
  run_rollout.py          a group of rollouts with a Tinker-sampled policy
  run_reference.py        scripted reference solution through the real environment
  patch_eltbench.py       make an ELT-Bench checkout runnable on current Airbyte (see DESIGN §6)
  build_terraform_mirror.sh  local Terraform provider mirror for sandboxes
  make_local_gt.py        regenerate ground truth of local tasks
reference/books/          reference Terraform template and dbt models for ELT-Bench `books`
tasks/toy_shop/           credential-free task in ELT-Bench layout (workspace/ + grader/)
tests/                    credential-free tests
```

## Install

Linux or macOS, Python 3.11+ (tinker-cookbook's `tml-renderers` has no Windows wheels; on Windows
use WSL 2 or the Docker image):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,snowflake]"
# or: docker build -t elt-rl . && docker run --rm elt-rl      # runs the test suite
```

## 1. Credential-free local integration test

```bash
pytest -q
```

No accounts or network. The toy task `tasks/toy_shop` (CSV, CSV and nested-JSONL sources; two data
models with derived, aggregated, categorical and ranked columns) runs on DuckDB, with scripted
policies driving the real cookbook tool loop (tool calls → execution → grading → reward). The tests
cover full and partial credit, double-loading, hand-made answer tables, constant and input-ignoring
models (perturbation check), ground-truth access via SQL and file tools, isolation of concurrent
rollouts, `preloaded` mode, teardown, grading on token limits, and the no-tool-call nudge.

## 2. Credentialed rollout on an official ELT-Bench task

**One-time setup** (what we ran; WSL 2 Ubuntu on Windows works):

1. Follow ELT-Bench's README: Docker; Airbyte via `abctl local install`; import
   `setup/elt_snowflake.yaml` in Airbyte's Builder and publish it ("ELT Bench"); fill
   `setup/airbyte/airbyte_credential.json`; Snowflake via `setup/destination/setup.sql`; fill
   `setup/destination/snowflake_credential.json` with an admin service user (role SYSADMIN, `TYPE =
   LEGACY_SERVICE` so password login works without MFA; also set `AIRBYTE_USER` to `LEGACY_SERVICE`).
2. If publishing the connector fails with *"No declarative manifest image version found"*, Airbyte's
   runner registry could not update from Docker Hub; insert the missing major versions into Airbyte's
   `declarative_manifest_image_version` table (see DESIGN §6).
3. `python setup/write_config.py --destination snowflake`, then
   `python scripts/patch_eltbench.py --eltbench /path/ELT-Bench` (client-credential auth, connector
   pins, REST API pagination fix, image pins), then start the sources (`elt-docker/docker compose up -d`,
   load data as in `setup/elt_setup.sh`), `docker build -t elt-swe agents/SWE-agent/docker/elt-swe`,
   and `hf download tttjjj/elt_bench --repo-type dataset --include "gt_snowflake/**" --local-dir ./ground_truth`.
4. `bash scripts/build_terraform_mirror.sh` (once, on a network that can reach GitHub releases).

**Validate the environment with the reference solution** (should score 1.0):

```bash
python scripts/run_reference.py --task books
```

**Roll out a model** (needs `tinker auth login` or `TINKER_API_KEY`):

```bash
python scripts/run_rollout.py task_source=eltbench only=books \
    eltbench_root=/path/ELT-Bench gt_root=/path/ELT-Bench/ground_truth/gt_snowflake \
    destination=snowflake snowflake_credential=/path/ELT-Bench/setup/destination/snowflake_credential.json \
    sandbox=docker model_name=Qwen/Qwen3.6-35B-A3B:peft:262144 renderer_name=qwen3_5_disable_thinking \
    max_turns=100 max_trajectory_tokens=200000 context_window=262144
# transform-only curriculum: add  el_mode=preloaded template=TEMPLATE_BOOKS
```

Each rollout gets its own Snowflake database, Airbyte workspace, docker sandbox and workspace copy;
all are removed at teardown (including cancelling in-flight syncs).

## 3. Tinker RL optimization step

```bash
python -m elt_rl.train max_steps=1 group_size=4 groups_per_batch=1     # toy task, DuckDB
python -m elt_rl.train task_source=eltbench only=books destination=snowflake sandbox=docker \
    snowflake_credential=... eltbench_root=... gt_root=... el_mode=preloaded template=TEMPLATE_BOOKS max_steps=1
```

Each step samples `groups_per_batch` tasks × `group_size` rollouts on Tinker, grades every rollout by
execution, centers rewards within each group and takes one LoRA policy-gradient step. `elt/*`
metrics (reward, stage-1 fraction, models/columns correct, flagged models, unsettled syncs) are logged.

## Adding a destination

Subclass `elt_rl.destinations.base.Destination` (namespace lifecycle, `execute`, `qualify`, table
listing, perturbation SQL) and register it in `DestinationSpec.build`. Env, tools and grader are unchanged.

## Credits

Tasks, prompts and evaluation semantics come from ELT-Bench (Jin, Zhu, Kang, 2025); the RL framework
and environment abstractions from tinker-cookbook.
