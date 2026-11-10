# ELT-Bench RL Environment: Implementation Design

A Tinker RLVR recipe that trains a language model to build ELT pipelines end to end: inspect the
task, extract and load the sources, write the transformations, use execution feedback, and submit.
The warehouse state produced by the submitted pipeline is graded by execution against ELT-Bench
ground truth, and that grade is the reward.

*Implemented with the help of an AI coding assistant (Claude Code). I directed the design and the
experiments, set up and debugged the credentialed environment (Snowflake, Airbyte, ELT-Bench
sources), and reviewed the results.*

## 1. The ELT task as an RL environment

**Components of a task.** A *project base*: `config.yaml` (source and destination connection info),
`data_model.yaml` (target models described in natural language), `schemas/` (source columns), tool
documentation, and a Terraform stub. A running *environment*: source systems (Postgres, MongoDB,
S3 on LocalStack, a REST API, flat files), Airbyte for extract/load, dbt for transform, and the
destination warehouse. Hidden from the agent: expected raw row counts, ground-truth CSVs and unique
keys for every model.

**Episode.**

| | |
|---|---|
| Initial observation | System prompt, tool schemas, task instruction |
| Action | One assistant message with tool calls (parsed by the cookbook renderer) |
| Tools | `list_files`, `read_file`, `write_file` (confined to the workspace / writable dirs), `sql` (agent privileges, the rollout's namespace), `bash` (docker sandbox: terraform, Airbyte API, dbt), `run_models` (local mode), `submit` |
| Transition | Tool executes against sandbox/warehouse; its (truncated) output is the next observation |
| Terminal | `submit`; `max_turns`; context budget exceeded; repeated malformed tool calls |
| Reward | 0 per step (minus a small malformed-call penalty), execution-based grade at the end |

A turn *without* a tool call does not end the episode: the policy is nudged ("You did not call a
tool. Continue, or call `submit`"). The cookbook default ends on such a turn, but long-horizon agents
often end a turn with planning text; ending there graded unfinished pipelines (observed in practice).
The official ELT-Bench harness likewise stops only on an explicit terminate action.

**The environment is stateful.** Actions mutate the outside world (warehouse tables, Airbyte
sources/connections/jobs, workspace files) and that state, not the transcript, is graded. Effects
accumulate: syncs use `full_refresh_append`, so re-triggering one duplicates rows. Therefore every
rollout gets its own:

* **warehouse namespace**: a Snowflake database `RL_<TASK>_<ID>` (DuckDB: its own file);
* **Airbyte workspace**: created per rollout; teardown cancels its running jobs and deletes every
  connection, source and destination in it. (`terraform destroy` alone is not enough: a policy
  deleted `terraform.tfstate` mid-episode, leaking resources.);
* **workspace copy and docker sandbox**, with `config.yaml` rewritten so Airbyte, dbt and the agent
  all target that rollout's database and workspace.

G rollouts of a GRPO group therefore run concurrently without interference, and teardown is
guaranteed even when an episode crashes.

**Built on tinker-cookbook**: `AgentToolMessageEnv` (tool loop, parse-error retries),
`EnvFromMessageEnv` (token-level `Env`), the `@tool` decorator, `SandboxInterface`, and `rl.train`
(group-centered advantages, LoRA on Tinker). The structure follows the `harbor_rl` recipe.

## 2. Reward

**Correct pipeline (ELT-Bench definition).** Stage 1: every source table exists in the destination
with exactly the expected row count. Stage 2: each data model matches its ground truth: rows sorted
by the model's unique key, every gold column present (case-insensitive), numbers within 1% relative
tolerance, strings trimmed and case-insensitive, NULL only matches NULL; extra columns allowed. The
comparator is ported from `evaluation/eva_stage2.py`, including the CSV round-trip.

**Verified by execution of the submitted pipeline.** In the default `pipeline` mode the grader drops
any table the agent made under a model's name, compiles the submitted `models/*.sql` (dbt `ref` /
`source` / `config`, falling back to dbt's `target/compiled`), runs them in dependency order in the
rollout's namespace, and compares. A `warehouse` mode grades leftover state like the official
evaluator. Before grading, the environment waits (bounded, 15 min) for the rollout's in-flight
Airbyte syncs, so a correct pipeline is not scored on Airbyte's copy speed. Grading always happens
when an episode ends, including on token/context limits (the cookbook's limit paths skip the grader
and return a flat penalty; we add the execution grade on top of the penalty).

**Reward.**

    r = 0.2 * (raw tables loaded with exact row counts / expected raw tables)
      + 0.6 * (data models fully correct / data models)        # = ELT-Bench SRDT per task
      + 0.2 * (gold columns correct / gold columns)             # dense shaping
      - 0.05 per malformed tool call;  - 0.1 if the context budget is exhausted

**Why outcome-based with verified partial credit.** Advantages are rewards centered within a group of
rollouts of the same task; if all score the same (e.g. 0, which frontier models get on ~96% of
ELT-Bench models), there is no gradient. Process rewards from a learned model, or for actions such as
"ran terraform", are dense but unverifiable and easy to farm. The middle ground used here is milestone
credit that is still an *outcome check on execution*: each table loaded, each model correct, each
column correct. In our credentialed runs the same model scored 0.0 (stuck on S3), 0.087 (14/15
tables loaded, then out of context) and, transform-only, 0.791 (2/3 models): exactly the spread GRPO
needs, where a binary reward would have scored all of them 0, and every unit of it is verified
warehouse state.

## 3. Destination abstraction

`Destination` (`elt_rl/destinations/base.py`) is the only warehouse-specific code: namespace
lifecycle (`provision`, `clone_from_template`, `teardown`), `execute` with agent vs. grader
privileges, `qualify`, table listing, a deterministic perturbation statement, and `agent_config`.
Env, tools and grader are warehouse-agnostic. Implemented: **DuckDB** (local, credential-free) and
**Snowflake** (official). Databricks/Redshift need one subclass each.

## 4. Reward hacking: loopholes and mitigations

| Loophole | Mitigation | Tested |
|---|---|---|
| Read ground truth (files, or SQL `read_csv`) | GT never copied in; file tools confined to the workspace; DuckDB external access limited to the workspace and config locked | yes |
| Read admin credentials | `*_credential.json` never copied into workspaces; the agent connects as the restricted Airbyte user | yes |
| Write the official evaluator's cached CSV (`eva_stage2` trusts `<folder>/<schema>/<table>.csv`) | Our grader always queries the warehouse | by design |
| Hand-make the answer table | `pipeline` mode drops model-named tables and rebuilds them from submitted model files | yes |
| Constant model (`VALUES ...`) | Static check: a model must reference a source table or another model | yes |
| Model that mentions a source but ignores it | **Perturbation check**: delete ~20% of every raw table, re-run the models; byte-identical output loses its credit | yes |
| Load twice / drop rows | Stage-1 exact row counts (contents are not checked, see §7) | yes |
| Vacuous match on an empty table | Non-empty GT requires non-empty output | by design |
| Extra columns, 1% tolerance, case-insensitive strings | Inherited from the official metric for comparability | n/a |
| Edit task files | `write_file` limited to `models/` (local) or `elt/` (docker) | yes |
| Interfere with sibling rollouts | Per-rollout databases and Airbyte workspaces (a docker-mode agent could still name a sibling database: residual) | partly |
| Submit immediately to avoid penalties | Penalties are far smaller than any task credit; immediate submit scores 0 | yes |

## 5. Training efficiency

* **EL dominates latency.** `el_mode="preloaded"` clones a template database with the raw tables
  already loaded (Snowflake zero-copy `CLONE`, about a second; DuckDB copies a file): a transform-only
  curriculum with very cheap rollouts. Full `agent` mode is kept for end-to-end training/evaluation.
* **Provider mirror.** Every `terraform init` downloaded the ~50 MB Airbyte provider; sandboxes now
  install it from a local read-only mirror (`scripts/build_terraform_mirror.sh`). This also keeps
  rollouts working on networks that intercept GitHub downloads (we hit one).
* **Context budget.** Bash output was 43% of a 57k-token trajectory; it is now capped (3k chars,
  head + tail). Long-context sampling (`:peft:262144` model variant) and thinking disabled for the
  tool-heavy task keep episodes within budget.
* **Grading cost.** GT parsed once per group; one `SELECT *` per model; the perturbation re-run only
  for models that earned credit.
* **Concurrency.** Group rollouts open concurrently; blocking drivers run in threads; tools within a
  rollout run sequentially (they share state).
* **Further work.** Pre-warmed sandbox/namespace pools; caching Airbyte syncs by a hash of the
  normalized connection config; dropping zero-variance groups.

## 6. Making the benchmark runnable today: environment bugs found and fixed

Real rollouts surfaced eight problems that made ELT-Bench unsolvable (reward 0 for any policy) on
current infrastructure. Each was diagnosed from logs and fixed in `scripts/patch_eltbench.py`
or the environment code:

1. **Airbyte rejects email/password API auth** (HTTP 401): configs switched to client credentials;
   job-status helper and docs use bearer tokens.
2. **Airbyte's connector-runner registry fails to update** (Docker Hub refuses anonymous paging past
   1,000 tags): missing runner versions registered manually.
3. **Terraform provider 0.6.5 crashes on concurrent operations**, plus an Airbyte race on first
   secret creation: `-parallelism=1` (prompt and teardown).
4. **S3 connector drift**: 4.16 requires a nested `credentials` block the pinned provider cannot
   express. Pinned to 4.13.4 (April 2025, ELT-Bench's release).
5. **MongoDB connector drift**: 3.x requires a `databases` list and was renamed; the April-2025
   `source-mongodb-v2:1.5.16` is registered as a custom definition.
6. **The REST API source ignored `offset`/`limit`**, so tables with ≥ 9,973 rows (the connector's page
   size) paginated forever and their sync never finished. Fixed in the API server.
7. **LocalStack `latest` now requires a paid auth token**: pinned to 4.4.
8. **Agent image no longer builds** (`software-properties-common` removed in Debian 13): pinned to
   `python:3.11-bookworm`.

## 7. Results

| Check | Result |
|---|---|
| 1. Credential-free integration test | 15/15 tests pass (toy task on DuckDB, scripted policies through the real tool loop) |
| 2. Credentialed rollout, official task `books` | **Reference solution: reward 1.0** (15/15 tables, 3/3 models, perturbation check passed, all syncs settled). Model (Qwen3.6-35B-A3B), full task: best run loaded **14/15** raw tables end to end (the 15th was the never-ending API sync, bug 6), then ran out of context before transforming (reward 0.087) |
| 3. Tinker RL step | Completed on the toy task (Qwen3.5-4B, group of 4, rewards 0 / 0 / 0.133 / 0.2, KL ≈ 6e-4) |
| Transform-only (`preloaded`), model | **2/3 models fully correct, 21/22 gold columns**, reward 0.791 (Stage-1 credit is free in this mode; transform part 0.591/0.8). Only `customers.per_order_shipped_internationally` was wrong; no model flagged by the perturbation check |

The reference solution (`reference/books/`, played by `scripts/run_reference.py` through the same
tool loop, sandbox, warehouse namespace, Airbyte workspace and grader a policy gets) validates that
the environment is solvable and that the grader awards full credit to a correct pipeline.

## 8. Limitations and next steps

* Stage 1 checks row counts, not contents (as ELT-Bench); value tampering in raw tables is undetected.
* The perturbation check can false-positive on a model legitimately insensitive to every perturbed table.
* Small open models rarely finish whole tasks; RL mostly sharpens behaviour the policy already shows.
  Start from `preloaded` mode and easier tasks, or warm-start with SFT on successful (e.g. reference)
  trajectories.
* Only `books` was run end to end with credentials; the patch script applies to all 100 tasks.
