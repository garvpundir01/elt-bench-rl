"""System prompts and task instructions.

The ELT-Bench instruction is adapted from the official agent runners
(``ELT-Bench/agents/claude_agent/run.py``), with the terminal condition changed
to the ``submit`` tool and dbt models required under ``elt/`` so the grader
can re-execute them.
"""

from __future__ import annotations

from elt_rl.task import ELTTask

SYSTEM_PROMPT = (
    "You are a data engineer skilled in databases, SQL, and building ELT pipelines. "
    "You work in a workspace that contains everything you need for the task. "
    "Work step by step: inspect the task files, act with one tool call at a time, read the "
    "result, and fix errors before moving on. When the pipeline is complete, call `submit`."
)

LOCAL_INSTRUCTION = """Build the ELT pipeline for `{task}`. The destination is {destination}.

Your workspace contains:
- config.yaml: the sources to extract (absolute path, format, target table) and the destination block
  (`{destination_key}.config.schema` is the schema all tables must be created in).
- data_model.yaml: the data models to build, with a description of every column.
- schemas/: column descriptions of the source tables.
- sources/: the source files.

Stage 1 - Extract and load:
  Load every source in config.yaml into `<schema>.<table>` exactly once, keeping all of its rows
  and columns, using the exact path from config.yaml
  (e.g. `CREATE TABLE elt.t AS SELECT * FROM read_csv_auto('<path from config.yaml>')`).
  Loading a source twice duplicates rows, which is an error.
Stage 2 - Transform:
  For every model in data_model.yaml write one SQL SELECT statement to `models/<model_name>.sql`
  producing exactly the described columns. Read from the loaded tables, either schema-qualified
  (`elt.orders`) or with `{{{{ source('elt', 'orders') }}}}`; use `{{{{ ref('other_model') }}}}`
  to build on another model. Use `run_models` to materialize them and `sql` to inspect results.
Finally call `submit`. Only tables produced by your model files count; tables you create by
hand are discarded before grading.
"""

ELTBENCH_INSTRUCTION = """Build the ELT pipeline for `{task}` now, using Airbyte Terraform and dbt.
All configuration values (including the {destination} database `{database}`) must come from
/workspace/config.yaml.

# Stage 1: Data Extraction and Loading
1. Configure the Airbyte provider in a Terraform file in /workspace/elt with the client_id,
   client_secret and server URL from /workspace/config.yaml (see
   /workspace/documentation/airbyte_Provider.md), then run `terraform init` in /workspace/elt.
   Do not modify the provided code in main.tf.
2. Configure the sources and the {destination} destination from /workspace/config.yaml in Terraform
   files in /workspace/elt. You must consult /workspace/documentation for the field names; the
   values in config.yaml are the real values. Field names may not match exactly, so infer them.
3. Create one connection per source (see /workspace/documentation/connection.md), with name and
   sync_mode in configuration.streams for exactly the tables listed in config.yaml. Each flat file
   needs its own source and connection.
4. Run `terraform apply -parallelism=1 -auto-approve` (the Airbyte provider crashes on concurrent
   operations, so always pass -parallelism=1), read the connection IDs from terraform.tfstate, and trigger
   each sync job ONCE via the Airbyte API (/workspace/documentation/trigger_job.md). Syncs use
   full_refresh_append, so triggering a job twice duplicates rows.
5. Wait for the jobs with `python /workspace/check_job_status.py --server <host:port> --client_id
   <client_id> --client_secret <client_secret>` until all succeed; fix and retry failures.

# Stage 2: Data Transformation
1. Create a dbt project under /workspace/elt (profiles from config.yaml) and remove the example models.
2. Read /workspace/data_model.yaml and /workspace/schemas, and write one model per data model as
   /workspace/elt/<project>/models/<model_name>.sql, named exactly after the model.
3. Run `dbt run` and fix errors; verify the tables with the `sql` tool.
4. Call `submit` when the models match data_model.yaml (or when you cannot make progress).
"""


def initial_instruction(task: ELTTask, destination_display: str, destination_key: str, database: str) -> str:
    if task.kind == "eltbench":
        return ELTBENCH_INSTRUCTION.format(task=task.name, destination=destination_display, database=database)
    return LOCAL_INSTRUCTION.format(
        task=task.name, destination=destination_display, destination_key=destination_key
    )
