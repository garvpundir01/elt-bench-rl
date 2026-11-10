"""Run a scripted reference solution through the real ELT-Bench environment.

This validates the environment itself: a correct pipeline, played through the
exact tool loop, sandbox, warehouse namespace, Airbyte workspace and grader that
a policy gets, should earn (close to) the maximum reward. It is not a model
result.

    python scripts/run_reference.py --task books               # full run, graded, torn down
    python scripts/run_reference.py --task books --stop-after-el
        # load data only and leave the rollout's resources up for SQL development

Reference assets live in reference/<task>/: pipeline.tf.tmpl (Terraform for all
sources, filled from the rollout's config.yaml) and models/*.sql (dbt models).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from string import Template

import yaml
from tinker_cookbook.renderers.base import ToolCall

from elt_rl.destinations import DestinationSpec
from elt_rl.env import ELTEnvConfig, ELTRollout
from elt_rl.grading import ELTGrader, GraderConfig
from elt_rl.task import load_eltbench_tasks

REPO = Path(__file__).resolve().parents[1]

TRIGGER_SYNCS = '''import json, requests, yaml
cfg = yaml.safe_load(open("/workspace/config.yaml"))["Airbyte"]["config"]
api = cfg["server_url"].rstrip("/")
tok = requests.post(f"{api}/applications/token", json={
    "client_id": cfg["client_id"], "client_secret": cfg["client_secret"],
    "grant-type": "client_credentials"}).json()["access_token"]
h = {"Authorization": f"Bearer {tok}", "accept": "application/json"}
state = json.load(open("/workspace/elt/terraform.tfstate"))
for r in state["resources"]:
    if r["type"] == "airbyte_connection":
        cid = r["instances"][0]["attributes"]["connection_id"]
        resp = requests.post(f"{api}/jobs", headers=h, json={"jobType": "sync", "connectionId": cid})
        print(r["name"], cid, resp.status_code)
'''


def _streams(names: list[str]) -> str:
    return ", ".join(f'{{ name = "{n}", sync_mode = "full_refresh_append" }}' for n in names)


def render_terraform(task_dir: Path, cfg: dict) -> str:
    a, sf, pg, mg = cfg["Airbyte"]["config"], cfg["snowflake"]["config"], cfg["postgres"]["config"], cfg["mongodb"]["config"]
    s3 = cfg["aws_s3"]
    bucket = s3["data"][0]["path"].split("/")[2]
    s3_streams = "\n".join(
        f'      {{ name = "{d["table"]}", globs = ["{d["path"].split("/", 3)[3]}"], '
        f'format = {{ csv_format = {{}} }} }},'
        for d in s3["data"]
    )
    file_sources, file_connections = [], []
    for f in cfg.get("flat_files", []):
        t = f["table"]
        file_sources.append(f'''resource "airbyte_source_file" "file_{t}" {{
  name          = "File {t}"
  workspace_id  = "{a["workspace_id"]}"
  definition_id = "{a["files_definition_id"]}"
  configuration = {{
    dataset_name = "{t}"
    format       = "{f["format"]}"
    provider     = {{ https_public_web = {{}} }}
    url          = "{f["path"]}"
  }}
}}''')
        file_connections.append(f'''resource "airbyte_connection" "file_{t}" {{
  name                 = "File {t} to Snowflake"
  source_id            = airbyte_source_file.file_{t}.source_id
  destination_id       = airbyte_destination_snowflake.snowflake.destination_id
  namespace_definition = "destination"
  configurations = {{ streams = [{_streams([t])}] }}
}}''')
    return Template((task_dir / "pipeline.tf.tmpl").read_text()).substitute(
        server_url=a["server_url"], client_id=a["client_id"], client_secret=a["client_secret"],
        workspace_id=a["workspace_id"], snowflake_definition_id=a["snowflake_definition_id"],
        postgres_definition_id=a["postgres_definition_id"], mongodb_definition_id=a["mongodb_definition_id"],
        custom_api_definition_id=a["custom_api_definition_id"], s3_definition_id=a["s3_definition_id"],
        snowflake_account=sf["account"], snowflake_role=sf["role"], snowflake_warehouse=sf["warehouse"],
        snowflake_database=sf["database"], snowflake_schema=sf["schema"], snowflake_username=sf["username"],
        snowflake_password=sf["password"],
        pg_host=pg["host"], pg_port=pg["port"], pg_database=pg["database"], pg_schema=pg["schema"],
        pg_user=pg["user"], pg_password=pg["password"], pg_streams=_streams(pg["tables"]),
        mongo_connection_string=mg["connection_string"], mongo_database=mg["database"],
        mongo_streams=_streams(mg["tables"]), api_streams=_streams(cfg["custom_api"]["config"]["tables"]),
        aws_access_key_id=s3["AWS_ACCESS_KEY_ID"], aws_secret_access_key=s3["AWS_SECRET_ACCESS_KEY"],
        aws_endpoint=s3["AWS_ENDPOINT_URL"], aws_region=s3["AWS_DEFAULT_REGION"], s3_bucket=bucket,
        s3_streams=s3_streams, s3_conn_streams=_streams([d["table"] for d in s3["data"]]),
        file_sources="\n\n".join(file_sources), file_connections="\n\n".join(file_connections),
    )


def dbt_files(task: str, cfg: dict, models_dir: Path) -> dict[str, str]:
    sf = cfg["snowflake"]["config"]
    project = f"elt/{task}_dbt"
    files = {
        f"{project}/dbt_project.yml": (
            f"name: {task}_dbt\nversion: '1.0'\nprofile: {task}_dbt\nmodel-paths: ['models']\n"
            f"models:\n  {task}_dbt:\n    +materialized: table\n"
        ),
        f"{project}/profiles.yml": yaml.safe_dump({f"{task}_dbt": {"target": "dev", "outputs": {"dev": {
            "type": "snowflake", "account": sf["account"], "user": sf["username"], "password": sf["password"],
            "role": sf["role"], "warehouse": sf["warehouse"], "database": sf["database"],
            "schema": sf["schema"], "threads": 1}}}}),
    }
    for model in sorted(models_dir.glob("*.sql")):
        files[f"{project}/models/{model.name}"] = model.read_text()
    return files


def call(name: str, args: dict, i: int) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)), id=f"r{i}")]}


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="books")
    p.add_argument("--eltbench", default=os.path.expanduser("~/ELT-Bench"))
    p.add_argument("--stop-after-el", action="store_true")
    args = p.parse_args()
    eb, ref = Path(args.eltbench), REPO / "reference" / args.task

    task = load_eltbench_tasks(eb / "inputs", eb / "evaluation", eb / "ground_truth/gt_snowflake", only=[args.task])[0]
    dest = DestinationSpec("snowflake", credential_path=str(eb / "setup/destination/snowflake_credential.json")).build()
    config = ELTEnvConfig(sandbox="docker", command_timeout=1800, max_turns=50)
    rollout = await ELTRollout.open(task, dest, ELTGrader(task, dest, GraderConfig()), config)
    keep = False
    try:
        keep = await play(rollout, config, args, ref)
    finally:
        if not keep:
            await rollout.close()
        dest.close()


async def play(rollout: ELTRollout, config: ELTEnvConfig, args: argparse.Namespace, ref: Path) -> bool:
    """Play the reference tool calls; returns True if resources should be left running."""
    cfg = yaml.safe_load((rollout.ns.workspace / "config.yaml").read_text())
    server = cfg["Airbyte"]["config"]["server_url"].split("//")[1].split("/")[0]

    calls = [
        ("write_file", {"path": "elt/pipeline.tf", "content": render_terraform(ref, cfg)}),
        ("bash", {"command": "cd /workspace/elt && terraform init -no-color >/dev/null && "
                             "terraform apply -parallelism=1 -auto-approve -no-color 2>&1 | tail -3"}),
        ("write_file", {"path": "elt/trigger_syncs.py", "content": TRIGGER_SYNCS}),
        ("bash", {"command": "python /workspace/elt/trigger_syncs.py"}),
        ("bash", {"command": f"python /workspace/check_job_status.py --server {server} "
                             f"--client_id {cfg['Airbyte']['config']['client_id']} "
                             f"--client_secret {cfg['Airbyte']['config']['client_secret']}"}),
    ]
    if not args.stop_after_el:
        calls += [("write_file", {"path": path, "content": content})
                  for path, content in dbt_files(args.task, cfg, ref / "models").items()]
        calls += [("bash", {"command": f"cd /workspace/elt/{args.task}_dbt && dbt run --profiles-dir . 2>&1 | tail -8"}),
                  ("submit", {"summary": "reference pipeline"})]

    env = rollout.message_env(rollout.initial_messages(None), config)
    await env.initial_observation()
    for i, (name, a) in enumerate(calls):
        step = await env.step(call(name, a, i))
        out = str(env.history[-1]["content"])
        print(f"[{i}] {name}: {out[-600:]}\n")
        if step.episode_done:
            print("REWARD:", step.reward)
            print(json.dumps({k: v for k, v in step.metrics.items() if k.startswith("elt/")}, indent=2))
            print(rollout.last_grade.summary() if rollout.last_grade else "")
            break

    if args.stop_after_el:
        print("LEFT RUNNING for SQL development:")
        print(json.dumps({"database": rollout.ns.database, "airbyte_workspace": rollout.airbyte[1],
                          "container": rollout.sandbox.container, "workspace": str(rollout.ns.workspace)}, indent=2))
        return True
    return False


if __name__ == "__main__":
    asyncio.run(main())
