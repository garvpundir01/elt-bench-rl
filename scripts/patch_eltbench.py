"""Make an ELT-Bench checkout work with current Airbyte (abctl) authentication.

Current Airbyte OSS rejects HTTP basic auth (email/password) on its public API
(HTTP 401); it requires an application token obtained with the instance's
client credentials. ELT-Bench's generated inputs only provide email/password,
so Stage 1 (Terraform + sync triggers) is impossible as shipped. This script:

1. replaces ``username``/``password`` in every ``inputs*/<task>/config.yaml``
   Airbyte block with ``client_id``/``client_secret`` (supported natively by the
   Airbyte Terraform provider, and listed in ``documentation/airbyte_Provider.md``);
2. replaces ``check_job_status.py`` with a token-authenticated version;
3. rewrites ``documentation/trigger_job.md`` to show the token flow;
4. pins connectors whose current versions no longer match ELT-Bench's
   documentation and Terraform provider (0.6.5): source-s3 -> 4.13.4 (newer
   versions moved the AWS keys into a ``credentials`` block the provider cannot
   express) and MongoDB -> airbyte/source-mongodb-v2:1.5.16 (newer versions,
   renamed to airbyte/source-mongodb, require a ``databases`` list). Both are the
   versions current at ELT-Bench's release (April 2025). MongoDB is registered as
   a custom definition and ``mongodb_definition_id`` is rewritten to it;
5. makes ELT-Bench's REST API source (elt-docker/rest_api/app.py) honour the
   ``offset``/``limit`` parameters the Airbyte connector paginates with. It
   ignored them, so every API table with >= 9973 rows (the connector's page size)
   was re-sent forever and its sync never finished;
6. pins images that no longer work as `latest`: LocalStack 4.4 (newer versions
   need a paid token) and the agent image base python:3.11-bookworm.

Not covered here (Airbyte-instance setup, see README): registering missing
declarative-manifest runner versions when Airbyte's Docker Hub update fails.

Run after ``setup/write_config.py``:

    python scripts/patch_eltbench.py --eltbench ~/ELT-Bench

Client credentials come from ``abctl local credentials`` (default) or the
AIRBYTE_CLIENT_ID / AIRBYTE_CLIENT_SECRET environment variables.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import yaml

CHECK_JOB_STATUS = '''"""Wait for this workspace's Airbyte sync jobs to finish and report their status."""
import argparse
import json
import time

import requests


def token(server, client_id, client_secret):
    response = requests.post(
        f"http://{server}/api/public/v1/applications/token",
        json={"client_id": client_id, "client_secret": client_secret, "grant-type": "client_credentials"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_connections():
    try:
        with open("/workspace/elt/terraform.tfstate", "r", encoding="utf-8") as f:
            tfstate = json.load(f)
        return [
            r["instances"][0]["attributes"]["connection_id"]
            for r in tfstate["resources"]
            if r["type"] == "airbyte_connection"
        ]
    except Exception as e:
        print(f"Error reading .tfstate file: {e}")
        return []


def get_job_status(server, client_id, client_secret):
    connections = get_connections()
    if not connections:
        print("Connection not found")
        return
    succeeded, failed = [], []
    while True:
        headers = {"Authorization": f"Bearer {token(server, client_id, client_secret)}", "accept": "application/json"}
        jobs = requests.get(f"http://{server}/api/public/v1/jobs?limit=100000", headers=headers, timeout=60).json()["data"]
        all_completed = True
        for job in jobs:
            if job["connectionId"] not in connections:
                continue
            if job["status"] == "succeeded":
                if job["connectionId"] not in succeeded:
                    succeeded.append(job["connectionId"])
            elif job["status"] in ("failed", "cancelled"):
                if job["connectionId"] not in failed:
                    failed.append(job["connectionId"])
            else:
                all_completed = False
        if all_completed:
            break
        time.sleep(10)
    print("The connection id of successful jobs are: ", succeeded)
    print("The connection id of failed jobs are: ", failed)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--server", required=True, help="Airbyte host:port, e.g. airbyte-abctl-control-plane:80")
    arg.add_argument("--client_id", required=True)
    arg.add_argument("--client_secret", required=True)
    args = arg.parse_args()
    get_job_status(args.server, args.client_id, args.client_secret)
'''

TRIGGER_JOB_MD = """# Trigger a sync

The Airbyte API uses bearer tokens. First exchange the client credentials for an
access token (tokens are short-lived; request a new one if you get HTTP 401):

```
curl --request POST \\
     --url http://<airbyte-server>/api/public/v1/applications/token \\
     --header 'content-type: application/json' \\
     --data '{"client_id": "<client_id>", "client_secret": "<client_secret>", "grant-type": "client_credentials"}'
```

The response contains `access_token`. Then trigger the sync:

```
curl --request POST \\
     --url http://<airbyte-server>/api/public/v1/jobs \\
     --header 'accept: application/json' \\
     --header 'authorization: Bearer <access_token>' \\
     --header 'content-type: application/json' \\
     --data '
{
  "jobType": "sync",
  "connectionId": "<connection_id>"
}
'
```
"""


API_PAGINATION_MARKER = "# elt-rl: honour offset/limit"
API_PAGINATION_PATCH = '''  data = data.to_dict(orient="records")
  # elt-rl: honour offset/limit. The Airbyte connector pages with OffsetIncrement
  # (page_size 9973) and stops on a short page; returning every row on every
  # request made any table with >= 9973 rows paginate forever.
  offset = request.args.get("offset", default=0, type=int)
  limit = request.args.get("limit", type=int)
  data = data[offset:offset + limit] if limit is not None else data[offset:]
  return data'''


def patch_rest_api(root: Path) -> bool:
    """Make ELT-Bench's REST API source honour the connector's offset/limit pagination."""
    app = root / "elt-docker" / "rest_api" / "app.py"
    text = app.read_text(encoding="utf-8")
    if API_PAGINATION_MARKER in text:
        return False
    old = '  data = data.to_dict(orient="records")\n  return data'
    if old not in text:
        raise SystemExit(f"unexpected load_csv() in {app}; patch manually")
    app.write_text(text.replace(old, API_PAGINATION_PATCH, 1), encoding="utf-8")
    return True


PINNED_IMAGES = [
    # LocalStack `latest` now refuses to start without a paid auth token.
    ("elt-docker/docker-compose.yml", "localstack/localstack:latest", "localstack/localstack:4.4"),
    # python:3.11 moved to Debian 13, which dropped software-properties-common.
    ("agents/SWE-agent/docker/elt-swe/Dockerfile", "FROM python:3.11\n", "FROM python:3.11-bookworm\n"),
]


def pin_images(root: Path) -> list[str]:
    changed = []
    for rel, old, new in PINNED_IMAGES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            changed.append(rel)
    return changed


def abctl_client_credentials() -> tuple[str, str]:
    out = subprocess.run(["abctl", "local", "credentials"], capture_output=True, text=True, check=True).stdout
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    cid = re.search(r"Client-Id:\s*(\S+)", out)
    secret = re.search(r"Client-Secret:\s*(\S+)", out)
    if not cid or not secret:
        raise SystemExit("could not parse Client-Id/Client-Secret from `abctl local credentials`")
    return cid.group(1), secret.group(1)


S3_VERSION = "4.13.4"
MONGODB_IMAGE = ("airbyte/source-mongodb-v2", "1.5.16")


def pin_connectors(api_url: str, client_id: str, client_secret: str, sample_config: dict) -> str:
    """Pin source-s3 and register the ELT-Bench-era MongoDB connector; returns its definition id."""
    import requests

    token = requests.post(f"{api_url}/public/v1/applications/token", json={
        "client_id": client_id, "client_secret": client_secret, "grant-type": "client_credentials"}, timeout=60)
    token.raise_for_status()
    h = {"Authorization": f"Bearer {token.json()['access_token']}", "content-type": "application/json"}
    workspace_id = sample_config["workspace_id"]

    r = requests.post(f"{api_url}/v1/source_definitions/update", headers=h, timeout=900, json={
        "sourceDefinitionId": sample_config["s3_definition_id"], "dockerImageTag": S3_VERSION,
        "workspaceId": workspace_id})
    r.raise_for_status()
    print(f"source-s3 pinned to {S3_VERSION}")

    repo, tag = MONGODB_IMAGE
    defs = requests.post(f"{api_url}/v1/source_definitions/list_for_workspace", headers=h, timeout=60,
                         json={"workspaceId": workspace_id}).json()["sourceDefinitions"]
    for d in defs:
        if d["dockerRepository"] == repo and d["dockerImageTag"] == tag:
            print(f"{repo}:{tag} already registered as {d['sourceDefinitionId']}")
            return d["sourceDefinitionId"]
    r = requests.post(f"{api_url}/v1/source_definitions/create_custom", headers=h, timeout=900, json={
        "workspaceId": workspace_id,
        "sourceDefinition": {"name": f"MongoDb (ELT-Bench {tag})", "dockerRepository": repo, "dockerImageTag": tag,
                             "documentationUrl": "https://docs.airbyte.com/integrations/sources/mongodb-v2"}})
    r.raise_for_status()
    print(f"registered {repo}:{tag} as {r.json()['sourceDefinitionId']}")
    return r.json()["sourceDefinitionId"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eltbench", type=Path, default=Path.home() / "ELT-Bench")
    parser.add_argument("--airbyte-url", default="http://localhost:8000/api",
                        help="Airbyte API root reachable from this machine (for connector pinning)")
    parser.add_argument("--no-pin", action="store_true", help="skip connector version pinning")
    args = parser.parse_args()
    root: Path = args.eltbench.expanduser()

    client_id = os.environ.get("AIRBYTE_CLIENT_ID")
    client_secret = os.environ.get("AIRBYTE_CLIENT_SECRET")
    if not (client_id and client_secret):
        client_id, client_secret = abctl_client_credentials()

    if patch_rest_api(root):
        print("patched elt-docker/rest_api/app.py; restart it: docker restart elt-api")
    for rel in pin_images(root):
        print(f"pinned image in {rel} (rebuild/recreate the affected container)")

    configs = sorted(root.glob("inputs*/*/config.yaml"))
    mongodb_definition_id = None
    if not args.no_pin and configs:
        sample = yaml.safe_load(configs[0].read_text(encoding="utf-8"))["Airbyte"]["config"]
        mongodb_definition_id = pin_connectors(args.airbyte_url, client_id, client_secret, sample)

    (root / "setup" / "check_job_status.py").write_text(CHECK_JOB_STATUS, encoding="utf-8")
    (root / "documentation" / "trigger_job.md").write_text(TRIGGER_JOB_MD, encoding="utf-8")

    n = 0
    for config_path in configs:
        task_dir = config_path.parent
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        airbyte = config["Airbyte"]["config"]
        airbyte.pop("username", None)
        airbyte.pop("password", None)
        airbyte["client_id"] = client_id
        airbyte["client_secret"] = client_secret
        if mongodb_definition_id:
            airbyte["mongodb_definition_id"] = mongodb_definition_id
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (task_dir / "check_job_status.py").write_text(CHECK_JOB_STATUS, encoding="utf-8")
        if (task_dir / "documentation").is_dir():
            (task_dir / "documentation" / "trigger_job.md").write_text(TRIGGER_JOB_MD, encoding="utf-8")
        n += 1
    print(f"patched {n} task input bundles under {root}")


if __name__ == "__main__":
    main()
