"""Per-rollout Airbyte workspaces.

An ELT-Bench rollout creates Airbyte sources, destinations and connections
through Terraform. Tearing those down with ``terraform destroy`` is not
reliable: the policy controls the workspace and can delete or corrupt
``terraform.tfstate`` (observed in practice), leaking resources into later
rollouts. Instead every rollout gets its own Airbyte workspace, the same
isolation idea as the per-rollout warehouse namespace; teardown deletes every
connection, source and destination in it and then the workspace itself.

Custom source definitions (ELT-Bench's REST API connector, the pinned
MongoDB connector) are granted to each new workspace so the task's
``*_definition_id`` values keep working.
"""

from __future__ import annotations

import asyncio
import logging

import requests

logger = logging.getLogger(__name__)


class AirbyteWorkspaces:
    def __init__(self, api_url: str, client_id: str, client_secret: str, timeout: int = 60):
        # api_url is the Airbyte API root reachable from the host, e.g. http://localhost:8000/api
        self.api_url = api_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        r = requests.post(
            f"{self.api_url}/public/v1/applications/token",
            json={"client_id": self.client_id, "client_secret": self.client_secret,
                  "grant-type": "client_credentials"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return {"Authorization": f"Bearer {r.json()['access_token']}", "content-type": "application/json"}

    def _create(self, name: str, grant_source_definitions: list[str]) -> str:
        h = self._headers()
        r = requests.post(f"{self.api_url}/public/v1/workspaces", headers=h, json={"name": name}, timeout=self.timeout)
        r.raise_for_status()
        workspace_id = r.json()["workspaceId"]
        for definition_id in grant_source_definitions:
            g = requests.post(
                f"{self.api_url}/v1/source_definitions/grant_definition",
                headers=h,
                json={"actorDefinitionId": definition_id, "scopeId": workspace_id, "scopeType": "workspace"},
                timeout=self.timeout,
            )
            # Built-in and organization-scoped definitions need no grant and may reject one.
            if not g.ok:
                logger.debug("grant of %s to %s: HTTP %s", definition_id, workspace_id, g.status_code)
        return workspace_id

    def _delete(self, workspace_id: str) -> None:
        h = self._headers()
        # Deleting connections does not stop syncs already running; a sync left
        # behind keeps consuming source and warehouse resources after teardown.
        jobs = requests.get(f"{self.api_url}/public/v1/jobs", headers=h, timeout=self.timeout,
                            params={"workspaceIds": workspace_id, "limit": 1000})
        if jobs.ok:
            for job in jobs.json().get("data", []):
                if job["status"] in ("pending", "running", "incomplete"):
                    requests.delete(f"{self.api_url}/public/v1/jobs/{job['jobId']}", headers=h,
                                    timeout=self.timeout)
        for kind, id_key in (("connections", "connectionId"), ("sources", "sourceId"),
                             ("destinations", "destinationId")):
            r = requests.get(f"{self.api_url}/public/v1/{kind}", headers=h,
                             params={"workspaceIds": workspace_id, "limit": 1000}, timeout=self.timeout)
            r.raise_for_status()
            for item in r.json().get("data", []):
                requests.delete(f"{self.api_url}/public/v1/{kind}/{item[id_key]}", headers=h, timeout=self.timeout)
        requests.delete(f"{self.api_url}/public/v1/workspaces/{workspace_id}", headers=h,
                        timeout=self.timeout).raise_for_status()

    def _running_jobs(self, workspace_id: str) -> int:
        h = self._headers()
        r = requests.get(f"{self.api_url}/public/v1/jobs", headers=h, timeout=self.timeout,
                         params={"workspaceIds": workspace_id, "limit": 1000})
        r.raise_for_status()
        return sum(j["status"] in ("pending", "running", "incomplete") for j in r.json().get("data", []))

    async def wait_for_syncs(self, workspace_id: str, max_wait_s: int, poll_s: int = 15) -> int:
        """Wait (bounded) until no sync job in the workspace is still in flight.

        Grading must measure the submitted pipeline, not how fast Airbyte copies
        data: a correct connection whose sync is still running when the episode
        ends would otherwise score as a missing table. Returns jobs still running
        after the wait (0 = settled).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait_s
        while True:
            running = await asyncio.to_thread(self._running_jobs, workspace_id)
            if running == 0 or loop.time() >= deadline:
                return running
            await asyncio.sleep(poll_s)

    async def create(self, name: str, grant_source_definitions: list[str]) -> str:
        return await asyncio.to_thread(self._create, name, grant_source_definitions)

    async def delete(self, workspace_id: str) -> None:
        await asyncio.to_thread(self._delete, workspace_id)
