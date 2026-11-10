"""Execution sandboxes implementing tinker-cookbook's ``SandboxInterface``.

* ``WorkspaceSandbox``: a plain directory on the host. File operations are
  confined to the workspace. ``run_command`` runs on the host, so the env only
  exposes a ``bash`` tool with this sandbox when explicitly allowed (it would
  otherwise be a path around the workspace confinement, e.g. to grader files).
* ``DockerSandbox``: the official ELT-Bench setup. The workspace is
  bind-mounted at ``/workspace`` in a fresh ``elt-swe`` container on the
  ELT-Bench docker network (sources, Airbyte), and commands run via
  ``docker exec``. Grader assets are never mounted.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path, PurePosixPath

from tinker_cookbook.sandbox import SandboxResult

logger = logging.getLogger(__name__)

CONTAINER_WORKDIR = "/workspace"
DEFAULT_MAX_OUTPUT_BYTES = 16 * 1024


def resolve_in_workspace(workspace: Path, path: str) -> Path:
    """Map an agent-supplied path to a host path inside ``workspace``.

    Accepts relative paths and ``/workspace/...`` container paths; rejects
    anything that escapes the workspace (``..``, other absolute paths, symlinks).
    """
    root = workspace.resolve()
    if path.startswith(CONTAINER_WORKDIR):
        path = str(PurePosixPath(path).relative_to(CONTAINER_WORKDIR))
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"path {path!r} is outside the workspace")
    return candidate


def _truncate(data: bytes, limit: int | None) -> str:
    text = data.decode("utf-8", errors="replace")
    limit = limit or DEFAULT_MAX_OUTPUT_BYTES
    if len(text) > limit:
        half = limit // 2
        text = text[:half] + f"\n...[{len(text) - limit} chars truncated]...\n" + text[-half:]
    return text


async def _run_process(args: list[str], timeout: int, max_output_bytes: int | None, cwd: Path | None = None) -> SandboxResult:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return SandboxResult(stdout="", stderr=f"TIMEOUT after {timeout}s", exit_code=124)
    return SandboxResult(
        stdout=_truncate(out, max_output_bytes),
        stderr=_truncate(err, max_output_bytes),
        exit_code=proc.returncode if proc.returncode is not None else -1,
    )


class _WorkspaceFiles:
    """File operations shared by both sandboxes (via the host-side workspace)."""

    workspace: Path

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        try:
            target = resolve_in_workspace(self.workspace, path)
            data = target.read_bytes()
        except (OSError, PermissionError) as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        if max_bytes is not None:
            data = data[:max_bytes]
        return SandboxResult(stdout=data.decode("utf-8", errors="replace"), stderr="", exit_code=0)

    async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
        try:
            target = resolve_in_workspace(self.workspace, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                content = content.encode("utf-8")
            target.write_bytes(content)
            if executable:
                target.chmod(0o755)
        except (OSError, PermissionError) as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        return SandboxResult(stdout=f"wrote {len(content)} bytes", stderr="", exit_code=0)

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None


class WorkspaceSandbox(_WorkspaceFiles):
    def __init__(self, workspace: Path):
        self.workspace = workspace

    @property
    def sandbox_id(self) -> str:
        return f"local:{self.workspace.name}"

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None) -> SandboxResult:
        cwd = resolve_in_workspace(self.workspace, workdir or ".")
        shell = ["cmd", "/c", command] if sys.platform == "win32" else ["bash", "-lc", command]
        return await _run_process(shell, timeout, max_output_bytes, cwd=cwd)

    async def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


class DockerSandbox(_WorkspaceFiles):
    """A fresh ELT-Bench agent container with the workspace mounted."""

    def __init__(self, workspace: Path, container: str, teardown_commands: list[str] | None = None):
        self.workspace = workspace
        self.container = container
        self.teardown_commands = teardown_commands or []

    @classmethod
    async def create(
        cls,
        workspace: Path,
        name: str,
        image: str = "elt-swe",
        network: str = "elt-docker_elt_network",
        teardown_commands: list[str] | None = None,
        terraform_mirror: Path | None = None,
    ) -> DockerSandbox:
        await _run_process(["docker", "rm", "-f", name], timeout=60, max_output_bytes=None)
        mirror_args: list[str] = []
        if terraform_mirror is not None and (terraform_mirror / "terraformrc").is_file():
            # Install Terraform providers from a local, read-only mirror (see
            # scripts/build_terraform_mirror.sh): no per-rollout download, and
            # rollouts keep working on networks that intercept GitHub downloads.
            mirror_args = ["-v", f"{terraform_mirror.resolve()}:/opt/tf-mirror:ro",
                           "-e", "TF_CLI_CONFIG_FILE=/opt/tf-mirror/terraformrc"]
        result = await _run_process(
            [
                "docker", "run", "-d", "--rm", "--name", name, "--network", network,
                "-v", f"{workspace.resolve()}:{CONTAINER_WORKDIR}", "-w", CONTAINER_WORKDIR,
                *mirror_args, image, "sleep", "infinity",
            ],
            timeout=300,
            max_output_bytes=None,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"failed to start container {name}: {result.stderr}")
        return cls(workspace, name, teardown_commands)

    @property
    def sandbox_id(self) -> str:
        return f"docker:{self.container}"

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 600, max_output_bytes: int | None = None) -> SandboxResult:
        return await _run_process(
            ["docker", "exec", "-w", workdir or CONTAINER_WORKDIR, self.container, "bash", "-lc", command],
            timeout,
            max_output_bytes,
        )

    async def cleanup(self) -> None:
        # Remove external side effects first (e.g. `terraform destroy` deletes the
        # rollout's Airbyte sources/connections), then the container itself.
        for command in self.teardown_commands:
            try:
                await self.run_command(command, timeout=600)
            except Exception as e:  # noqa: BLE001
                logger.warning("teardown command failed in %s: %s", self.container, e)
        await _run_process(["docker", "rm", "-f", self.container], timeout=60, max_output_bytes=None)
        shutil.rmtree(self.workspace, ignore_errors=True)
