"""Tools the policy uses to build a pipeline.

Observation = the conversation; action = one assistant message with tool
calls; each tool result is the execution feedback for the next turn. The
episode ends when the policy calls ``submit`` (or stops calling tools, or
runs out of turns), and only then is the warehouse graded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import Tool, ToolResult, simple_tool_result, tool

from elt_rl.destinations import Destination, Namespace, format_result
from elt_rl.models import compile_model, execution_order, find_model_files
from elt_rl.sandbox import resolve_in_workspace

MAX_TOOL_OUTPUT_CHARS = 12_000
# Shell output is the largest context consumer in long rollouts (Terraform logs,
# `cat terraform.tfstate`, polling loops): measured at 43% of a 57k-token trajectory.
# Keep its head and tail, where commands put their errors and results.
MAX_BASH_STDOUT_CHARS = 3_000
MAX_BASH_STDERR_CHARS = 1_500
MAX_READ_LINES = 400


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} chars truncated]...\n" + text[-half:]


class ELTTools:
    """Per-rollout tool set bound to one sandbox and one warehouse namespace."""

    def __init__(
        self,
        sandbox: SandboxInterface,
        destination: Destination,
        ns: Namespace,
        *,
        writable_dirs: tuple[str, ...] | None = ("models",),
        allow_bash: bool = False,
        allow_run_models: bool = True,
        command_timeout: int = 600,
    ):
        self.sandbox = sandbox
        self.destination = destination
        self.ns = ns
        self.writable_dirs = writable_dirs
        self.allow_bash = allow_bash
        self.allow_run_models = allow_run_models
        self.command_timeout = command_timeout
        self.submitted = False

    def tool_list(self) -> list[Tool]:
        tools: list[Tool] = [self.list_files, self.read_file, self.write_file, self.sql]
        if self.allow_run_models:
            tools.append(self.run_models)
        if self.allow_bash:
            tools.append(self.bash)
        tools.append(self.submit)
        return tools

    @property
    def workspace(self) -> Path:
        return self.ns.workspace

    # -- files ----------------------------------------------------------------
    @tool
    async def list_files(
        self, path: Annotated[str, "Directory relative to the workspace root."] = "."
    ) -> ToolResult:
        """List files in a workspace directory (two levels deep)."""
        try:
            root = resolve_in_workspace(self.workspace, path)
        except PermissionError as e:
            return simple_tool_result(f"error: {e}")
        if not root.is_dir():
            return simple_tool_result(f"error: {path!r} is not a directory")
        base = self.workspace.resolve()
        entries = []
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if len(rel.parts) <= 2:
                entries.append(str(p.relative_to(base).as_posix()) + ("/" if p.is_dir() else ""))
        return simple_tool_result(_clip("\n".join(entries) or "(empty)"))

    @tool
    async def read_file(
        self,
        path: Annotated[str, "File path relative to the workspace root."],
        offset: Annotated[int, "First line to return (0-based)."] = 0,
        limit: Annotated[int, "Maximum number of lines to return."] = MAX_READ_LINES,
    ) -> ToolResult:
        """Read a text file from the workspace."""
        result = await self.sandbox.read_file(path)
        if result.exit_code != 0:
            return simple_tool_result(f"error: {result.stderr}")
        lines = result.stdout.splitlines()
        limit = max(1, min(limit, MAX_READ_LINES))
        chunk = lines[offset : offset + limit]
        text = "\n".join(chunk)
        if offset + limit < len(lines):
            text += f"\n...[{len(lines) - offset - limit} more lines; use offset to continue]"
        return simple_tool_result(_clip(text))

    @tool
    async def write_file(
        self,
        path: Annotated[str, "File path relative to the workspace root."],
        content: Annotated[str, "Full new content of the file."],
    ) -> ToolResult:
        """Create or overwrite a file in the workspace."""
        if self.writable_dirs is not None:
            try:
                target = resolve_in_workspace(self.workspace, path)
            except PermissionError as e:
                return simple_tool_result(f"error: {e}")
            allowed = [resolve_in_workspace(self.workspace, d) for d in self.writable_dirs]
            if not any(target == d or d in target.parents for d in allowed):
                return simple_tool_result(
                    f"error: you may only write under {', '.join(d + '/' for d in self.writable_dirs)}"
                )
        result = await self.sandbox.write_file(path, content)
        return simple_tool_result(result.stdout if result.exit_code == 0 else f"error: {result.stderr}")

    # -- warehouse ------------------------------------------------------------
    @tool
    async def sql(
        self, query: Annotated[str, "A single SQL statement to run in the destination warehouse."]
    ) -> ToolResult:
        """Run SQL in your destination namespace and return the result (first 50 rows)."""
        try:
            result = await self.destination.execute(self.ns, query, as_agent=True)
        except Exception as e:  # noqa: BLE001 - SQL errors are feedback for the policy
            return simple_tool_result(f"SQL error: {_clip(str(e))}")
        return simple_tool_result(_clip(format_result(result)))

    @tool
    async def run_models(self) -> ToolResult:
        """Materialize every SQL file under models/ as a table (like `dbt run`) and report the outcome."""
        models = find_model_files(self.workspace)
        if not models:
            return simple_tool_result("no model files found under models/")
        try:
            order = execution_order(models, list(models))
        except ValueError as e:
            return simple_tool_result(f"error: {e}")
        report = []
        for name in order:
            try:
                compiled = compile_model(
                    self.workspace, models[name], lambda t: self.destination.qualify(self.ns, t)
                )
                await self.destination.materialize(self.ns, name, compiled)
                rows = await self.destination.row_count(self.ns, name)
                report.append(f"{name}: OK ({rows} rows)")
            except Exception as e:  # noqa: BLE001
                report.append(f"{name}: ERROR {str(e).splitlines()[0][:500]}")
        return simple_tool_result("\n".join(report))

    # -- shell ----------------------------------------------------------------
    @tool
    async def bash(self, command: Annotated[str, "Shell command to run in /workspace."]) -> ToolResult:
        """Run a shell command in the task container (terraform, dbt, python, curl, ...)."""
        result = await self.sandbox.run_command(
            command, timeout=self.command_timeout, max_output_bytes=MAX_TOOL_OUTPUT_CHARS
        )
        return simple_tool_result(
            json.dumps({
                "exit_code": result.exit_code,
                "stdout": _clip(result.stdout, MAX_BASH_STDOUT_CHARS),
                "stderr": _clip(result.stderr, MAX_BASH_STDERR_CHARS),
            })
        )

    # -- terminal action --------------------------------------------------------
    @tool
    async def submit(
        self, summary: Annotated[str, "One-line summary of the pipeline you built."] = ""
    ) -> ToolResult:
        """Submit the pipeline for grading. Call this once, when you are done."""
        self.submitted = True
        return simple_tool_result("submitted", should_stop=True)
