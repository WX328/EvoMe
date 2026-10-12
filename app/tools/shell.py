from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.tools.workspace import WorkspaceGuard


class BashTool:
    """S2 shell tool.

    Permission/approval policy intentionally belongs to S3. For S2 the shell
    tool is disabled by default and, when enabled, starts in the workspace root.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        timeout_seconds: int = 20,
        max_output_chars: int = 12000,
    ) -> None:
        self.guard = WorkspaceGuard(workspace_root)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_output_chars = max(1000, int(max_output_chars))

    def _clip(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        remaining = len(value) - self.max_output_chars
        return value[: self.max_output_chars] + f"\n...[truncated {remaining} chars]"

    def run(self, command: str) -> dict[str, Any]:
        command = str(command).strip()
        if not command:
            raise ValueError("command cannot be empty")

        completed = subprocess.run(
            command,
            cwd=str(self.guard.root),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
        )
        return {
            "command": command,
            "cwd": str(self.guard.root),
            "returncode": int(completed.returncode),
            "stdout": self._clip(completed.stdout or ""),
            "stderr": self._clip(completed.stderr or ""),
        }
