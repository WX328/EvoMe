from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class PermissionDecision:
    action: str  # allow | ask | deny
    risk: str
    reason: str


@dataclass(frozen=True)
class PendingApproval:
    id: str
    tool_name: str
    arguments: dict[str, Any]
    risk: str
    reason: str


class PermissionPolicy:
    def __init__(
        self, *,
        bash_mode: str = "ask",
        workspace_root: str | Path = ".",
        protected_memory_directory: str | Path | None = None,
        protected_task_directory: str | Path | None = None,
        protected_schedule_directory: str | Path | None = None,
        protected_team_directory: str | Path | None = None,
        protected_workflow_directory: str | Path | None = None,
    ) -> None:
        self.bash_mode = bash_mode
        self.workspace_root = Path(workspace_root).resolve()
        self.protected_memory_directory = (
            Path(protected_memory_directory).resolve()
            if protected_memory_directory is not None else None
        )
        self.protected_task_directory = (
            Path(protected_task_directory).resolve()
            if protected_task_directory is not None else None
        )
        self.protected_schedule_directory = (
            Path(protected_schedule_directory).resolve()
            if protected_schedule_directory is not None else None
        )
        self.protected_team_directory = (
            Path(protected_team_directory).resolve()
            if protected_team_directory is not None else None
        )
        self.protected_workflow_directory = (
            Path(protected_workflow_directory).resolve()
            if protected_workflow_directory is not None else None
        )

    def evaluate(self, tool_name: str, arguments: dict[str, Any]) -> PermissionDecision:
        # S09 writes go through explicit candidate confirmation, never through
        # a normal Agent file-edit Tool, even if that Tool is approved.
        if self.protected_memory_directory is not None and tool_name in {"write_file", "edit_file"}:
            raw_path = arguments.get("path")
            if isinstance(raw_path, str):
                try:
                    requested = (self.workspace_root / raw_path).resolve(strict=False)
                    if requested.is_relative_to(self.protected_memory_directory):
                        return PermissionDecision(
                            "deny", "high",
                            "S09 working memory is protected: approve an extracted candidate instead",
                        )
                except (ValueError, OSError, RuntimeError):
                    return PermissionDecision("deny", "high", "unresolvable memory write target")
        if self.protected_task_directory is not None and tool_name in {"write_file", "edit_file"}:
            path = arguments.get("path")
            if isinstance(path, str):
                try:
                    requested = (self.workspace_root / path).resolve(strict=False)
                    if requested.is_relative_to(self.protected_task_directory):
                        return PermissionDecision(
                            "deny", "high",
                            "S10 task state must be modified through the TaskStore",
                        )
                except (ValueError, OSError, RuntimeError):
                    return PermissionDecision("deny", "high", "unresolvable task write target")
        if self.protected_schedule_directory is not None and tool_name in {"write_file", "edit_file"}:
            path = arguments.get("path")
            if isinstance(path, str):
                try:
                    requested = (self.workspace_root / path).resolve(strict=False)
                    if requested.is_relative_to(self.protected_schedule_directory):
                        return PermissionDecision(
                            "deny", "high", "S12 schedule state cannot be edited through file tools"
                        )
                except (ValueError, OSError, RuntimeError):
                    return PermissionDecision("deny", "high", "unresolvable schedule write target")
        if self.protected_team_directory is not None and tool_name in {"write_file", "edit_file"}:
            path = arguments.get("path")
            if isinstance(path, str):
                try:
                    requested = (self.workspace_root / path).resolve(strict=False)
                    if requested.is_relative_to(self.protected_team_directory):
                        return PermissionDecision(
                            "deny", "high", "S13 team inbox/roster is managed only by TeamRuntime"
                        )
                except (ValueError, OSError, RuntimeError):
                    return PermissionDecision("deny", "high", "invalid team write target")
        if self.protected_workflow_directory is not None and tool_name in {"read_file", "write_file", "edit_file"}:
            target = arguments.get("path")
            if isinstance(target, str):
                try:
                    resolved = (self.workspace_root / target).resolve(strict=False)
                    if resolved.is_relative_to(self.protected_workflow_directory):
                        return PermissionDecision(
                            "deny", "high", "S16 private journal/snapshot files are inaccessible"
                        )
                except (ValueError, OSError, RuntimeError):
                    return PermissionDecision("deny", "high", "invalid workflow write path")
        if tool_name == "Workflow":
            return PermissionDecision(
                "ask", "high", "S16 saved Workflow launch/resume needs explicit approval"
            )
        # MCP tools are remote capabilities. Never trust server-provided
        # readOnlyHint or descriptions as a substitute for approval.
        if tool_name.startswith("mcp__"):
            return PermissionDecision(
                "ask", "high", "external MCP tool invocation requires operator approval"
            )
        if tool_name in {"task_list", "task_get", "task_ready"}:
            return PermissionDecision("allow", "low", "read-only durable task lookup")
        if tool_name in {
            "task_create", "task_add_dependency", "task_claim",
            "task_complete", "task_release",
        }:
            return PermissionDecision(
                "ask", "medium", "persistent task mutation requires approval"
            )
        if tool_name in {"read_file", "glob", "recall_relevant_memory", "plan_show", "delegate_research", "load_skill", "recall_agent_memory"}:
            return PermissionDecision("allow", "low", "read-only or memory retrieval")

        if tool_name in {"write_file", "edit_file"}:
            return PermissionDecision("ask", "medium", "workspace content will be modified")

        if tool_name == "bash":
            mode = self.bash_mode if self.bash_mode in {"allow", "ask", "deny"} else "ask"
            risk = "high"
            return PermissionDecision(mode, risk, "shell command execution")

        return PermissionDecision("deny", "high", "tool has no permission rule")


class ApprovalStore:
    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}

    def create(self, *, tool_name: str, arguments: dict[str, Any], risk: str, reason: str) -> PendingApproval:
        item = PendingApproval(
            id=uuid4().hex[:8],
            tool_name=tool_name,
            arguments=dict(arguments),
            risk=risk,
            reason=reason,
        )
        self._pending[item.id] = item
        return item

    def get(self, approval_id: str) -> PendingApproval | None:
        return self._pending.get(approval_id)

    def pop(self, approval_id: str) -> PendingApproval | None:
        return self._pending.pop(approval_id, None)

    def list_pending(self) -> list[PendingApproval]:
        return list(self._pending.values())
