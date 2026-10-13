from __future__ import annotations

from dataclasses import dataclass
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
    def __init__(self, *, bash_mode: str = "ask") -> None:
        self.bash_mode = bash_mode

    def evaluate(self, tool_name: str, arguments: dict[str, Any]) -> PermissionDecision:
        if tool_name in {"read_file", "glob", "recall_relevant_memory", "plan_show", "delegate_research", "load_skill"}:
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
