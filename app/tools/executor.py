from __future__ import annotations

from typing import Any

from app.tools.base import ToolResult
from app.tools.permissions import ApprovalStore, PermissionPolicy
from app.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_policy: PermissionPolicy | None = None,
        approval_store: ApprovalStore | None = None,
    ) -> None:
        self.registry = registry
        self.permission_policy = permission_policy or PermissionPolicy()
        self.approval_store = approval_store or ApprovalStore()

    def _run(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            implementation = self.registry.resolve(tool_name)
            output = implementation.execute(**arguments)
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(tool_name=tool_name, success=True, output=output)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        approved: bool = False,
    ) -> ToolResult:
        args = dict(arguments or {})
        decision = self.permission_policy.evaluate(tool_name, args)

        if decision.action == "deny":
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"permission denied: {decision.reason}",
                status="denied",
                risk=decision.risk,
            )

        if decision.action == "ask" and not approved:
            pending = self.approval_store.create(
                tool_name=tool_name,
                arguments=args,
                risk=decision.risk,
                reason=decision.reason,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="approval required",
                status="approval_required",
                risk=decision.risk,
                approval_id=pending.id,
            )

        result = self._run(tool_name, args)
        return ToolResult(
            tool_name=result.tool_name,
            success=result.success,
            output=result.output,
            error=result.error,
            status="executed" if result.success else "failed",
            risk=decision.risk,
        )

    def approve(self, approval_id: str) -> ToolResult:
        pending = self.approval_store.pop(approval_id)
        if pending is None:
            return ToolResult(
                tool_name="",
                success=False,
                error=f"unknown approval id: {approval_id}",
                status="missing",
            )
        return self.execute(
            pending.tool_name,
            pending.arguments,
            approved=True,
        )

    def deny(self, approval_id: str) -> bool:
        return self.approval_store.pop(approval_id) is not None
