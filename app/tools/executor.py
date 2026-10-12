from __future__ import annotations

from typing import Any

from app.hooks.manager import HookContext, HookManager
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
        hook_manager: HookManager | None = None,
    ) -> None:
        self.registry = registry
        self.permission_policy = permission_policy
        self.approval_store = approval_store or ApprovalStore()
        self.hook_manager = hook_manager

    def _run(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        risk: str = "low",
    ) -> ToolResult:
        if self.hook_manager is not None:
            report = self.hook_manager.dispatch(HookContext(
                event="before_tool",
                tool_name=tool_name,
                risk=risk,
                arguments=dict(arguments),
            ))
            if not report.allowed:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    status="denied",
                    risk=risk,
                    error=f"hook denied: {report.reason}",
                )
        try:
            implementation = self.registry.resolve(tool_name)
            output = implementation.execute(**arguments)
            result = ToolResult(
                tool_name=tool_name, success=True, output=output,
                status="executed", risk=risk,
            )
        except Exception as exc:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                status="failed", risk=risk,
            )
        if self.hook_manager is not None:
            self.hook_manager.dispatch(HookContext(
                event="after_tool",
                tool_name=tool_name,
                risk=risk,
                status=result.status,
            ))
        return result

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        approved: bool = False,
    ) -> ToolResult:
        args = dict(arguments or {})
        if self.permission_policy is None:
            return self._run(tool_name, args)

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

        return self._run(tool_name, args, risk=decision.risk)

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
