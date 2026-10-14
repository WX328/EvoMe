from __future__ import annotations

from typing import Any

from app.hooks.manager import HookContext, HookManager
from app.tools.base import ToolResult
from app.tools.permissions import ApprovalStore, PermissionPolicy
from app.tools.registry import ToolRegistry
from app.telemetry.tracing import current_trace, traced


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
        self._file_guard = None
        if getattr(registry, "workspace_root", None) is not None:
            from app.hooks.builtin import build_sensitive_path_guard
            self._file_guard = build_sensitive_path_guard(registry.workspace_root)

    def _run(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        risk: str = "low",
    ) -> ToolResult:
        # Resolve against the actual file-tool workspace, even if an embedder
        # passed a policy with a different cwd or disabled lifecycle hooks.
        if self._file_guard is not None:
            try:
                verdict = self._file_guard(HookContext(event="before_tool", tool_name=tool_name, arguments=arguments))
            except (OSError, ValueError, RuntimeError):
                verdict = None
            if verdict is None or not verdict.allow:
                return ToolResult(tool_name=tool_name, success=False, status="denied", risk="high",
                    error="PermissionError: protected or unresolvable file target")
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

    @traced("tool", "execute")
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
            try:
                impl = self.registry.resolve(tool_name)
                binding = (id(impl.handler), impl.name, impl.version)
            except LookupError:
                return ToolResult(tool_name=tool_name, success=False,
                    status="failed", error="tool implementation unavailable", risk=decision.risk)
            pending = self.approval_store.create(
                tool_name=tool_name,
                arguments=args,
                risk=decision.risk,
                reason=decision.reason,
                implementation_token=binding,
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
        context = pending.trace_context or current_trace()
        if context:
            with context.recorder.span("approval", "approve", reference=approval_id, parent=context) as span:
                result = self._approve_pending(pending)
                span["status"] = result.status
                return result
        return self._approve_pending(pending)

    def _approve_pending(self, pending) -> ToolResult:
        if self.approval_store.expired(pending):
            return ToolResult(tool_name=pending.tool_name, success=False,
                status="expired", error="approval expired; submit a new request")
        try:
            impl = self.registry.resolve(pending.tool_name)
            binding = (id(impl.handler), impl.name, impl.version)
        except LookupError:
            binding = None
        if binding != pending.implementation_token:
            return ToolResult(tool_name=pending.tool_name, success=False,
                status="invalidated", error="tool changed after request; new approval required")
        # Re-evaluate current permissions and hooks after consuming the grant.
        return self.execute(pending.tool_name, pending.arguments, approved=True)

    def deny(self, approval_id: str) -> bool:
        pending = self.approval_store.pop(approval_id)
        if pending and pending.trace_context:
            context = pending.trace_context
            with context.recorder.span("approval", "deny", reference=approval_id, parent=context) as span:
                span["status"] = "denied"
        return pending is not None
