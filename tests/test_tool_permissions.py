from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.executor import ToolExecutor
from app.tools.permissions import ApprovalStore, PermissionPolicy
from app.tools.registry import ToolRegistry


def registry():
    reg = ToolRegistry()
    for name in ("read_file", "write_file", "bash"):
        reg.register_tool(ToolDefinition(name=name, description=name))
        reg.register_implementation(
            ToolImplementation(
                name=f"{name}_v1",
                tool_name=name,
                handler=lambda **kwargs: {"ok": True, **kwargs},
            )
        )
    return reg


def test_read_only_tool_is_allowed():
    ex = ToolExecutor(registry(), permission_policy=PermissionPolicy())
    result = ex.execute("read_file", {"path": "x.txt"})
    assert result.success
    assert result.status == "executed"
    assert result.risk == "low"


def test_write_requires_approval_then_executes():
    store = ApprovalStore()
    ex = ToolExecutor(
        registry(),
        permission_policy=PermissionPolicy(),
        approval_store=store,
    )
    pending = ex.execute("write_file", {"path": "x.txt", "content": "hi"})
    assert pending.status == "approval_required"
    assert pending.approval_id

    result = ex.approve(pending.approval_id)
    assert result.success
    assert result.status == "executed"


def test_denied_bash_never_executes():
    ex = ToolExecutor(
        registry(),
        permission_policy=PermissionPolicy(bash_mode="deny"),
    )
    result = ex.execute("bash", {"command": "echo hi"})
    assert not result.success
    assert result.status == "denied"


def test_unknown_tool_is_denied_by_policy():
    ex = ToolExecutor(registry(), permission_policy=PermissionPolicy())
    result = ex.execute("mystery", {})
    assert not result.success
    assert result.status == "denied"
