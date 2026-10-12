from __future__ import annotations

from app.agent import AgentLoop
from app.hooks import HookContext, HookDecision, HookManager, protect_sensitive_paths, build_sensitive_path_guard
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.executor import ToolExecutor
from app.tools.permissions import PermissionPolicy
from app.tools.registry import ToolRegistry


def make_registry(handler=None):
    registry = ToolRegistry()
    for name in ("read_file", "write_file", "bash"):
        registry.register_tool(ToolDefinition(name=name, description=name))
        registry.register_implementation(ToolImplementation(
            name=f"test_{name}",
            tool_name=name,
            handler=handler or (lambda **kwargs: kwargs),
        ))
    return registry


def test_priority_and_order():
    seen = []
    manager = HookManager()
    manager.register("before_tool", lambda ctx: seen.append("normal"), name="normal")
    manager.register("before_tool", lambda ctx: seen.append("first"), name="first", priority=10)
    manager.register("before_tool", lambda ctx: seen.append("last"), name="last")
    assert manager.dispatch(HookContext("before_tool")).allowed
    assert seen == ["first", "normal", "last"]


def test_pre_hook_veto_prevents_actual_execution():
    called = []
    manager = HookManager()
    manager.register(
        "before_tool",
        lambda ctx: HookDecision(False, "blocked by policy"),
        name="guard",
    )
    executor = ToolExecutor(
        make_registry(lambda **kwargs: called.append("executed")),
        permission_policy=PermissionPolicy(),
        hook_manager=manager,
    )
    result = executor.execute("read_file", {"path": "README.md"})
    assert result.status == "denied"
    assert "blocked by policy" in result.error
    assert called == []
    assert any(x.outcome == "blocked" for x in manager.recent())


def test_pre_hook_exception_fails_closed():
    manager = HookManager()

    def crash(ctx):
        raise RuntimeError("guard failed")

    manager.register("before_tool", crash)
    executor = ToolExecutor(make_registry(), hook_manager=manager)
    result = executor.execute("read_file", {"path": "README.md"})
    assert result.status == "denied"
    assert "guard failed" in result.error


def test_post_hook_exception_does_not_undo_tool():
    manager = HookManager()

    def crash(ctx):
        raise RuntimeError("observer failed")

    manager.register("after_tool", crash)
    executor = ToolExecutor(make_registry(), hook_manager=manager)
    result = executor.execute("read_file", {"path": "README.md"})
    assert result.success
    assert any(x.outcome == "error" and x.event == "after_tool" for x in manager.recent())


def test_hook_history_does_not_include_arguments():
    manager = HookManager(history_limit=3)
    executor = ToolExecutor(make_registry(), hook_manager=manager)
    executor.execute("read_file", {"path": "SECRET_CANARY.env"})
    assert len(manager.recent()) <= 3
    assert "SECRET_CANARY" not in repr(manager.recent())


def test_default_secret_file_hook_blocks_direct_access():
    manager = HookManager()
    manager.register("before_tool", protect_sensitive_paths, name="sensitive")
    executor = ToolExecutor(
        make_registry(), permission_policy=PermissionPolicy(), hook_manager=manager
    )
    assert executor.execute("read_file", {"path": ".env"}).status == "denied"
    assert executor.execute("read_file", {"path": ".git/config"}).status == "denied"
    assert executor.execute("read_file", {"path": ".env.example"}).success
    assert executor.execute("read_file", {"path": "README.md"}).success


def test_sensitive_guard_blocks_resolved_workspace_escape(tmp_path):
    manager = HookManager()
    manager.register("before_tool", build_sensitive_path_guard(tmp_path))
    executor = ToolExecutor(make_registry(), hook_manager=manager)
    result = executor.execute("read_file", {"path": "../.env"})
    assert result.status == "denied"
    assert "escapes workspace" in result.error


def test_s3_approval_cannot_be_bypassed_by_hook():
    manager = HookManager()
    manager.register("before_tool", lambda ctx: HookDecision(True), name="allow")
    called = []
    executor = ToolExecutor(
        make_registry(lambda **kwargs: called.append(kwargs)),
        permission_policy=PermissionPolicy(),
        hook_manager=manager,
    )
    pending = executor.execute("write_file", {"path": "demo.txt", "content": "hello"})
    assert pending.status == "approval_required"
    assert called == []
    result = executor.approve(pending.approval_id)
    assert result.success
    assert called == [{"path": "demo.txt", "content": "hello"}]
    assert any(x.event == "after_tool" for x in manager.recent())


def test_hook_does_not_override_bash_permission_deny():
    manager = HookManager()
    manager.register("before_tool", lambda ctx: HookDecision(True))
    executor = ToolExecutor(
        make_registry(),
        permission_policy=PermissionPolicy(bash_mode="deny"),
        hook_manager=manager,
    )
    assert executor.execute("bash", {"command": "echo nope"}).status == "denied"


class ScriptedLLM:
    def json_chat(self, messages, **kwargs):
        return {"action": "final", "answer": "ok"}


def test_agent_lifecycle_emits_before_and_after():
    manager = HookManager()
    agent = AgentLoop(llm=ScriptedLLM(), tool_registry=make_registry(), hook_manager=manager)
    result = agent.run(user_input="hello")
    assert result.answer == "ok"
    history = manager.recent()
    assert [x.event for x in history] == ["before_agent", "after_agent"]
    assert history[-1].status == "final_answer"


def test_agent_hook_can_block_without_calling_llm():
    class NeverLLM:
        def json_chat(self, messages, **kwargs):
            raise AssertionError("LLM should never run")

    manager = HookManager()
    manager.register("before_agent", lambda ctx: HookDecision(False, "maintenance"))
    agent = AgentLoop(llm=NeverLLM(), tool_registry=make_registry(), hook_manager=manager)
    result = agent.run(user_input="hello")
    assert result.termination_reason == "blocked_by_hook"
    assert "maintenance" in result.answer
    assert manager.recent()[-1].event == "after_agent"
