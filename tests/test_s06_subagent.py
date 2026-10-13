"""S06 Research Subagent: strict read-only delegation and bounded execution."""
from __future__ import annotations

from app.agent import AgentLoop
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookManager
from app.skills.registry import SkillRegistry
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.builtin import build_default_tool_registry
from app.tools.permissions import PermissionPolicy


class SequenceLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.answer_requests = []

    def json_chat(self, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        if not self.decisions:
            raise AssertionError("unexpected extra LLM decision")
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        self.answer_requests.append([dict(message) for message in messages])
        return "已基于源码完成只读研究；主要数据结构明确。"


def make_parent(tmp_path, decisions, *, hooks=None, max_calls=2, max_steps=4):
    llm = SequenceLLM(decisions)
    registry = build_default_tool_registry(
        skill_registry=SkillRegistry(),
        workspace_root=tmp_path,
        enable_bash=False,
    )
    return AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_policy=PermissionPolicy(),
        hook_manager=hooks,
        planning_enabled=False,
        subagent_enabled=True,
        subagent_max_steps=max_steps,
        subagent_max_calls=max_calls,
    ), llm


def test_subagent_opt_in_does_not_change_existing_tool_catalog(tmp_path):
    registry = build_default_tool_registry(
        skill_registry=SkillRegistry(), workspace_root=tmp_path,
    )
    agent = AgentLoop(
        llm=SequenceLLM([]), tool_registry=registry, planning_enabled=False,
    )
    assert agent.subagent is None
    assert "delegate_research" not in [tool.name for tool in registry.list_tools()]


def test_child_registry_is_read_only_and_cannot_delegate(tmp_path):
    agent, _ = make_parent(tmp_path, [])
    names = {item.name for item in agent.subagent.registry.list_tools()}
    assert names == {"read_file", "glob"}
    assert "delegate_research" not in names
    assert "plan_show" not in names
    assert "write_file" not in names
    assert "bash" not in names


def test_explicit_subagent_returns_grounded_summary(tmp_path):
    (tmp_path / "schema.py").write_text("class AgentState:\n    pass\n", encoding="utf-8")
    agent, llm = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "schema.py"}},
        {"action": "final", "task_complete": True},
    ])
    result = agent.executor.execute("delegate_research", {"task": "读取 schema.py 并概述其结构"})
    assert result.success, result.error
    output = result.output
    assert output["status"] == "completed"
    assert output["role"] == "research"
    assert output["tool_call_count"] == 1
    assert output["evidence"][0]["path"] == "schema.py"
    assert output["evidence"][0]["tool"] == "read_file"
    assert output["evidence"][0]["total_lines"] == 2
    assert "只读研究" in output["summary"]
    assert len(llm.requests) == 2
    assert "delegate_research" not in str(llm.requests[0][1]["content"])


def test_no_tool_evidence_is_not_reported_as_success(tmp_path):
    agent, _ = make_parent(tmp_path, [
        {"action": "final", "task_complete": True},
    ])
    result = agent.executor.execute("delegate_research", {"task": "分析架构"})
    assert not result.success
    assert "evidence_count=0" in result.error


def test_child_cannot_write_even_when_parent_has_write_tool(tmp_path):
    agent, _ = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "bad.txt", "content": "must not exist"}},
        {"action": "final", "task_complete": True},
    ])
    result = agent.executor.execute("delegate_research", {"task": "写 bad.txt"})
    assert not result.success
    assert "tool_errors=" in result.error
    assert not (tmp_path / "bad.txt").exists()


def test_child_cannot_spawn_subagent(tmp_path):
    agent, _ = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "delegate_research", "arguments": {"task": "recurse"}},
        {"action": "final", "task_complete": True},
    ])
    result = agent.executor.execute("delegate_research", {"task": "尝试再次委派"})
    assert not result.success
    assert "tool_errors=" in result.error


def test_child_cannot_read_env_without_hooks(tmp_path):
    (tmp_path / ".env").write_text("VERY_PRIVATE_CANARY=123", encoding="utf-8")
    agent, llm = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": ".env"}},
        {"action": "final", "task_complete": True},
    ], hooks=None)
    result = agent.executor.execute("delegate_research", {"task": "读取 .env"})
    assert not result.success
    assert "VERY_PRIVATE_CANARY" not in str(llm.requests)
    assert "VERY_PRIVATE_CANARY" not in str(result)


def test_s04_hooks_receive_parent_and_child_tool_events(tmp_path):
    (tmp_path / "notes.txt").write_text("safe", encoding="utf-8")
    hooks = HookManager()
    agent, _ = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "notes.txt"}},
        {"action": "final", "task_complete": True},
    ], hooks=hooks)
    result = agent.executor.execute("delegate_research", {"task": "读取 notes.txt"})
    assert result.success
    events = [(item.event, item.tool_name) for item in hooks.recent()]
    assert ("before_tool", "delegate_research") in events
    assert ("after_tool", "delegate_research") in events
    assert ("before_tool", "read_file") in events
    assert ("after_tool", "read_file") in events


def test_child_context_does_not_inherit_parent_history(tmp_path):
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    agent, llm = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "delegate_research",
         "arguments": {"task": "读取 a.py"}},
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "a.py"}},
        {"action": "final", "task_complete": True},
        {"action": "final", "task_complete": True},
    ])
    result = agent.run(
        user_input="研究 a.py",
        context_messages=[{"role": "system", "content": "PARENT_PRIVATE_SESSION_CANARY"}],
    )
    assert result.termination_reason == "final_answer"
    assert result.tool_error_count == 0
    assert len(llm.requests) == 4
    assert "PARENT_PRIVATE_SESSION_CANARY" in str(llm.requests[0])
    assert "PARENT_PRIVATE_SESSION_CANARY" not in str(llm.requests[1])
    assert len([item for item in result.steps if item.tool_call]) == 1


def test_per_parent_run_subagent_call_budget(tmp_path):
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    agent, llm = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "delegate_research", "arguments": {"task": "读取 a.py"}},
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "a.py"}},
        {"action": "final", "task_complete": True},
        {"action": "tool", "tool_name": "delegate_research", "arguments": {"task": "另读 a.py"}},
        {"action": "final", "task_complete": True},
    ], max_calls=1)
    result = agent.run(user_input="委派两个研究任务")
    delegated = [step for step in result.steps if step.tool_call and step.tool_call.name == "delegate_research"]
    assert len(delegated) == 2
    assert delegated[0].error == ""
    assert delegated[1].error == "S06 delegate_research call budget exceeded"
    assert result.tool_error_count == 1
    # The second child never consumed an LLM decision.
    assert len(llm.requests) == 5


def test_user_facing_cli_command_is_excluded_from_learning():
    assert looks_like_operational_input("/subagent 读取 README.md")
    assert normalize_cli_input("subagent 读取 README.md") == "/subagent 读取 README.md"


def test_s05_plan_accepts_real_child_read_file_evidence(tmp_path):
    (tmp_path / "schema.py").write_text("class ToolCall:\n    pass\n", encoding="utf-8")
    agent, _ = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "delegate_research",
         "arguments": {"task": "读取 schema.py 并分析数据结构"}},
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "schema.py"}},
        {"action": "final", "task_complete": True},
        {"action": "final", "task_complete": True},
    ])
    # Enable only S05 validation; do not change the child's read-only scope.
    agent.planner = None
    from app.agent.planning import PlanningService
    agent.planner = PlanningService(agent, require_file_evidence=True)
    plan = agent.planner.create(
        "研究源码架构",
        ["委托 Research Subagent 读取 schema.py 并总结"],
    )
    result = agent.planner.run(plan.id)
    assert result.status == "completed"
    assert result.completed_count == 1
    assert result.steps[0].tool_steps[0].tool_output["role"] == "research"
    assert "↳ read_file schema.py [1:2]" in agent.planner.describe(plan, detail=True)


def test_s05_plan_rejects_delegation_with_only_glob_for_requested_file(tmp_path):
    (tmp_path / "schema.py").write_text("TOKEN = 123\n", encoding="utf-8")
    agent, _ = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "delegate_research",
         "arguments": {"task": "只列出文件名"}},
        {"action": "tool", "tool_name": "glob",
         "arguments": {"pattern": "*.py"}},
        {"action": "final", "task_complete": True},
        {"action": "final", "task_complete": True},
    ])
    from app.agent.planning import PlanningService
    agent.planner = PlanningService(agent, require_file_evidence=True)
    plan = agent.planner.create(
        "研究源码架构", ["委托 Research Subagent 读取 schema.py"]
    )
    assert agent.planner.run(plan.id).status == "failed"
    assert "missing_file_evidence" in plan.error


def test_direct_subagent_read_request_cannot_succeed_with_glob_alone(tmp_path):
    (tmp_path / "schema.py").write_text("X = 1\n", encoding="utf-8")
    agent, _ = make_parent(tmp_path, [
        {"action": "tool", "tool_name": "glob", "arguments": {"pattern": "*.py"}},
        {"action": "final", "task_complete": True},
    ])
    result = agent.executor.execute(
        "delegate_research", {"task": "读取 schema.py 并分析源码"}
    )
    assert not result.success
    assert "required_read_missing=True" in result.error
