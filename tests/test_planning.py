"""S05: exercise explicit planning, autonomous plan decisions and S03/S04 boundaries."""
from __future__ import annotations

import pytest

from app.agent import AgentLoop
from app.agent.planning import PlanStore
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookManager
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.permissions import PermissionPolicy
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.chat_calls = 0

    def json_chat(self, messages, **kwargs):
        self.requests.append(messages)
        if not self.decisions:
            raise AssertionError("unexpected additional planning/agent decision")
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        return "no progress fallback"


def make_registry(calls):
    reg = ToolRegistry()
    for name in ("glob", "read_file", "write_file"):
        reg.register_tool(ToolDefinition(name=name, description=name))
        reg.register_implementation(ToolImplementation(
            name=f"test_{name}",
            tool_name=name,
            handler=lambda _name=name, **kwargs: calls.append((_name, kwargs)) or {"ok": True},
        ))
    return reg


def make_agent(
    decisions,
    *,
    hook_manager=None,
    permission_policy=None,
    max_tasks=5,
    replans=1,
    auto_replan=False,
    step_max_steps=4,
    continuations=2,
    handler_calls=None,
):
    calls = [] if handler_calls is None else handler_calls
    agent = AgentLoop(
        llm=ScriptedLLM(decisions),
        tool_registry=make_registry(calls),
        hook_manager=hook_manager,
        permission_policy=permission_policy,
        planning_enabled=True,
        plan_max_tasks=max_tasks,
        plan_step_max_steps=step_max_steps,
        plan_step_max_continuations=continuations,
        plan_max_replans=replans,
        plan_auto_replan=auto_replan,
    )
    return agent, calls


def test_create_plan_validates_step_budget():
    agent, _ = make_agent([], max_tasks=2)
    planner = agent.planner
    plan = planner.create("查看项目", ["列出文件", "检查文件"])
    assert plan.status == "pending"
    assert [x.number for x in plan.steps] == [1, 2]
    with pytest.raises(ValueError):
        planner.create("too many", ["a", "b", "c"])
    with pytest.raises(ValueError):
        planner.create("invalid", [""])


def test_generate_plan_then_execute_in_order():
    agent, calls = make_agent([
        {"steps": [{"title": "列出 Python 文件"}, {"title": "读取 README"}]},
        {"action": "tool", "tool_name": "glob", "arguments": {"pattern": "**/*.py"}},
        {"action": "final", "answer": "找到文件"},
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "README.md"}},
        {"action": "final", "answer": "README 分析完成"},
    ], permission_policy=PermissionPolicy())
    planner = agent.planner
    plan = planner.create_from_llm("检查仓库目录")
    result = planner.run(plan.id)
    assert result.status == "completed"
    assert result.completed_count == 2
    assert [name for name, _ in calls] == ["glob", "read_file"]
    assert "README 分析完成" in planner.describe(plan)
    assert "规划器" in agent.llm.requests[0][0]["content"]


def test_autonomous_plan_action_creates_plan_and_executes():
    agent, _ = make_agent([
        {"action": "plan", "steps": ["理解结构", "指出问题"]},
        {"action": "final", "answer": "已读取结构"},
        {"action": "final", "answer": "有两个潜在问题"},
    ])
    result = agent.run(user_input="帮我分析架构并提出建议")
    assert result.termination_reason == "plan_completed"
    assert result.plan_id
    assert agent.planner.get(result.plan_id).completed_count == 2


def test_write_plan_pauses_for_approval_then_resumes_without_duplicate():
    hooks = HookManager()
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "demo.txt", "content": "ok"}},
        {"action": "final", "answer": "最后检查完成"},
    ], permission_policy=PermissionPolicy(), hook_manager=hooks)
    plan = agent.planner.create("创建并检查", ["创建一个文件", "检查结果"])
    agent.planner.run(plan.id)
    assert plan.status == "waiting_approval"
    assert calls == []
    approval_id = plan.steps[0].approval_id
    assert approval_id
    assert agent.planner.run(plan.id).status == "waiting_approval"

    result = agent.executor.approve(approval_id)
    assert result.success
    assert agent.planner.on_approval(approval_id, result)
    assert plan.status == "pending"
    assert calls == [("write_file", {"path": "demo.txt", "content": "ok"})]
    assert agent.planner.run(plan.id).status == "completed"
    assert len(calls) == 1
    assert any(rec.event == "before_tool" for rec in hooks.recent())


def test_denied_approval_fails_plan():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "demo.txt", "content": "ok"}},
    ], permission_policy=PermissionPolicy())
    plan = agent.planner.create("create", ["write"])
    agent.planner.run(plan.id)
    approval_id = plan.steps[0].approval_id
    assert agent.executor.deny(approval_id)
    agent.planner.on_approval(approval_id, None, denied=True)
    assert plan.status == "failed"
    assert calls == []


def test_failed_step_does_not_claim_completion_and_can_replan():
    agent, _ = make_agent([
        {"action": "bad"},
        {"action": "bad"},
        {"steps": ["改为分解任务"]},
        {"action": "final", "answer": "新步骤完成"},
    ], step_max_steps=2)
    plan = agent.planner.create("inspect", ["第一次尝试"])
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert plan.completed_count == 0
    agent.planner.replan(plan.id)
    assert plan.replans == 1
    assert plan.steps[0].status == "skipped"
    assert agent.planner.run(plan.id).status == "completed"
    with pytest.raises(ValueError, match="only failed"):
        agent.planner.replan(plan.id)


def test_failed_side_effect_tool_is_not_auto_replanned():
    calls = []

    def explode(_name, kwargs):
        calls.append((_name, kwargs))
        raise RuntimeError("disk I/O failed")

    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="write_file", description="write"))
    registry.register_implementation(ToolImplementation(
        name="write", tool_name="write_file",
        handler=lambda **kwargs: explode("write_file", kwargs)
    ))
    agent = AgentLoop(
        llm=ScriptedLLM([
            {"action": "tool", "tool_name": "write_file", "arguments": {"path": "x"}},
            {"action": "final", "answer": "write failed"},
        ]),
        tool_registry=registry,
        planning_enabled=True,
        plan_auto_replan=True,
    )
    plan = agent.planner.create("write", ["write"])
    assert agent.planner.run(plan.id).status == "failed"
    assert plan.replans == 0
    assert len(calls) == 1


def test_cancel_and_store_capacity():
    agent, _ = make_agent([])
    plan = agent.planner.create("goal", ["one"])
    assert agent.planner.cancel(plan.id).status == "cancelled"
    assert agent.planner.run(plan.id).status == "cancelled"
    store = PlanStore(capacity=1)
    store.add(plan)
    second = agent.planner.create("next", ["two"])
    store.add(second)
    assert [item.id for item in store.list()] == [second.id]


def test_plan_commands_are_operational_not_learning_data():
    for value in (
        "/plans", "plan test", "/plan-show abc", "/plan-run abc",
        "/plan-replan abc", "/plan-cancel abc",
    ):
        assert looks_like_operational_input(value)
    assert normalize_cli_input("plan 检查项目") == "/plan 检查项目"


def test_full_notes_are_stored_displayed_and_available_to_next_step():
    note = "源码分析。" * 150 + "关键证据：loop 调用 executor。"
    final = "架构结论。" * 150 + "最后一条改进建议。"
    agent, _ = make_agent([
        {"action": "final", "answer": note},
        {"action": "final", "answer": final},
    ])
    plan = agent.planner.create("分析", ["读取源码", "综合建议"])
    agent.planner.run(plan.id)
    assert plan.steps[0].note == note
    assert plan.steps[1].note == final
    assert note in str(agent.llm.requests[1])
    summary = agent.planner.describe(plan)
    assert "已省略" in summary
    assert final in summary
    assert "/plan-show " + plan.id in summary
    detail = agent.planner.describe(plan, detail=True)
    assert note in detail and final in detail


def test_replan_replaces_all_old_pending_tasks_but_keeps_history():
    agent, calls = make_agent([
        {"action": "final", "answer": "完成目录检查"},
        {"action": "bad"},
        {"steps": ["替代的读取步骤", "新的综合步骤"]},
        {"action": "final", "answer": "新读取完成"},
        {"action": "final", "answer": "新综合完成"},
    ], step_max_steps=1)
    plan = agent.planner.create("分析", ["目录检查", "失败读取", "旧读取", "旧综合"])
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    agent.planner.replan(plan.id)
    assert [s.status for s in plan.steps] == ["completed", "skipped", "skipped", "skipped", "pending", "pending"]
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert [s.attempts for s in plan.steps] == [1, 1, 0, 0, 1, 1]
    assert plan.completed_count == 3


def test_plan_can_read_four_files_and_finish_with_four_tool_budget():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": str(i)}}
        for i in range(4)
    ] + [
        {"action": "final", "task_complete": True, "answer": "四个文件分析完成"},
        {"action": "final", "answer": "三个改进建议"},
    ])
    plan = agent.planner.create("分析", ["读四个文件", "综合建议"])
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert len(calls) == 4
    assert plan.completed_count == 2


def test_incomplete_budget_summary_pauses_with_checkpoint():
    agent, _ = make_agent([
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "a.py"}},
        {"action": "final", "task_complete": False, "answer": "还缺 b.py，不能完成对比"},
    ], step_max_steps=1, continuations=0)
    plan = agent.planner.create("对比", ["读取两个文件"])
    agent.planner.run(plan.id)
    assert plan.status == "paused"
    assert plan.completed_count == 0
    assert "还缺 b.py" in plan.steps[0].note
    assert plan.steps[0].checkpoint is not None
    assert "/plan-run " + plan.id in agent.planner.describe(plan)
    assert "/plan-replan " + plan.id not in agent.planner.describe(plan)


def test_runtime_plan_show_forwards_full_detail():
    from types import SimpleNamespace
    from app.runtime import EvoMeRuntime

    agent, _ = make_agent([])
    plan = agent.planner.create("分析", ["说明"])
    plan.steps[0].note = "完整记录" * 200
    runtime = object.__new__(EvoMeRuntime)
    runtime.strategy = SimpleNamespace(agent_loop=agent)
    assert plan.steps[0].note in runtime.describe_plan(plan.id, detail=True)
