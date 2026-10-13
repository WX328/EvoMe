"""S05: approval-result resume, exactly-once effects and source evidence."""
from __future__ import annotations

from app.agent import AgentLoop
from app.tools import ToolDefinition, ToolImplementation, ToolRegistry
from app.tools.permissions import PermissionPolicy
from tests.test_planning import ScriptedLLM, make_agent


def test_approved_write_resumes_same_step_and_requires_verification():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "verification.txt", "content": "v1"}},
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "verification.txt"}},
        {"action": "final", "task_complete": True, "answer": "已读取验证"},
        {"action": "final", "task_complete": True, "answer": "任务结束"},
    ], permission_policy=PermissionPolicy())
    plan = agent.planner.create(
        "写文件并确认", ["写入 verification.txt 并读取验证", "报告最终状态"],
    )
    agent.planner.run(plan.id)
    assert plan.status == "waiting_approval"
    assert plan.completed_count == 0
    approval_id = plan.steps[0].approval_id
    assert approval_id
    assert plan.steps[0].checkpoint is not None
    assert calls == []

    result = agent.executor.approve(approval_id)
    assert result.success
    assert agent.planner.on_approval(approval_id, result)
    assert plan.status == "pending"
    assert plan.completed_count == 0
    assert plan.steps[0].status == "pending"
    assert plan.steps[0].checkpoint is not None
    assert plan.steps[0].tool_steps[0].error == ""

    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert plan.completed_count == 2
    assert calls == [
        ("write_file", {"path": "verification.txt", "content": "v1"}),
        ("read_file", {"path": "verification.txt"}),
    ]
    resumed_input = str(agent.llm.requests[1])
    assert "[TOOL_RESULT]" in resumed_input
    assert "不要再次执行相同操作" in resumed_input


def test_duplicate_write_after_approval_is_blocked_without_second_effect():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "x.txt", "content": "x"}},
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "x.txt", "content": "x"}},
        {"action": "final", "task_complete": True, "answer": "done"},
    ], permission_policy=PermissionPolicy())
    plan = agent.planner.create("write", ["write"])
    agent.planner.run(plan.id)
    approval_id = plan.steps[0].approval_id
    executed = agent.executor.approve(approval_id)
    assert executed.success
    assert not agent.planner.on_approval("incorrect-id", executed)
    assert plan.status == "waiting_approval"
    assert agent.planner.on_approval(approval_id, executed)
    assert agent.planner.run(plan.id).status == "failed"
    assert calls == [("write_file", {"path": "x.txt", "content": "x"})]
    assert any("duplicate tool call blocked" == step.error for step in plan.steps[0].tool_steps)


def test_approval_is_bound_to_pending_tool_and_checkpoint():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "x.txt", "content": "x"}},
    ], permission_policy=PermissionPolicy())
    plan = agent.planner.create("write", ["write"])
    agent.planner.run(plan.id)
    approval_id = plan.steps[0].approval_id
    executed = agent.executor.approve(approval_id)
    assert executed.success
    forged = type(executed)(
        tool_name="read_file",
        success=True,
        output={"ok": True},
        status="executed",
    )
    assert agent.planner.on_approval(approval_id, forged)
    assert plan.status == "failed"
    assert plan.completed_count == 0
    assert calls == [("write_file", {"path": "x.txt", "content": "x"})]


def test_non_boolean_completion_flag_is_not_accepted():
    agent, _ = make_agent([
        {"action": "final", "task_complete": "true", "answer": "wrong type"},
        {"action": "final", "task_complete": True, "answer": "valid"},
    ])
    result = agent.run(user_input="describe")
    assert result.answer == "valid"
    assert any(step.error == "task_complete must be boolean" for step in result.steps)


def make_paged_agent(decisions, outputs):
    llm = ScriptedLLM(decisions)
    output_iter = iter(outputs)
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="read_file", description="read"))
    registry.register_implementation(ToolImplementation(
        name="test_read_file",
        tool_name="read_file",
        handler=lambda **kwargs: next(output_iter),
    ))
    return AgentLoop(
        llm=llm, tool_registry=registry,
        planning_enabled=True, plan_require_file_evidence=True,
    )


def test_named_file_must_really_be_read():
    agent, _ = make_agent([
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "app/tools/base.py"}},
        {"action": "final", "answer": "I read the target"},
    ])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析源码", ["读取 app/agent/loop.py"])
    assert agent.planner.run(plan.id).status == "failed"
    assert plan.error.endswith("missing_file_evidence")


def test_glob_is_not_sufficient_when_file_contents_required():
    agent, _ = make_agent([
        {"action": "tool", "tool_name": "glob",
         "arguments": {"pattern": "app/agent/*.py"}},
        {"action": "final", "answer": "已阅读全部源文件"},
    ])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析源码", ["读取 app/agent/loop.py"])
    assert agent.planner.run(plan.id).status == "failed"


def test_truncated_read_is_not_full_source_evidence():
    agent = make_paged_agent([
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "app/agent/loop.py"}},
        {"action": "final", "answer": "finished"},
    ], [{
        "path": "app/agent/loop.py", "start_line": 1, "end_line": 10,
        "total_lines": 20, "truncated": True, "next_start_line": 11,
        "line_truncated": False, "content": "first 10 lines",
    }])
    plan = agent.planner.create("分析源码", ["读取 app/agent/loop.py"])
    assert agent.planner.run(plan.id).status == "failed"


def test_contiguous_multi_page_read_counts_as_full_source_evidence():
    agent = make_paged_agent([
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "app/agent/loop.py", "start_line": 1}},
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "app/agent/loop.py", "start_line": 11}},
        {"action": "final", "answer": "both pages read"},
    ], [
        {"path": "app/agent/loop.py", "start_line": 1, "end_line": 10,
         "total_lines": 20, "truncated": True, "next_start_line": 11,
         "line_truncated": False, "content": "first"},
        {"path": "app/agent/loop.py", "start_line": 11, "end_line": 20,
         "total_lines": 20, "truncated": False, "next_start_line": None,
         "line_truncated": False, "content": "last"},
    ])
    plan = agent.planner.create("分析源码", ["读取 app/agent/loop.py"])
    assert agent.planner.run(plan.id).status == "completed"


def test_truncated_single_line_cannot_count_as_verified_source():
    agent = make_paged_agent([
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "x.py"}},
        {"action": "final", "answer": "done"},
    ], [{
        "path": "x.py", "start_line": 1, "end_line": 1,
        "total_lines": 1, "truncated": True, "next_start_line": None,
        "line_truncated": True, "content": "partial line",
    }])
    plan = agent.planner.create("分析源码", ["读取 x.py"])
    assert agent.planner.run(plan.id).status == "failed"


def test_summary_can_reuse_previously_verified_named_source_file():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "app/agent/loop.py"}},
        {"action": "final", "task_complete": True, "answer": "已读取"},
        {"action": "final", "task_complete": True, "answer": "已分析"},
    ])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create(
        "分析源码",
        ["读取 app/agent/loop.py", "综合分析 app/agent/loop.py 的结构"],
    )
    assert agent.planner.run(plan.id).status == "completed"
    assert calls == [("read_file", {"path": "app/agent/loop.py"})]
