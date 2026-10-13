"""Regression for S05 write evidence and real post-approval readback."""
from __future__ import annotations

from app.agent import AgentLoop
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.permissions import PermissionPolicy
from tests.test_planning import ScriptedLLM


def real_agent(tmp_path, decisions):
    return AgentLoop(
        llm=ScriptedLLM(decisions),
        tool_registry=build_default_tool_registry(
            skill_registry=SkillRegistry(), workspace_root=tmp_path,
        ),
        permission_policy=PermissionPolicy(),
        planning_enabled=True,
        plan_require_file_evidence=True,
    )


def write_and_read_plan(agent):
    return agent.planner.create(
        "创建 demo_s05.txt，写入 hello evome，然后读取文件确认内容正确",
        [
            "使用 write_file 创建 demo_s05.txt，内容为 hello evome",
            "使用 read_file 读取 demo_s05.txt，确认内容为 hello evome",
        ],
    )


def test_approved_write_step_succeeds_without_unnecessary_read(tmp_path):
    agent = real_agent(tmp_path, [
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "demo_s05.txt", "content": "hello evome"}},
        {"action": "final", "task_complete": True, "answer": "写入成功"},
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "demo_s05.txt"}},
        {"action": "final", "task_complete": True, "answer": "内容一致"},
    ])
    plan = write_and_read_plan(agent)
    assert agent.planner.run(plan.id).status == "waiting_approval"
    approval_id = plan.steps[0].approval_id
    result = agent.executor.approve(approval_id)
    assert result.success
    assert result.output["bytes"] == 11
    assert agent.planner.on_approval(approval_id, result)
    assert plan.status == "pending"
    assert plan.completed_count == 0
    # Resumes the same step, marks the successfully written file complete,
    # then executes the separate readback Todo.
    assert agent.planner.run(plan.id).status == "completed"
    assert plan.completed_count == 2
    assert (tmp_path / "demo_s05.txt").read_text(encoding="utf-8") == "hello evome"
    calls = [
        e.tool_call.name for step in plan.steps for e in step.tool_steps
        if e.tool_call is not None and not e.error
    ]
    assert calls == ["write_file", "read_file"]


def test_mismatched_readback_content_does_not_complete_plan(tmp_path):
    agent = real_agent(tmp_path, [
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "demo_s05.txt", "content": "hello evome"}},
        {"action": "final", "task_complete": True, "answer": "写入成功"},
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "demo_s05.txt"}},
        {"action": "final", "task_complete": True, "answer": "内容一致"},
    ])
    plan = write_and_read_plan(agent)
    agent.planner.run(plan.id)
    result = agent.executor.approve(plan.steps[0].approval_id)
    assert agent.planner.on_approval(plan.steps[0].approval_id, result)
    # File drift between approval and verification must be detected.
    (tmp_path / "demo_s05.txt").write_text("tampered", encoding="utf-8")
    assert agent.planner.run(plan.id).status == "failed"
    assert plan.completed_count == 1
    assert plan.steps[0].status == "completed"
    assert plan.steps[1].status == "failed"
    assert "missing_file_evidence" in plan.error


def test_without_a_successful_write_cannot_claim_creation(tmp_path):
    agent = real_agent(tmp_path, [
        {"action": "final", "task_complete": True, "answer": "文件已创建"},
    ])
    plan = agent.planner.create("新建文件", ["使用 write_file 创建 demo_s05.txt"])
    assert agent.planner.run(plan.id).status == "failed"
    assert not (tmp_path / "demo_s05.txt").exists()


def test_write_to_different_file_cannot_satisfy_target(tmp_path):
    agent = real_agent(tmp_path, [
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "other.txt", "content": "hello evome"}},
        {"action": "final", "task_complete": True, "answer": "已创建目标文件"},
    ])
    plan = agent.planner.create(
        "创建目标", ["使用 write_file 创建 demo_s05.txt"],
    )
    agent.planner.run(plan.id)
    approved = agent.executor.approve(plan.steps[0].approval_id)
    assert agent.planner.on_approval(plan.steps[0].approval_id, approved)
    assert agent.planner.run(plan.id).status == "failed"
    assert (tmp_path / "other.txt").exists()
    assert not (tmp_path / "demo_s05.txt").exists()
