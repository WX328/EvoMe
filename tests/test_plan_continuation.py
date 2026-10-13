import pytest

from app.agent import AgentLoop
from app.hooks import HookManager
from app.hooks.manager import HookDecision
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.filesystem import FileSystemTools
from app.tools.permissions import PermissionPolicy
from tests.test_planning import make_agent


def read(path, **kwargs):
    return {"action": "tool", "tool_name": "read_file", "arguments": {"path": path, **kwargs}}


def partial():
    return {"action": "final", "task_complete": False, "answer": "已读取部分源码，需要继续读取剩余文件"}


def test_read_only_step_continues_without_losing_results_or_replaying_calls():
    agent, calls = make_agent([
        read("a.py"), read("b.py"), partial(),
        read("c.py"), {"action": "final", "answer": "三个文件读取完成"},
        {"action": "final", "answer": "三个改进点"},
    ], step_max_steps=2)
    plan = agent.planner.create("分析", ["读取源码", "总结"])
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert [args["path"] for _, args in calls] == ["a.py", "b.py", "c.py"]
    assert plan.steps[0].attempts == 2
    resumed_prompt = str(agent.llm.requests[3])
    assert "a.py" in resumed_prompt and "b.py" in resumed_prompt and "[TOOL_RESULT]" in resumed_prompt
    assert "只能基于下方实际工具结果回答" not in resumed_prompt
    assert plan.steps[0].checkpoint is None
    assert "read_file a.py" in agent.planner.describe(plan, detail=True)
    indices = [entry.index for entry in plan.steps[0].tool_steps]
    assert indices == sorted(set(indices))


def test_automatic_continuation_is_bounded_and_plan_run_resumes_from_checkpoint():
    agent, calls = make_agent([
        read("a.py"), partial(), read("b.py"), partial(), read("c.py"), partial(),
        read("d.py"), {"action": "final", "task_complete": True, "answer": "完成"},
    ], step_max_steps=1)
    plan = agent.planner.create("分析", ["读取多个文件"])
    agent.planner.run(plan.id)
    assert plan.status == "paused"
    assert plan.completed_count == 0
    assert len(calls) == 3
    assert plan.steps[0].attempts == 3
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert [args["path"] for _, args in calls] == ["a.py", "b.py", "c.py", "d.py"]
    assert plan.steps[0].attempts == 4


def test_duplicate_call_is_still_blocked_after_resume():
    agent, calls = make_agent([
        read("a.py"), partial(), read("a.py"),
    ], step_max_steps=1)
    plan = agent.planner.create("分析", ["读取"])
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert len(calls) == 1
    assert "duplicate tool call blocked" in agent.planner.describe(plan, detail=True)


def test_successful_write_does_not_trigger_automatic_continuation():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file", "arguments": {"path": "x.txt", "content": "x"}},
        partial(),
    ], step_max_steps=1)
    plan = agent.planner.create("write", ["write and verify"])
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert len(calls) == 1
    assert plan.steps[0].attempts == 1


def test_resumed_step_still_requires_approval_for_write():
    agent, calls = make_agent([
        read("a.py"), partial(),
        {"action": "tool", "tool_name": "write_file", "arguments": {"path": "x.txt", "content": "x"}},
    ], step_max_steps=1, permission_policy=PermissionPolicy())
    plan = agent.planner.create("inspect", ["inspect"])
    agent.planner.run(plan.id)
    assert plan.status == "waiting_approval"
    assert [name for name, _ in calls] == ["read_file"]
    assert plan.steps[0].approval_id


def test_resumed_agent_still_runs_before_agent_hook():
    hooks = HookManager()
    agent, calls = make_agent([
        read("a.py"), partial(), read("b.py"),
        {"action": "final", "task_complete": True, "answer": "done"},
    ], step_max_steps=1, hook_manager=hooks)
    plan = agent.planner.create("inspect", ["inspect"])
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert sum(event.event == "before_agent" for event in hooks.recent()) == 2
    assert sum(event.event == "before_tool" for event in hooks.recent()) == 2


def test_before_agent_hook_can_block_checkpoint_resume():
    hooks = HookManager()
    agent, calls = make_agent([
        read("a.py"), partial(),
    ], step_max_steps=1, continuations=0, hook_manager=hooks)
    plan = agent.planner.create("inspect", ["inspect"])
    agent.planner.run(plan.id)
    assert plan.status == "paused"
    hooks.register("before_agent", lambda ctx: HookDecision(False, "maintenance"))
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert "Hook" in plan.steps[0].note
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["cancel", "replan"])
def test_paused_plan_can_be_cancelled_or_replanned(operation):
    decisions = [read("a.py"), partial()]
    if operation == "replan":
        decisions.append({"steps": ["更小的任务"]})
    agent, _ = make_agent(decisions, step_max_steps=1, continuations=0)
    plan = agent.planner.create("inspect", ["inspect", "summarize"])
    agent.planner.run(plan.id)
    assert plan.status == "paused"
    getattr(agent.planner, operation)(plan.id)
    assert plan.steps[0].checkpoint is None
    assert plan.status == ("cancelled" if operation == "cancel" else "pending")
    assert plan.steps[0].status == ("cancelled" if operation == "cancel" else "skipped")
    assert plan.steps[1].status == plan.steps[0].status


def test_checkpoint_cannot_be_used_for_different_goal():
    agent, _ = make_agent([read("a.py"), partial()], step_max_steps=1, continuations=0)
    plan = agent.planner.create("inspect", ["inspect"])
    agent.planner.run(plan.id)
    with pytest.raises(ValueError, match="different task"):
        agent.run(user_input="unrelated task", resume_from=plan.steps[0].checkpoint)


def test_real_file_tools_read_all_pages_across_multiple_batches(tmp_path):
    # Unlike a one-call stub, this fixture needs more than four real reads.
    files = {
        "long.py": "".join(f"def function_{i}(): return 'source_evidence_{i:03d}'\n" for i in range(100)),
        "small.py": "SMALL_FILE_EVIDENCE = True\n",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    fs = FileSystemTools(workspace_root=tmp_path, max_output_chars=1000)
    reads = []
    for name in files:
        start = 1
        while True:
            page = fs.read_file(name, start_line=start)
            reads.append(read(name, start_line=start))
            if not page["truncated"]:
                break
            assert page["next_start_line"] > start
            start = page["next_start_line"]
    assert 4 < len(reads) <= 12
    decisions = []
    for offset, decision in enumerate(reads, 1):
        decisions.append(decision)
        if offset % 4 == 0 and offset < len(reads):
            decisions.append(partial())
    decisions.append({"action": "final", "task_complete": True, "answer": "完整源码证据已读取"})
    decisions.append({"action": "final", "answer": "三个改进建议"})
    agent, _ = make_agent(decisions)
    registry = build_default_tool_registry(
        skill_registry=SkillRegistry(), workspace_root=tmp_path, max_output_chars=1000,
    )
    agent = AgentLoop(
        llm=agent.llm, tool_registry=registry, permission_policy=PermissionPolicy(),
        planning_enabled=True, plan_step_max_steps=4,
    )
    plan = agent.planner.create("分析源码，不修改文件", ["读取并分析全部源码", "总结三个改进"])
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert plan.steps[0].attempts >= 2
    actual = [entry for entry in plan.steps[0].tool_steps if entry.tool_call]
    assert len(actual) == len(reads)
    contents = "\n".join(entry.tool_output["content"] for entry in actual)
    assert "source_evidence_000" in contents and "source_evidence_099" in contents
    assert "SMALL_FILE_EVIDENCE" in contents
    first_page = actual[0].tool_output
    detail = agent.planner.describe(plan, detail=True)
    assert f"[1:{first_page['end_line']}]" in detail
    assert f"下一段从 {first_page['next_start_line']} 行开始" in detail
    assert all((tmp_path / name).read_text(encoding="utf-8") == text for name, text in files.items())
