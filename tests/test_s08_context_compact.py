"""S08 context compaction: model messages shrink; execution records stay intact."""
from __future__ import annotations

import pytest
from app.agent import AgentLoop
from app.agent.compaction import ContextBudgetError, ContextCompactor, MARKER
from app.agent.schema import AgentStep, ToolCall
from app.cli_commands import normalize_cli_input, looks_like_operational_input
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.permissions import PermissionPolicy
from app.tools.registry import ToolRegistry

CANARY = "OLD_TOOL_CONTENT_MUST_NOT_BE_IN_METADATA_917"


class RecordingLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.chat_requests = []

    def json_chat(self, messages, **kwargs):
        self.requests.append([dict(row) for row in messages])
        if not self.decisions:
            raise AssertionError("unexpected Agent decision")
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        self.chat_requests.append([dict(row) for row in messages])
        return "根据当前工具证据完成步骤"


def make_agent(decisions, *, compact=True, planning=False, max_chars=4000):
    calls = []
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="read_file", description="read file"))
    registry.register_implementation(ToolImplementation(
        name="read_v1", tool_name="read_file",
        handler=lambda path: calls.append(("read_file", path)) or {
            "path": path, "start_line": 1, "end_line": 25,
            "total_lines": 25, "truncated": False,
            "content": CANARY + "a" * 7000,
        },
    ))
    registry.register_tool(ToolDefinition(name="write_file", description="write file"))
    registry.register_implementation(ToolImplementation(
        name="write_v1", tool_name="write_file",
        handler=lambda path, content: calls.append(("write_file", path)) or
            {"path": path, "bytes": len(content), "mode": "rewrite"},
    ))
    llm = RecordingLLM(decisions)
    agent = AgentLoop(
        llm=llm, tool_registry=registry,
        permission_policy=PermissionPolicy(),
        planning_enabled=planning,
        context_compact_enabled=compact,
        context_max_chars=max_chars,
        context_keep_recent_messages=2,
    )
    return agent, llm, calls


def read(path):
    return {"action": "tool", "tool_name": "read_file", "arguments": {"path": path}}


def write(path):
    return {"action": "tool", "tool_name": "write_file",
            "arguments": {"path": path, "content": "hello"}}


def test_short_context_is_unchanged():
    c = ContextCompactor(max_chars=4000)
    before = [{"role": "system", "content": "rule"},
              {"role": "user", "content": "goal"}]
    after, report = c.compact(before, user_input="goal")
    assert after == before and report is None
    assert c.stats()["compaction_count"] == 0


def test_context_summary_is_metadata_not_source_text():
    c = ContextCompactor(max_chars=4000, keep_recent_messages=2)
    step = AgentStep(
        index=1, action="tool",
        tool_call=ToolCall(name="read_file", arguments={"path": "a.py"}),
        tool_output={"path": "a.py", "start_line": 1, "end_line": 100,
                     "total_lines": 100, "content": CANARY * 200},
    )
    before = [
        {"role": "system", "content": "PINNED INSTRUCTIONS"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "a" * 6500},
        {"role": "user", "content": "[TOOL_RESULT] " + CANARY * 300},
        {"role": "assistant", "content": "short"},
        {"role": "user", "content": "b" * 6500},
    ]
    after, report = c.compact(before, user_input="goal", steps=(step,))
    assert report is not None and report.after_chars <= 4000
    assert report.saved_chars > 0 and report.messages_omitted >= 2
    assert after[0]["content"] == "PINNED INSTRUCTIONS"
    assert after[1]["content"] == "goal"
    note = next(row["content"] for row in after if row["content"].startswith(MARKER))
    assert "read_file a.py" in note and "end_line=100" in note
    assert CANARY not in note
    assert step.tool_output["content"] == CANARY * 200


def test_required_pinned_instructions_fail_closed():
    with pytest.raises(ContextBudgetError):
        ContextCompactor(max_chars=4000).compact(
            [{"role": "system", "content": "S" * 3800},
             {"role": "user", "content": "original goal"},
             {"role": "user", "content": "x" * 6000}],
            user_input="original goal",
        )


def test_agent_enforces_budget_and_preserves_full_agentstep_evidence():
    agent, llm, calls = make_agent([
        read("a.py"), read("b.py"),
        {"action": "final", "task_complete": True},
    ])
    result = agent.run(user_input="read two files")
    assert result.termination_reason == "final_answer"
    assert calls == [("read_file", "a.py"), ("read_file", "b.py")]
    assert any(s.action == "context_compaction" for s in result.steps)
    assert all(
        sum(len(row["content"]) for row in request) <= 4000
        for request in llm.requests + llm.chat_requests
    )
    full = next(
        s for s in result.steps
        if s.tool_call is not None and s.tool_call.arguments["path"] == "a.py"
    )
    assert CANARY in full.tool_output["content"]
    assert agent.context_compactor.stats()["total_saved_chars"] > 0


def test_checkpoint_resume_does_not_reexecute_previous_tool():
    agent, llm, calls = make_agent([
        read("a.py"),
        {"action": "final", "task_complete": False},
        read("a.py"),
        {"action": "final", "task_complete": True},
    ])
    first = agent.run(user_input="analyze")
    assert first.termination_reason == "incomplete"
    cp = first.checkpoint
    assert cp is not None and len(cp.seen_calls) == 1
    assert cp.compactions > 0 and cp.compacted_chars > 0
    resumed = agent.run(user_input="analyze", resume_from=cp)
    assert calls == [("read_file", "a.py")]
    assert any(s.error == "duplicate tool call blocked" for s in resumed.steps)


def test_approval_resume_preserves_state_after_compaction():
    agent, llm, calls = make_agent([
        read("a.py"), write("x.txt"),
        {"action": "final", "task_complete": True},
    ])
    first = agent.run(user_input="read then write")
    assert first.termination_reason == "approval_required"
    assert first.checkpoint is not None
    assert first.checkpoint.compactions > 0
    approved = agent.executor.approve(first.approval_id)
    assert approved.success
    cp = agent.checkpoint_after_approval(
        first.checkpoint, first.approval_id, approved,
    )
    assert cp.compactions == first.checkpoint.compactions
    assert cp.compacted_chars == first.checkpoint.compacted_chars
    assert cp.seen_calls == first.checkpoint.seen_calls
    after = agent.run(user_input="read then write", resume_from=cp)
    assert after.termination_reason == "final_answer"
    assert calls == [("read_file", "a.py"), ("write_file", "x.txt")]
    assert not any(s.error for s in after.steps)


def test_budget_overflow_never_enters_tool_or_llm():
    agent, llm, calls = make_agent([read("a.py")])
    result = agent.run(user_input="G" * 6000)
    assert result.termination_reason == "context_overflow"
    assert calls == [] and llm.requests == []


def test_compact_opt_in_is_backwards_compatible():
    agent, llm, calls = make_agent([
        read("a.py"), {"action": "final", "task_complete": True},
    ], compact=False)
    result = agent.run(user_input="read")
    assert result.termination_reason == "final_answer"
    assert agent.context_compactor is None
    assert not any(s.action == "context_compaction" for s in result.steps)
    assert calls == [("read_file", "a.py")]


def test_s05_planning_trace_shows_compaction_and_all_tools():
    agent, llm, calls = make_agent([
        read("a.py"), read("b.py"),
        {"action": "final", "task_complete": True},
    ], planning=True)
    plan = agent.planner.create("inspect", ["read and inspect a.py and b.py"])
    assert agent.planner.run(plan.id).status == "completed"
    detail = agent.planner.describe(plan, detail=True)
    assert "S08 上下文压缩" in detail
    assert "read_file a.py" in detail
    assert "read_file b.py" in detail
    assert calls == [("read_file", "a.py"), ("read_file", "b.py")]


def test_s07_skill_instruction_body_is_not_privileged_in_summary():
    c = ContextCompactor(max_chars=4000)
    step = AgentStep(
        index=1, action="tool",
        tool_call=ToolCall(name="load_skill", arguments={"name": "code-review"}),
        tool_output={"name": "code-review", "content": CANARY * 200},
    )
    before = [
        {"role": "system", "content": "skill catalog: code-review"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": CANARY * 300},
        {"role": "user", "content": CANARY * 300},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    after, report = c.compact(before, user_input="goal", steps=(step,))
    note = next(row["content"] for row in after if row["content"].startswith(MARKER))
    assert "skill_body=not_retained" in note
    assert CANARY not in note
    assert step.tool_output["content"].startswith(CANARY)


def test_context_stats_command_not_a_learning_sample():
    assert normalize_cli_input("context-stats") == "/context-stats"
    assert looks_like_operational_input("context-stats")
