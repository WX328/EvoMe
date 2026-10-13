"""S08 stability regression: read-page ledger, headroom, S05 continuation.

Pytest must be executed by the project owner, never by this patch builder.
"""
from __future__ import annotations

import json
import re

import pytest

from app.agent import AgentLoop
from app.agent.compaction import ContextBudgetError, ContextCompactor
from app.agent.schema import AgentStep, ToolCall
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.registry import ToolRegistry
from app.tools.permissions import PermissionPolicy


def page(start, end, total=801, *, truncated=True, line_truncated=False):
    return AgentStep(
        index=start,
        action="tool",
        tool_call=ToolCall("read_file", {"path": "app/agent/planning.py", "start_line": start}),
        tool_output={
            "path": "app/agent/planning.py",
            "start_line": start, "end_line": end, "total_lines": total,
            "truncated": truncated, "line_truncated": line_truncated,
            "next_start_line": end + 1 if truncated and not line_truncated else None,
            "content": "SOURCE_TEXT_IN_AUDIT_ONLY",
        },
    )


def test_read_ledger_reports_missing_page_after_compaction():
    note = AgentLoop._read_progress_note([
        page(1, 330), page(331, 685),
    ])
    assert "total_lines=801" in note
    assert "next_unread_line=686" in note
    assert "complete=false" in note
    assert "SOURCE_TEXT_IN_AUDIT_ONLY" not in note


def test_read_ledger_handles_noncontiguous_ranges():
    initial = [page(1, 150), page(201, 300)]
    assert "next_unread_line=151" in AgentLoop._read_progress_note(initial)
    filled = initial + [page(151, 200), page(301, 801, truncated=False)]
    note = AgentLoop._read_progress_note(filled)
    assert "next_unread_line=none" in note and "complete=true" in note


def test_read_ledger_does_not_claim_truncated_final_page_complete():
    bad = AgentLoop._read_progress_note([page(1, 801, truncated=True)])
    assert "complete=false" in bad
    bad_line = AgentLoop._read_progress_note([
        page(1, 801, truncated=True, line_truncated=True),
    ])
    assert "complete=false" in bad_line


def test_85_percent_target_avoids_repeated_micro_compaction():
    compactor = ContextCompactor(max_chars=16000, keep_recent_messages=2)
    source = [
        {"role": "system", "content": "Instruction " * 50},
        {"role": "user", "content": "audit"},
        {"role": "assistant", "content": "old" * 1800},
        {"role": "user", "content": "old" * 1800},
        {"role": "assistant", "content": "recent " * 1000},
        {"role": "user", "content": "current " * 1000},
    ]
    compacted, report = compactor.compact(source, user_input="audit")
    assert report is not None
    assert report.after_chars <= int(16000 * .85)
    assert report.saved_chars > 0
    followup = compacted + [{"role": "user", "content": "very small followup"}]
    _, second = compactor.compact(followup, user_input="audit")
    assert second is None
    assert compactor.stats()["compaction_count"] == 1


def test_compaction_rejects_missing_original_task():
    compactor = ContextCompactor(max_chars=4000)
    with pytest.raises(ContextBudgetError, match="original user task"):
        compactor.compact([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "not the original"},
            {"role": "user", "content": "X" * 5000},
        ], user_input="original")


class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.request_snapshots = []

    def json_chat(self, messages, **kwargs):
        self.request_snapshots.append([dict(m) for m in messages])
        if not self.decisions:
            raise AssertionError("unexpected decision")
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        return "模型自然语言最终总结"


def _read_call(path, start=1):
    return {"action": "tool", "tool_name": "read_file",
            "arguments": {"path": path, "start_line": start}}


def test_duplicate_read_remains_blocked_and_ledger_is_visible():
    calls = []
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition("read_file", "read source"))
    registry.register_implementation(ToolImplementation(
        "read", "read_file",
        handler=lambda path, start_line=1: (
            calls.append((path, start_line))
            or {"path": path, "start_line": start_line, "end_line": 20,
                "total_lines": 801, "truncated": True,
                "next_start_line": 21, "content": "C" * 6000}
        ),
    ))
    llm = ScriptedLLM([
        _read_call("planning.py"), _read_call("planning.py"),
        {"action": "final", "task_complete": True},
    ])
    agent = AgentLoop(
        llm=llm, tool_registry=registry,
        context_compact_enabled=True, context_max_chars=4000,
        context_keep_recent_messages=2,
    )
    result = agent.run(user_input="读取 planning.py")
    assert calls == [("planning.py", 1)]
    assert any(s.error == "duplicate tool call blocked" for s in result.steps)
    assert len(llm.request_snapshots) == 3
    second_decision_context = str(llm.request_snapshots[1])
    assert "[AGENT_READ_PROGRESS]" in second_decision_context
    assert "next_unread_line=21" in second_decision_context


class PaginationLLM:
    """Follow real read-progress metadata rather than reproducing pages."""

    def __init__(self):
        self.starts = []
        self.call_count = 0

    def json_chat(self, messages, **kwargs):
        self.call_count += 1
        if "工具调用预算已用完" in messages[0]["content"]:
            return {"action": "final", "task_complete": False}
        progress = next((
            m["content"] for m in reversed(messages)
            if m["content"].startswith("[AGENT_READ_PROGRESS]")
        ), "")
        next_match = re.search(r"next_unread_line=(none|\d+)", progress)
        if next_match and next_match.group(1) == "none":
            return {"action": "final", "task_complete": True}
        start = int(next_match.group(1)) if next_match else 1
        self.starts.append(start)
        return _read_call("long.py", start)

    def chat(self, messages, **kwargs):
        return "长文件分页读取结果"


def test_long_file_paginates_after_context_compaction_with_s05(tmp_path):
    text = "".join(
        f"def function_{i:03d}(): return {i}  # source line number {i}\n"
        for i in range(140)
    )
    (tmp_path / "long.py").write_text(text, encoding="utf-8")
    model = PaginationLLM()
    tools = build_default_tool_registry(
        skill_registry=SkillRegistry(),
        workspace_root=tmp_path,
        max_output_chars=1000,
    )
    agent = AgentLoop(
        llm=model, tool_registry=tools,
        permission_policy=PermissionPolicy(),
        planning_enabled=True, plan_step_max_steps=4,
        plan_step_max_continuations=2,
        plan_require_file_evidence=True,
        context_compact_enabled=True,
        context_max_chars=4000,
        context_keep_recent_messages=2,
    )
    plan = agent.planner.create("读取 long.py 全部内容", ["读取 long.py 全部 140 行"])
    completed = agent.planner.run(plan.id)
    assert completed.status == "completed", agent.planner.describe(plan, detail=True)
    assert completed.steps[0].status == "completed"
    assert model.starts[0] == 1
    assert model.starts == sorted(set(model.starts))
    assert len(model.starts) >= 4
    assert agent.context_compactor.compaction_count >= 1
    assert "S08 上下文压缩" in agent.planner.describe(plan, detail=True)
    assert "duplicate tool call blocked" not in agent.planner.describe(plan, detail=True)


def test_s05_failed_tool_displays_actual_diagnostic_in_plan_show(tmp_path):
    llm = ScriptedLLM([
        _read_call("a.py"), _read_call("a.py"),
        {"action": "final", "task_complete": True},
    ])
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition("read_file", "read"))
    registry.register_implementation(ToolImplementation(
        "read", "read_file", handler=lambda path, start_line=1: {
            "path": path, "start_line": 1, "end_line": 1,
            "total_lines": 1, "truncated": False,
        },
    ))
    agent = AgentLoop(
        llm=llm, tool_registry=registry,
        planning_enabled=True, plan_step_max_steps=4,
        context_compact_enabled=True, context_max_chars=4000,
    )
    plan = agent.planner.create("读代码", ["读取 a.py"])
    assert agent.planner.run(plan.id).status == "failed"
    detail = agent.planner.describe(plan, detail=True)
    assert "[AGENT_RUNTIME] 工具错误：" in detail
    assert "duplicate tool call blocked" in detail


def test_compaction_preserves_full_pagination_note_as_unprivileged_message():
    compactor = ContextCompactor(max_chars=4000, keep_recent_messages=2)
    progress = (
        "[AGENT_READ_PROGRESS] Trusted runtime read state.\n"
        'file="planning.py" total_lines=801 ranges=1-685 '
        "next_unread_line=686 complete=false"
    )
    before = [
        {"role": "system", "content": "SYSTEM_RULES " * 160},
        {"role": "user", "content": "analyze planning.py"},
        {"role": "assistant", "content": "old" * 3000},
        {"role": "user", "content": "recent tool content " * 450},
        {"role": "user", "content": progress},
    ]
    after, report = compactor.compact(
        before, user_input="analyze planning.py",
    )
    assert report is not None
    assert sum(len(msg["content"]) for msg in after) <= 4000
    ledger = [msg for msg in after if msg["content"].startswith("[AGENT_READ_PROGRESS]")]
    assert len(ledger) == 1
    assert ledger[0]["role"] == "user"
    assert ledger[0]["content"] == progress
