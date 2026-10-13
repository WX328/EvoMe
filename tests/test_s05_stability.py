"""Regression tests for S05 malformed JSON and source-evidence gating."""
from types import SimpleNamespace

import pytest

from app.agent import AgentLoop
from app.llm import DeepSeekClient
from tests.test_planning import make_agent


def fake_json_client(raw_content: str, *, finish_reason: str = "stop"):
    client = object.__new__(DeepSeekClient)
    choice = SimpleNamespace(
        message=SimpleNamespace(content=raw_content),
        finish_reason=finish_reason,
    )
    response = SimpleNamespace(choices=[choice])

    def create(**kwargs):
        return response

    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    client._record_usage = lambda **kwargs: None
    return client


def test_repaired_agent_final_is_untrusted_and_raw_content_not_logged(capsys):
    secret = "CANARY_PRIVATE_FILE_CONTENT"
    # Actual line break inside a JSON string is invalid JSON.
    raw = '{"action":"final","answer":"first line\n' + secret + '"}'
    client = fake_json_client(raw)
    result = client.json_chat([], purpose="agent_loop")
    assert result["_json_repaired"] is True
    assert result["action"] == "final"
    captured = capsys.readouterr().out
    assert "JSON" in captured
    assert secret not in captured
    assert "Raw content" not in captured


def test_normal_json_does_not_get_repaired_marker():
    client = fake_json_client('{"action":"final","answer":"ok"}')
    result = client.json_chat([], purpose="agent_loop")
    assert result == {"action": "final", "answer": "ok"}


def test_truncated_agent_plan_is_never_accepted_as_plan():
    client = fake_json_client(
        '{"steps":["first","partial"',
        finish_reason="length",
    )
    decision = client.json_chat([], purpose="agent_plan")
    assert decision["action"] == "output_limit"
    assert "steps" not in decision


def test_repaired_tool_action_is_not_executed():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "write_file",
         "arguments": {"path": "x.txt", "content": "danger"},
         "_json_repaired": True},
        {"action": "final", "answer": "没有执行写入"},
    ])
    result = agent.run(user_input="写文件")
    assert calls == []
    assert result.termination_reason == "final_answer"
    assert any(step.action == "invalid_json_retry" for step in result.steps)


def test_repaired_final_needs_strict_json_confirmation():
    agent, calls = make_agent([
        {"action": "final", "answer": "未经确认的分析", "_json_repaired": True},
        {"action": "final", "answer": "已确认格式的回答"},
    ])
    result = agent.run(user_input="简单分析")
    assert result.answer == "已确认格式的回答"
    assert result.termination_reason == "final_answer"
    assert calls == []


def test_repaired_planning_json_is_not_accepted():
    agent, _ = make_agent([
        {"steps": ["读取源码"], "_json_repaired": True}
    ])
    with pytest.raises(ValueError, match="malformed JSON"):
        agent.planner.create_from_llm("分析源码")


def test_file_inspection_without_tool_evidence_cannot_complete():
    agent, calls = make_agent([
        {"action": "final", "answer": "我已经阅读 app/agent 全部源码"}
    ])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析 app/agent 源码", ["读取 app/agent 下的代码"])
    result = agent.planner.run(plan.id)
    assert result.status == "failed"
    assert result.completed_count == 0
    assert "missing_file_evidence" in result.error
    assert calls == []


def test_file_inspection_with_real_tool_result_completes():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "read_file",
         "arguments": {"path": "app/agent/loop.py"}},
        {"action": "final", "answer": "根据实际文件结果完成分析"},
        {"action": "final", "answer": "改进点总结"},
    ])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create(
        "分析 app/agent 源码", ["读取 app/agent/loop.py", "综合改进建议"],
    )
    result = agent.planner.run(plan.id)
    assert result.status == "completed"
    assert result.completed_count == 2
    assert calls == [("read_file", {"path": "app/agent/loop.py"})]


def test_task_complete_false_does_not_mark_step_done():
    agent, _ = make_agent([
        {"action": "final", "task_complete": False, "answer": "还需更多证据"}
    ])
    plan = agent.planner.create("分析项目", ["分析某个模块"])
    result = agent.planner.run(plan.id)
    assert result.status == "failed"
    assert result.completed_count == 0
