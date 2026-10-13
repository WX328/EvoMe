from types import SimpleNamespace

import pytest

import app.llm as llm_module
from app.llm import DeepSeekClient


def fake_client(monkeypatch, content, finish_reason="stop"):
    requests = []
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])

    def create(**kwargs):
        requests.append(kwargs)
        return response

    client = object.__new__(DeepSeekClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(client, "_record_usage", lambda **kwargs: None)
    monkeypatch.setattr(llm_module, "EVOME_MAX_TOKENS_AGENT", 4096)
    monkeypatch.setattr(llm_module, "EVOME_MAX_TOKENS_JSON_DEFAULT", 768)
    return client, requests


def test_agent_answer_budget_is_separate_from_internal_json(monkeypatch):
    client, requests = fake_client(monkeypatch, '{"action":"final","answer":"done"}')
    client.json_chat([], purpose="agent_loop")
    client.json_chat([], purpose="agent_plan")
    client.json_chat([], purpose="json_internal")
    assert [request["max_tokens"] for request in requests] == [4096, 768, 768]
    assert all(request["extra_body"]["thinking"]["type"] == "disabled" for request in requests)


@pytest.mark.parametrize("content", [
    '{"action":"tool","tool_name":"write_file","arguments":{"path":"x","content":"cut',
    '{"action":"final","answer":"partial',
    '{"action":"final","task_complete":true,"answer":"looks valid"}',
])
def test_length_limited_agent_json_is_never_a_runnable_or_completed_decision(monkeypatch, content):
    client, _ = fake_client(monkeypatch, content, finish_reason="length")
    decision = client.json_chat([], purpose="agent_loop")
    assert decision["action"] == "output_limit"
    assert "任务未确认完成" in decision["answer"]
    assert "EVOME_MAX_TOKENS_AGENT" in decision["answer"]
