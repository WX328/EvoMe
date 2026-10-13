from app.agent import AgentLoop
from app.tools import ToolDefinition, ToolImplementation, ToolRegistry


class ScriptedLLM:
    def __init__(self, decisions, fallback="fallback") -> None:
        self.decisions = list(decisions)
        self.fallback = fallback
        self.chat_calls = 0
        self.requests = []
        self.fallback_messages = []

    def json_chat(self, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.fallback_messages = messages
        return self.fallback


def build_registry(handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="memory_search",
            description="search personal memory",
            input_schema={"query": "string"},
        )
    )
    registry.register_implementation(
        ToolImplementation(
            name="memory_search_v1",
            tool_name="memory_search",
            handler=handler,
        )
    )
    return registry


def test_agent_loop_executes_tool_then_finishes():
    llm = ScriptedLLM(
        [
            {
                "action": "tool",
                "tool_name": "memory_search",
                "arguments": {"query": "EvoMe"},
            },
            {"action": "final", "answer": "找到了相关记忆。"},
        ]
    )
    registry = build_registry(lambda query: [f"memory:{query}"])

    result = AgentLoop(llm=llm, tool_registry=registry, max_steps=4).run(
        user_input="你记得 EvoMe 吗？"
    )

    assert result.answer == "找到了相关记忆。"
    assert result.termination_reason == "final_answer"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "memory_search"
    assert result.steps[0].tool_output == ["memory:EvoMe"]


def test_agent_loop_blocks_duplicate_call_and_recovers():
    llm = ScriptedLLM(
        [
            {
                "action": "tool",
                "tool_name": "memory_search",
                "arguments": {"query": "same"},
            },
            {
                "action": "tool",
                "tool_name": "memory_search",
                "arguments": {"query": "same"},
            },
            {"action": "final", "answer": "done"},
        ]
    )
    registry = build_registry(lambda query: [query])

    result = AgentLoop(llm=llm, tool_registry=registry, max_steps=4).run(
        user_input="test"
    )

    assert result.answer == "done"
    assert result.steps[1].error == "duplicate tool call blocked"


def test_agent_loop_stops_at_max_steps():
    llm = ScriptedLLM(
        [
            {"action": "bad"},
            {"action": "bad"},
        ],
        fallback="safe fallback",
    )
    registry = build_registry(lambda query: [query])

    result = AgentLoop(llm=llm, tool_registry=registry, max_steps=2).run(
        user_input="test"
    )

    assert result.answer == "safe fallback"
    assert result.termination_reason == "max_steps"
    assert llm.chat_calls == 1


def test_tool_executor_surfaces_errors_as_data():
    llm = ScriptedLLM(
        [
            {
                "action": "tool",
                "tool_name": "memory_search",
                "arguments": {"query": "x"},
            },
            {"action": "final", "answer": "handled"},
        ]
    )

    def fail(query):
        raise RuntimeError("boom")

    result = AgentLoop(
        llm=llm,
        tool_registry=build_registry(fail),
        max_steps=3,
    ).run(user_input="test")

    assert result.answer == "handled"
    assert result.tool_error_count == 1
    assert "RuntimeError: boom" in result.steps[0].error


def test_last_tool_has_a_separate_final_answer_turn():
    llm = ScriptedLLM([
        {"action": "tool", "tool_name": "memory_search", "arguments": {"query": str(i)}}
        for i in range(4)
    ] + [{"action": "final", "task_complete": True, "answer": "完整分析"}])
    result = AgentLoop(
        llm=llm, tool_registry=build_registry(lambda query: "evidence:" + query), max_steps=4,
    ).run(user_input="analyze", context_messages=[{"role": "system", "content": "user context"}])
    assert result.termination_reason == "final_answer"
    assert result.answer == "完整分析"
    assert len(result.tool_calls) == 4
    assert llm.chat_calls == 0
    transcript = str(llm.requests[-1])
    assert all("evidence:" + str(i) in transcript for i in range(4))
    assert "user context" in transcript
    assert "可用 Tools" not in transcript


def test_final_only_turn_does_not_claim_incomplete_work_is_done():
    for complete in (False, None, "true"):
        llm = ScriptedLLM([
            {"action": "tool", "tool_name": "memory_search", "arguments": {"query": "x"}},
            {"action": "final", "task_complete": complete, "answer": "还需要更多证据"},
        ])
        result = AgentLoop(llm=llm, tool_registry=build_registry(lambda query: query), max_steps=1).run(user_input="test")
        assert result.termination_reason == "max_steps"
        assert result.answer == "还需要更多证据"


def test_final_only_turn_cannot_execute_an_extra_tool():
    calls = []
    llm = ScriptedLLM([
        {"action": "tool", "tool_name": "memory_search", "arguments": {"query": "first"}},
        {"action": "tool", "tool_name": "memory_search", "arguments": {"query": "extra"}},
    ])
    result = AgentLoop(
        llm=llm, tool_registry=build_registry(lambda query: calls.append(query) or "source-content"), max_steps=1,
    ).run(user_input="test")
    assert calls == ["first"]
    assert result.termination_reason == "max_steps"
    assert "source-content" in str(llm.fallback_messages)
    assert "可用 Tools" not in str(llm.fallback_messages)


def test_error_fallback_preserves_successful_tool_results():
    llm = ScriptedLLM([
        {"action": "tool", "tool_name": "memory_search", "arguments": {"query": "first"}},
        {"action": "bad"},
    ])
    result = AgentLoop(
        llm=llm, tool_registry=build_registry(lambda query: "already-read-source"), max_steps=2,
    ).run(user_input="test")
    assert result.termination_reason == "max_steps"
    assert result.tool_error_count == 1
    assert "already-read-source" in str(llm.fallback_messages)


def test_truncated_output_does_not_execute_tools_or_report_completion():
    calls = []
    llm = ScriptedLLM([{"action": "output_limit", "answer": "partial answer"}])
    result = AgentLoop(
        llm=llm, tool_registry=build_registry(lambda query: calls.append(query)),
    ).run(user_input="test")
    assert result.termination_reason == "output_limit"
    assert calls == []


def test_truncated_final_only_answer_is_not_completion():
    llm = ScriptedLLM([
        {"action": "tool", "tool_name": "memory_search", "arguments": {"query": "x"}},
        {"action": "output_limit", "answer": "partial"},
    ])
    result = AgentLoop(
        llm=llm, tool_registry=build_registry(lambda query: query), max_steps=1,
    ).run(user_input="test")
    assert result.termination_reason == "output_limit"
    assert len(result.tool_calls) == 1
