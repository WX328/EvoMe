from app.agent import AgentLoop
from app.tools import ToolDefinition, ToolImplementation, ToolRegistry


class ScriptedLLM:
    def __init__(self, decisions, fallback="fallback") -> None:
        self.decisions = list(decisions)
        self.fallback = fallback
        self.chat_calls = 0

    def json_chat(self, messages, **kwargs):
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
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
