from app.agent import AgentLoop
from app.skills import builtin as builtin_skills
from app.strategy.default_workflow import DefaultConversationStrategy
from app.tools import build_tool_registry_from_skills


class DecisionLLM:
    def __init__(self):
        self.calls = 0

    def json_chat(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "action": "tool",
                "tool_name": "recall_relevant_memory",
                "arguments": {"query": "EvoMe", "top_k": 3},
            }
        return {"action": "final", "answer": "已根据记忆回答。"}

    def chat(self, messages, **kwargs):
        return "fallback"


class Memory:
    def __init__(self, content, confidence=0.9):
        self.content = content
        self.confidence = confidence
        self.memory_type = "project"
        self.importance = 0.8


class MemoryService:
    def search(self, query, top_k=5):
        return [Memory("EvoMe 是长期个人 AI 项目。", 0.9)]


def test_skill_adapter_exposes_memory_as_tool():
    skills = builtin_skills.build_default_skill_registry(MemoryService())
    tools = build_tool_registry_from_skills(skills)

    assert tools.get_tool("recall_relevant_memory").name == "recall_relevant_memory"
    result = tools.resolve("recall_relevant_memory").execute(query="EvoMe", top_k=3)
    assert result[0].content.startswith("EvoMe")


def test_agent_loop_can_use_existing_memory_skill():
    skills = builtin_skills.build_default_skill_registry(MemoryService())
    tools = build_tool_registry_from_skills(skills)
    result = AgentLoop(
        llm=DecisionLLM(),
        tool_registry=tools,
        max_steps=4,
    ).run(user_input="你还记得 EvoMe 吗？")

    assert result.answer == "已根据记忆回答。"
    assert result.termination_reason == "final_answer"
    assert result.tool_calls[0].name == "recall_relevant_memory"
    assert result.tool_error_count == 0
