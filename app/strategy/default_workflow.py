from langgraph.graph import END, START, StateGraph

from app.strategy.state import EvoMeState


SYSTEM_PROMPT = '''
你是 EvoMe，一个长期陪伴用户成长的 Personal AI。

当前版本原则：
1. 当前用户明确输入永远优先于历史 Memory。
2. Memory 是历史经验，不保证现在仍然完全正确。
3. 只在真正有帮助时使用历史信息，不要强行展示“我记得你”。
4. 当前 Strategy 是一个可执行 Workflow；Skill 是能力抽象。
5. 回答应帮助用户解决当前问题，同时保持连续性。
'''


class DefaultConversationStrategy:
    name = "default_conversation_strategy_v1"

    def __init__(
        self,
        *,
        llm,
        memory_service,
        reflection_service,
        evaluation_service,
        skill_registry,
    ) -> None:
        self.llm = llm
        self.memory_service = memory_service
        self.reflection_service = reflection_service
        self.evaluation_service = evaluation_service
        self.skill_registry = skill_registry

        self.graph = self._build_graph()

    def _retrieve_memory(self, state: EvoMeState) -> dict:
        impl = self.skill_registry.resolve(
            "recall_relevant_memory"
        )
        memories = impl.execute(
            query=state["user_input"],
            top_k=5,
        )
        return {
            "relevant_memories": [
                memory.content for memory in memories
            ]
        }

    def _respond(self, state: EvoMeState) -> dict:
        memories = state.get("relevant_memories", [])
        memory_context = (
            "\n".join(f"- {m}" for m in memories)
            if memories
            else "暂无直接相关长期经验。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "与当前问题相关的 Memory：\n"
                    f"{memory_context}"
                ),
            },
            {
                "role": "user",
                "content": state["user_input"],
            },
        ]

        return {
            "response": self.llm.chat(messages)
        }

    def _evaluate(self, state: EvoMeState) -> dict:
        evaluation = self.evaluation_service.evaluate(
            user_input=state["user_input"],
            response=state["response"],
            relevant_memories=state.get(
                "relevant_memories",
                [],
            ),
        )
        return {"evaluation": evaluation}

    def _reflect(self, state: EvoMeState) -> dict:
        reflection = self.reflection_service.reflect(
            user_input=state["user_input"],
            response=state["response"],
            evaluation=state["evaluation"],
            relevant_memories=state.get(
                "relevant_memories",
                [],
            ),
        )
        return {"reflection": reflection}

    def _write_memory(self, state: EvoMeState) -> dict:
        reflection = state.get("reflection", {})

        if not reflection.get("should_save"):
            return {"memory_saved": False}

        summary = reflection.get("summary", "").strip()
        lesson = reflection.get("lesson", "").strip()

        if not summary:
            return {"memory_saved": False}

        content = summary
        if lesson:
            content += f"\n经验：{lesson}"

        if self.memory_service.is_duplicate(content):
            return {"memory_saved": False}

        self.memory_service.save(
            memory_type="experience",
            content=content,
            importance=reflection.get("importance", 0.5),
            confidence=reflection.get("confidence", 0.8),
            source="reflection",
            memory_data={
                "strategy": self.name,
                "evaluation": state["evaluation"],
                "strategy_signal": reflection.get(
                    "strategy_signal",
                    "",
                ),
            },
        )

        return {"memory_saved": True}

    def _build_graph(self):
        builder = StateGraph(EvoMeState)

        builder.add_node(
            "retrieve_memory",
            self._retrieve_memory,
        )
        builder.add_node(
            "respond",
            self._respond,
        )
        builder.add_node(
            "evaluate",
            self._evaluate,
        )
        builder.add_node(
            "reflect",
            self._reflect,
        )
        builder.add_node(
            "write_memory",
            self._write_memory,
        )

        builder.add_edge(START, "retrieve_memory")
        builder.add_edge("retrieve_memory", "respond")
        builder.add_edge("respond", "evaluate")
        builder.add_edge("evaluate", "reflect")
        builder.add_edge("reflect", "write_memory")
        builder.add_edge("write_memory", END)

        return builder.compile()

    def invoke(self, user_input: str) -> EvoMeState:
        return self.graph.invoke(
            {
                "user_input": user_input,
                "strategy_name": self.name,
                "relevant_memories": [],
                "memory_saved": False,
            }
        )
