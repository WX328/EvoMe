from langgraph.graph import END, START, StateGraph

from app.strategy.state import EvoMeState


SYSTEM_PROMPT = """
你是 EvoMe，一个长期陪伴用户成长的 Personal AI。

原则：
1. 当前用户明确输入永远优先于历史 Personal Memory。
2. Personal Memory 是历史认识，不保证现在仍然有效。
3. 只在真正有帮助时使用历史信息。
4. Strategy 是可执行 Workflow；Skill 是能力抽象。
5. InteractionRecord 是审计日志，不参与常规检索。
"""


class DefaultConversationStrategy:
    name = "default_conversation_strategy"
    version = 2

    def __init__(
        self,
        *,
        llm,
        audit_service,
        personal_memory_service,
        personal_memory_gate,
        experience_gate,
        evolution_memory_service,
        reflection_service,
        evaluation_service,
        skill_registry,
    ) -> None:
        self.llm = llm
        self.audit_service = audit_service
        self.personal_memory_service = personal_memory_service
        self.personal_memory_gate = personal_memory_gate
        self.experience_gate = experience_gate
        self.evolution_memory_service = evolution_memory_service
        self.reflection_service = reflection_service
        self.evaluation_service = evaluation_service
        self.skill_registry = skill_registry
        self.graph = self._build_graph()

    def _retrieve_memory(self, state: EvoMeState) -> dict:
        skill_name = "recall_relevant_memory"
        impl = self.skill_registry.resolve(skill_name)
        memories = impl.execute(query=state["user_input"], top_k=5)
        return {
            "relevant_memories": [memory.content for memory in memories],
            "skills_used": [skill_name],
        }

    def _respond(self, state: EvoMeState) -> dict:
        memories = state.get("relevant_memories", [])
        memory_context = (
            "\n".join(f"- {memory}" for memory in memories)
            if memories else "暂无直接相关的 Personal Memory。"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "与当前问题相关的 Personal Memory：\n" + memory_context,
            },
            {"role": "user", "content": state["user_input"]},
        ]
        return {"response": self.llm.chat(messages)}

    def _evaluate(self, state: EvoMeState) -> dict:
        return {
            "evaluation": self.evaluation_service.evaluate(
                user_input=state["user_input"],
                response=state["response"],
                relevant_memories=state.get("relevant_memories", []),
            )
        }

    def _record_interaction(self, state: EvoMeState) -> dict:
        record = self.audit_service.save_interaction(
            session_id=state["session_id"],
            user_input=state["user_input"],
            assistant_response=state["response"],
            strategy_name=f"{self.name}_v{self.version}",
            record_metadata={
                "evaluation": state["evaluation"],
                "skills_used": state.get("skills_used", []),
            },
        )
        return {"interaction_id": record.id}

    def _extract_personal_memory(self, state: EvoMeState) -> dict:
        result = self.personal_memory_gate.extract(state["user_input"])
        return {
            "personal_memory_candidates": result["memories"]
            if result["should_save"] else []
        }

    def _write_personal_memory(self, state: EvoMeState) -> dict:
        saved = 0
        for candidate in state.get("personal_memory_candidates", []):
            content = candidate["content"]
            if self.personal_memory_service.is_duplicate(content):
                continue
            self.personal_memory_service.save(
                memory_type=candidate["memory_type"],
                content=content,
                importance=candidate["importance"],
                confidence=candidate["confidence"],
                source_interaction_id=state["interaction_id"],
            )
            saved += 1
        return {"personal_memories_saved": saved}

    def _experience_gate(self, state: EvoMeState) -> dict:
        result = self.experience_gate.evaluate(
            user_input=state["user_input"],
            assistant_response=state["response"],
            evaluation=state["evaluation"],
            strategy_name=f"{self.name}_v{self.version}",
            skills_used=state.get("skills_used", []),
        )
        return {"experience_gate": result}

    def _write_experience(self, state: EvoMeState) -> dict:
        gate = state["experience_gate"]
        if not gate.get("should_save"):
            return {"experience_saved": False, "experience_id": None}

        evaluation = state["evaluation"]
        experience = self.evolution_memory_service.save_experience(
            interaction_id=state["interaction_id"],
            task_type=gate["task_type"],
            strategy_name=self.name,
            strategy_version=self.version,
            skills_used=state.get("skills_used", []),
            result_summary=gate["result_summary"],
            success=bool(evaluation.get("success", False)),
            score=float(evaluation.get("score", 0.0)),
            experience_metadata={"gate_reason": gate.get("reason", "")},
        )
        return {"experience_saved": True, "experience_id": experience.id}

    @staticmethod
    def _route_after_experience(state: EvoMeState) -> str:
        return "reflect_experience" if state.get("experience_saved") else "end"

    def _reflect_experience(self, state: EvoMeState) -> dict:
        gate = state["experience_gate"]
        reflection = self.reflection_service.reflect(
            task_type=gate["task_type"],
            result_summary=gate["result_summary"],
            evaluation=state["evaluation"],
            strategy_name=f"{self.name}_v{self.version}",
            skills_used=state.get("skills_used", []),
        )
        return {"reflection": reflection}

    def _write_reflection(self, state: EvoMeState) -> dict:
        reflection = state.get("reflection", {})
        observation = reflection.get("observation", "").strip()
        lesson = reflection.get("lesson", "").strip()
        if not observation and not lesson:
            return {"reflection_saved": False}

        self.evolution_memory_service.save_reflection(
            experience_id=state["experience_id"],
            reflection_type=reflection["reflection_type"],
            observation=observation,
            lesson=lesson,
            strategy_signal=reflection.get("strategy_signal", ""),
            skill_signal=reflection.get("skill_signal", ""),
            confidence=reflection.get("confidence", 0.8),
        )
        return {"reflection_saved": True}

    def _build_graph(self):
        builder = StateGraph(EvoMeState)
        builder.add_node("retrieve_personal_memory", self._retrieve_memory)
        builder.add_node("respond", self._respond)
        builder.add_node("evaluate", self._evaluate)
        builder.add_node("record_interaction", self._record_interaction)
        builder.add_node("personal_memory_gate", self._extract_personal_memory)
        builder.add_node("write_personal_memory", self._write_personal_memory)
        builder.add_node("experience_gate", self._experience_gate)
        builder.add_node("write_experience", self._write_experience)
        builder.add_node("reflect_experience", self._reflect_experience)
        builder.add_node("write_reflection", self._write_reflection)

        builder.add_edge(START, "retrieve_personal_memory")
        builder.add_edge("retrieve_personal_memory", "respond")
        builder.add_edge("respond", "evaluate")
        builder.add_edge("evaluate", "record_interaction")
        builder.add_edge("record_interaction", "personal_memory_gate")
        builder.add_edge("personal_memory_gate", "write_personal_memory")
        builder.add_edge("write_personal_memory", "experience_gate")
        builder.add_edge("experience_gate", "write_experience")
        builder.add_conditional_edges(
            "write_experience",
            self._route_after_experience,
            {"reflect_experience": "reflect_experience", "end": END},
        )
        builder.add_edge("reflect_experience", "write_reflection")
        builder.add_edge("write_reflection", END)
        return builder.compile()

    def invoke(self, *, user_input: str, session_id: str) -> EvoMeState:
        return self.graph.invoke(
            {
                "user_input": user_input,
                "strategy_name": self.name,
                "strategy_version": self.version,
                "session_id": session_id,
                "relevant_memories": [],
                "skills_used": [],
                "personal_memory_candidates": [],
                "personal_memories_saved": 0,
                "experience_saved": False,
                "experience_id": None,
                "reflection_saved": False,
            }
        )
