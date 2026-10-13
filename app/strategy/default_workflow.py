from langgraph.graph import END, START, StateGraph

from app.agent import AgentLoop, PlanStore
from app.hooks.manager import HookManager
from app.hypothesis.canonical import build_canonical_signal
from app.tools import ApprovalStore, PermissionPolicy, build_default_tool_registry
from app.skills.loader import SkillLoader
from app.learning.post_turn_assessor import PostTurnAssessor
from app.models import ReflectionRecord
from app.strategy.state import EvoMeState


SYSTEM_PROMPT = """
你是 EvoMe，一个长期陪伴用户成长的 Personal AI。

原则：
1. 当前用户明确输入永远优先于历史 Personal Memory。
2. Personal Memory 是历史认识，不保证现在仍然有效。
3. 只在真正有帮助时使用历史信息。
4. Strategy 是可执行 Workflow；Skill 是能力抽象。
5. InteractionRecord 是审计日志，不参与常规检索。
6. Reflection/Hypothesis 只是进化证据，不代表已验证的 Strategy。
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
        learning_eligibility_gate,
        experience_gate,
        context_builder,
        evolution_memory_service,
        reflection_service,
        hypothesis_aggregator,
        evaluation_service,
        post_turn_assessor: PostTurnAssessor,
        skill_registry,
        enable_agent_loop: bool = False,
        agent_max_steps: int = 6,
        tool_workspace: str = ".",
        enable_bash_tool: bool = False,
        bash_timeout_seconds: int = 20,
        tool_max_output_chars: int = 12000,
        bash_permission: str = "ask",
        approval_store: ApprovalStore | None = None,
        hook_manager: HookManager | None = None,
        planning_enabled: bool = False,
        plan_store: PlanStore | None = None,
        plan_max_tasks: int = 5,
        plan_step_max_steps: int = 4,
        plan_step_max_continuations: int = 2,
        plan_max_replans: int = 1,
        plan_auto_replan: bool = False,
        plan_require_file_evidence: bool = False,
        subagent_enabled: bool = False,
        subagent_max_steps: int = 4,
        subagent_max_calls: int = 2,
        skill_loading_enabled: bool = False,
        skill_directory: str = "skills",
        skill_max_content_chars: int = 16000,
    ) -> None:
        self.llm = llm
        self.audit_service = audit_service
        self.personal_memory_service = personal_memory_service
        self.personal_memory_gate = personal_memory_gate
        self.learning_eligibility_gate = learning_eligibility_gate
        self.experience_gate = experience_gate
        self.context_builder = context_builder
        self.evolution_memory_service = evolution_memory_service
        self.reflection_service = reflection_service
        self.hypothesis_aggregator = hypothesis_aggregator
        self.evaluation_service = evaluation_service
        self.post_turn_assessor = post_turn_assessor
        self.skill_registry = skill_registry
        self.tool_registry = build_default_tool_registry(
            skill_registry=skill_registry,
            workspace_root=tool_workspace,
            enable_bash=enable_bash_tool,
            bash_timeout_seconds=bash_timeout_seconds,
            max_output_chars=tool_max_output_chars,
        )
        self.approval_store = approval_store or ApprovalStore()
        self.agent_loop = (
            AgentLoop(
                llm=llm,
                tool_registry=self.tool_registry,
                max_steps=agent_max_steps,
                permission_policy=PermissionPolicy(bash_mode=bash_permission),
                approval_store=self.approval_store,
                hook_manager=hook_manager,
                planning_enabled=planning_enabled,
                plan_store=plan_store,
                plan_max_tasks=plan_max_tasks,
                plan_step_max_steps=plan_step_max_steps,
                plan_step_max_continuations=plan_step_max_continuations,
                plan_max_replans=plan_max_replans,
                plan_auto_replan=plan_auto_replan,
                plan_require_file_evidence=plan_require_file_evidence,
                subagent_enabled=subagent_enabled,
                subagent_max_steps=subagent_max_steps,
                subagent_max_calls=subagent_max_calls,
                skill_loader=(
                    SkillLoader(
                        workspace_root=tool_workspace,
                        directory=skill_directory,
                        max_content_chars=skill_max_content_chars,
                    )
                    if skill_loading_enabled else None
                ),
            )
            if enable_agent_loop
            else None
        )
        self.graph = self._build_graph()

    def _retrieve_memory(self, state: EvoMeState) -> dict:
        skill_name = "recall_relevant_memory"
        impl = self.skill_registry.resolve(skill_name)
        memories = impl.execute(query=state["user_input"], top_k=5)
        quality = (
            sum(float(memory.confidence) for memory in memories) / len(memories)
            if memories
            else 0.0
        )
        return {
            "relevant_memories": [memory.content for memory in memories],
            "memory_quality": quality,
            "skills_used": [skill_name],
        }

    def _response_messages(self, state: EvoMeState) -> list[dict[str, str]]:
        memories = state.get("relevant_memories", [])
        memory_context = (
            "\n".join(f"- {memory}" for memory in memories)
            if memories
            else "暂无直接相关的 Personal Memory。"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "与当前问题相关的 Personal Memory：\n" + memory_context,
            },
            {"role": "user", "content": state["user_input"]},
        ]

    @staticmethod
    def _agent_step_dict(step) -> dict:
        call = step.tool_call
        return {
            "index": step.index,
            "action": step.action,
            "tool_name": call.name if call is not None else None,
            "arguments": dict(call.arguments) if call is not None else {},
            "error": step.error,
        }

    @staticmethod
    def _memory_state_from_agent(result) -> tuple[list[str], float]:
        contents: list[str] = []
        confidences: list[float] = []
        for step in result.steps:
            call = step.tool_call
            if call is None or call.name != "recall_relevant_memory":
                continue
            output = step.tool_output
            if not isinstance(output, (list, tuple)):
                continue
            for item in output:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if content:
                    contents.append(str(content))
                confidence = getattr(item, "confidence", None)
                if confidence is None and isinstance(item, dict):
                    confidence = item.get("confidence")
                if confidence is not None:
                    try:
                        confidences.append(float(confidence))
                    except (TypeError, ValueError):
                        pass
        quality = sum(confidences) / len(confidences) if confidences else 0.0
        return contents, quality

    def _run_agent(self, state: EvoMeState) -> dict:
        if self.agent_loop is None:
            return {"response": self.llm.chat(self._response_messages(state))}

        result = self.agent_loop.run(
            user_input=state["user_input"],
            context_messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        )
        memories, memory_quality = self._memory_state_from_agent(result)
        skills_used = [
            call.name
            for call in result.tool_calls
            if call.name
        ]
        return {
            "response": result.answer,
            "relevant_memories": memories,
            "memory_quality": memory_quality,
            "skills_used": skills_used,
            "agent_steps": [self._agent_step_dict(step) for step in result.steps],
            "agent_termination_reason": result.termination_reason,
            "tool_error_count": result.tool_error_count,
        }

    def _respond(self, state: EvoMeState) -> dict:
        return self._run_agent(state)

    def stream_response(self, state: EvoMeState):
        if self.agent_loop is not None:
            result = self._run_agent(state)
            state.update(result)
            if result["response"]:
                yield result["response"]
            return

        messages = self._response_messages(state)
        stream_chat = getattr(self.llm, "stream_chat", None)
        if callable(stream_chat):
            yield from stream_chat(messages)
            return
        # 测试/兼容路径：旧 LLM adapter 没有 stream_chat 时仍可工作。
        response = self.llm.chat(messages)
        if response:
            yield response

    def _classify_learning_eligibility(self, state: EvoMeState) -> dict:
        result = self.learning_eligibility_gate.classify(state["user_input"])
        return {"learning_eligibility": result}

    @staticmethod
    def _route_after_eligibility(state: EvoMeState) -> str:
        eligibility = state.get("learning_eligibility", {})
        return "post_turn_assess" if eligibility.get("allow_evolution") else "record_interaction"

    @staticmethod
    def _route_after_interaction(state: EvoMeState) -> str:
        eligibility = state.get("learning_eligibility", {})
        if eligibility.get("allow_personal_memory"):
            return "personal_memory_gate"
        if eligibility.get("allow_evolution"):
            return "experience_gate"
        return "end"

    @staticmethod
    def _route_after_personal_memory(state: EvoMeState) -> str:
        eligibility = state.get("learning_eligibility", {})
        return "experience_gate" if eligibility.get("allow_evolution") else "end"

    def _assess_post_turn(self, state: EvoMeState) -> dict:
        assessment = self.post_turn_assessor.assess(
            user_input=state["user_input"],
            assistant_response=state["response"],
            relevant_memories=state.get("relevant_memories", []),
            strategy_name=f"{self.name}_v{self.version}",
            skills_used=state.get("skills_used", []),
        )
        evaluation = self.evaluation_service.normalize_result(
            assessment.get("evaluation")
        )
        return {
            "post_turn_assessment": assessment,
            "evaluation": evaluation,
        }

    def _record_interaction(self, state: EvoMeState) -> dict:
        metadata = {
            "skills_used": state.get("skills_used", []),
            "learning_eligibility": state.get("learning_eligibility", {}),
            "evolution_deferred_by_cadence": bool(
                state.get("evolution_deferred_by_cadence", False)
            ),
            "agent": {
                "termination_reason": state.get("agent_termination_reason", ""),
                "tool_error_count": int(state.get("tool_error_count", 0)),
                "steps": state.get("agent_steps", []),
            },
        }
        evaluation = state.get("evaluation")
        if evaluation:
            metadata["evaluation"] = evaluation

        record = self.audit_service.save_interaction(
            session_id=state["session_id"],
            user_input=state["user_input"],
            assistant_response=state["response"],
            strategy_name=f"{self.name}_v{self.version}",
            record_metadata=metadata,
        )
        return {"interaction_id": record.id}

    def _extract_personal_memory(self, state: EvoMeState) -> dict:
        result = self.personal_memory_gate.extract(state["user_input"])
        return {
            "personal_memory_candidates": result["memories"]
            if result["should_save"]
            else []
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

    def _build_context(self, state: EvoMeState) -> dict:
        gate = state["experience_gate"]
        if not gate.get("should_save"):
            return {"context_data": {}}

        available_skills = [skill.name for skill in self.skill_registry.list_skills()]
        assessment = state.get("post_turn_assessment", {})
        context_data = self.context_builder.normalize_result(
            user_input=state["user_input"],
            result=assessment.get("context") if isinstance(assessment, dict) else {},
            relevant_memories=state.get("relevant_memories", []),
            available_skills=available_skills,
            available_tools=[tool.name for tool in self.tool_registry.list_tools()],
            memory_quality=float(state.get("memory_quality", 0.0)),
            task_type_hint=gate.get("task_type"),
            prefer_task_type_hint=str(gate.get("decision_source", "")).startswith(
                "deterministic"
            ),
        )
        return {"context_data": context_data}

    def _write_experience(self, state: EvoMeState) -> dict:
        gate = state["experience_gate"]
        if not gate.get("should_save"):
            return {"experience_saved": False, "experience_id": None}

        evaluation = state["evaluation"]
        context_data = state.get("context_data", {})
        task_type = str(context_data.get("task_type") or gate.get("task_type") or "other")
        experience = self.evolution_memory_service.save_experience(
            interaction_id=state["interaction_id"],
            task_type=task_type,
            strategy_name=self.name,
            strategy_version=self.version,
            skills_used=state.get("skills_used", []),
            result_summary=gate["result_summary"],
            success=bool(evaluation.get("success", False)),
            score=float(evaluation.get("score", 0.0)),
            experience_metadata={
                "gate_reason": gate.get("reason", ""),
                "gate_source": gate.get("decision_source", "unknown"),
            },
            context_data=context_data,
        )
        return {"experience_saved": True, "experience_id": experience.id}

    @staticmethod
    def _route_after_experience(state: EvoMeState) -> str:
        return "reflect_experience" if state.get("experience_saved") else "end"

    def _reflect_experience(self, state: EvoMeState) -> dict:
        assessment = state.get("post_turn_assessment", {})
        raw_reflection = (
            assessment.get("reflection", {})
            if isinstance(assessment, dict)
            else {}
        )
        reflection = self.reflection_service.normalize_result(
            raw_reflection,
            context_data=state.get("context_data", {}),
        )
        return {"reflection": reflection}

    def _write_reflection(self, state: EvoMeState) -> dict:
        reflection = state.get("reflection", {})
        observation = str(reflection.get("observation", "")).strip()
        insight = str(reflection.get("insight", reflection.get("lesson", ""))).strip()
        if not observation and not insight:
            return {"reflection_saved": False, "reflection_id": None}

        canonical_signal = ""
        strategy_signal = str(reflection.get("strategy_signal", "")).strip()
        if strategy_signal and insight:
            canonical_signal = build_canonical_signal(
                applicable_context=reflection.get("applicable_context", {}),
                proposed_change=strategy_signal,
                expected_effect=str(reflection.get("expected_effect", "")),
                insight=insight,
            )

        record = self.evolution_memory_service.save_reflection(
            experience_id=state["experience_id"],
            reflection_type=reflection["reflection_type"],
            observation=observation,
            insight=insight,
            applicable_context=reflection.get("applicable_context", {}),
            expected_effect=str(reflection.get("expected_effect", "")),
            canonical_signal=canonical_signal,
            strategy_signal=strategy_signal,
            skill_signal=reflection.get("skill_signal", ""),
            confidence=reflection.get("confidence", 0.8),
        )
        return {"reflection_saved": True, "reflection_id": record.id}

    @staticmethod
    def _route_after_reflection(state: EvoMeState) -> str:
        return "aggregate_hypothesis" if state.get("reflection_saved") else "end"

    def _aggregate_hypothesis(self, state: EvoMeState) -> dict:
        reflection_id = state.get("reflection_id")
        if reflection_id is None:
            return {
                "hypothesis_id": None,
                "hypothesis_action": "skipped",
                "revision_signal_count": 0,
            }
        reflection = self.evolution_memory_service.db.get(ReflectionRecord, reflection_id)
        if reflection is None:
            return {
                "hypothesis_id": None,
                "hypothesis_action": "skipped",
                "revision_signal_count": 0,
            }
        result = self.hypothesis_aggregator.process(reflection)
        return {
            "hypothesis_id": result.get("hypothesis_id"),
            "hypothesis_action": result.get("action", "skipped"),
            "revision_signal_count": int(result.get("revision_signal_count", 0)),
        }

    def _initial_state(self, *, user_input: str, session_id: str) -> EvoMeState:
        return {
            "user_input": user_input,
            "strategy_name": self.name,
            "strategy_version": self.version,
            "session_id": session_id,
            "relevant_memories": [],
            "memory_quality": 0.0,
            "skills_used": [],
            "agent_steps": [],
            "agent_termination_reason": "",
            "tool_error_count": 0,
            "learning_eligibility": {},
            "evaluation": {},
            "post_turn_assessment": {},
            "evolution_deferred_by_cadence": False,
            "personal_memory_candidates": [],
            "personal_memories_saved": 0,
            "context_data": {},
            "experience_saved": False,
            "experience_id": None,
            "reflection_saved": False,
            "reflection_id": None,
            "hypothesis_id": None,
            "hypothesis_action": "skipped",
            "revision_signal_count": 0,
        }

    def prepare_fast_state(self, *, user_input: str, session_id: str) -> EvoMeState:
        """Synchronous critical path before the first answer token."""
        state = self._initial_state(user_input=user_input, session_id=session_id)
        if self.agent_loop is None:
            state.update(self._retrieve_memory(state))
        return state

    def process_slow_state(
        self,
        state: EvoMeState,
        *,
        run_evolution: bool = True,
    ) -> EvoMeState:
        """Post-response learning path. Safe to execute asynchronously.

        Learning Eligibility is the first Slow-Path boundary. Every turn may be
        audited, but only eligible lanes are allowed to touch PersonalMemory or
        the Experience -> Reflection -> Hypothesis evolution chain.
        """
        state = dict(state)
        state.update(self._classify_learning_eligibility(state))
        eligibility = state["learning_eligibility"]
        evolution_enabled = bool(eligibility.get("allow_evolution")) and bool(
            run_evolution
        )
        state["evolution_deferred_by_cadence"] = bool(
            eligibility.get("allow_evolution") and not run_evolution
        )

        if evolution_enabled:
            state.update(self._assess_post_turn(state))

        # InteractionRecord is append-only audit and is intentionally written
        # even when both learning lanes are closed.
        state.update(self._record_interaction(state))

        if eligibility.get("allow_personal_memory"):
            state.update(self._extract_personal_memory(state))
            state.update(self._write_personal_memory(state))

        if not evolution_enabled:
            return state

        state.update(self._experience_gate(state))
        state.update(self._build_context(state))
        state.update(self._write_experience(state))
        if not state.get("experience_saved"):
            return state

        state.update(self._reflect_experience(state))
        state.update(self._write_reflection(state))
        if not state.get("reflection_saved"):
            return state

        state.update(self._aggregate_hypothesis(state))
        return state

    def _build_graph(self):
        builder = StateGraph(EvoMeState)
        builder.add_node("retrieve_personal_memory", self._retrieve_memory)
        builder.add_node("respond", self._respond)
        builder.add_node("learning_eligibility_gate", self._classify_learning_eligibility)
        builder.add_node("post_turn_assess", self._assess_post_turn)
        builder.add_node("record_interaction", self._record_interaction)
        builder.add_node("personal_memory_gate", self._extract_personal_memory)
        builder.add_node("write_personal_memory", self._write_personal_memory)
        builder.add_node("experience_gate", self._experience_gate)
        builder.add_node("build_context", self._build_context)
        builder.add_node("write_experience", self._write_experience)
        builder.add_node("reflect_experience", self._reflect_experience)
        builder.add_node("write_reflection", self._write_reflection)
        builder.add_node("aggregate_hypothesis", self._aggregate_hypothesis)

        builder.add_edge(START, "retrieve_personal_memory")
        builder.add_edge("retrieve_personal_memory", "respond")
        builder.add_edge("respond", "learning_eligibility_gate")
        builder.add_conditional_edges(
            "learning_eligibility_gate",
            self._route_after_eligibility,
            {
                "post_turn_assess": "post_turn_assess",
                "record_interaction": "record_interaction",
            },
        )
        builder.add_edge("post_turn_assess", "record_interaction")
        builder.add_conditional_edges(
            "record_interaction",
            self._route_after_interaction,
            {
                "personal_memory_gate": "personal_memory_gate",
                "experience_gate": "experience_gate",
                "end": END,
            },
        )
        builder.add_edge("personal_memory_gate", "write_personal_memory")
        builder.add_conditional_edges(
            "write_personal_memory",
            self._route_after_personal_memory,
            {"experience_gate": "experience_gate", "end": END},
        )
        builder.add_edge("experience_gate", "build_context")
        builder.add_edge("build_context", "write_experience")
        builder.add_conditional_edges(
            "write_experience",
            self._route_after_experience,
            {"reflect_experience": "reflect_experience", "end": END},
        )
        builder.add_edge("reflect_experience", "write_reflection")
        builder.add_conditional_edges(
            "write_reflection",
            self._route_after_reflection,
            {"aggregate_hypothesis": "aggregate_hypothesis", "end": END},
        )
        builder.add_edge("aggregate_hypothesis", END)
        return builder.compile()

    def invoke(self, *, user_input: str, session_id: str) -> EvoMeState:
        # 保留同步入口用于测试和非交互调用；CLI 默认走 Fast/Slow 分离。
        state = self.prepare_fast_state(user_input=user_input, session_id=session_id)
        state.update(self._respond(state))
        return self.process_slow_state(state)
