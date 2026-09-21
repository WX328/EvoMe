from app.runtime import EvoMeRuntime


class FakeLLM:
    def chat(self, messages, temperature=0.4):
        return "这是一个测试回复。"

    def json_chat(self, messages, temperature=0.1):
        system = messages[0]["content"]
        user_text = messages[-1]["content"]

        if "Evaluator" in system:
            return {"score": 0.9, "success": True, "reason": "测试通过"}

        if "Personal Memory Gate" in system:
            if "正在开发 EvoMe" in user_text:
                return {
                    "should_save": True,
                    "memories": [{
                        "memory_type": "project",
                        "content": "用户正在开发 EvoMe，目标是构建长期自进化 Personal AI。",
                        "importance": 0.95,
                        "confidence": 1.0,
                    }],
                }
            return {"should_save": False, "memories": []}

        if system.strip().startswith("你是 EvoMe 的 Experience Gate"):
            if "复杂任务" in user_text or "假设样例" in user_text:
                return {
                    "should_save": True,
                    "task_type": "architecture_design",
                    "result_summary": "EvoMe 完成了一次有学习价值的复杂设计任务。",
                    "reason": "值得形成执行经验",
                }
            return {
                "should_save": False,
                "task_type": "general",
                "result_summary": "",
                "reason": "普通交互，无需形成进化经验",
            }

        if "MODULE: CONTEXT_BUILDER" in system:
            stage = "memory_design"
            if "样例B" in user_text:
                stage = "skill_design"
            elif "样例C" in user_text:
                stage = "strategy_design"
            return {
                "task_type": "architecture_design",
                "task_type_confidence": 0.95,
                "interaction_mode": "collaborative",
                "interaction_mode_confidence": 0.95,
                "explicit_instruction": "先讨论设计",
                "project_name": "EvoMe",
                "project_name_confidence": 0.95,
                "project_stage": stage,
                "project_stage_confidence": 0.9,
                "complexity_indicators": {
                    "multi_step": True,
                    "ambiguity": True,
                    "cross_module": True,
                    "external_tool": False,
                    "long_term_state": True,
                    "tradeoff": True,
                },
                "risk_indicators": {
                    "destructive_change": False,
                    "external_side_effect": False,
                    "security_sensitive": False,
                    "irreversible": False,
                },
            }

        if "MODULE: REFLECTION" in system:
            return {
                "reflection_type": "pattern",
                "observation": "复杂设计任务中，先完成设计对齐后结果较好。",
                "insight": "复杂开放式工程任务在执行前进行设计对齐可能减少返工。",
                "applicable_context": {
                    "task_type": "architecture_design",
                    "complexity": "high",
                    "interaction_mode": "collaborative",
                },
                "strategy_signal": "在 execute 前增加 design_alignment",
                "skill_signal": "",
                "expected_effect": "减少用户纠正和返工",
                "confidence": 0.9,
            }

        if "MODULE: HYPOTHESIS_RELATION_JUDGE" in system:
            return {
                "relation": "SUPPORTS",
                "confidence": 0.95,
                "reason": "Context、行为改变与预期结果一致",
            }

        return {}


def test_personal_memory_and_audit_are_separated(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM(), session_id="test-session")

    first = runtime.run("我正在开发 EvoMe。")
    assert first["personal_memories_saved"] == 1
    assert len(runtime.list_personal_memories()) == 1
    assert len(runtime.list_interactions()) == 1

    second = runtime.run("EvoMe 是什么项目？")
    assert second["personal_memories_saved"] == 0
    assert len(runtime.list_personal_memories()) == 1
    assert len(runtime.list_interactions()) == 2
    assert len(runtime.list_experiences()) == 0


def test_experience_triggers_context_reflection_and_hypothesis(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM(), session_id="test-session")
    result = runtime.run("这是一个复杂任务，请测试完整进化经验链。")

    assert result["experience_saved"] is True
    assert result["reflection_saved"] is True
    assert result["hypothesis_action"] == "created"

    experiences = runtime.list_experiences()
    reflections = runtime.list_reflections()
    hypotheses = runtime.list_hypotheses()

    assert len(experiences) == 1
    assert len(reflections) == 1
    assert len(hypotheses) == 1
    assert experiences[0].context_data["task_type"] == "architecture_design"
    assert reflections[0].experience_id == experiences[0].id
    assert reflections[0].insight
    assert hypotheses[0].supporting_count == 1


def test_similar_reflections_aggregate_into_one_hypothesis(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM(), session_id="test-session")

    runtime.run("复杂任务 假设样例A，我们先讨论设计。")
    runtime.run("复杂任务 假设样例B，我们先讨论设计。")
    runtime.run("复杂任务 假设样例C，我们先讨论设计。")

    hypotheses = runtime.list_hypotheses()
    assert len(hypotheses) == 1

    hypothesis = hypotheses[0]
    assert hypothesis.supporting_count == 3
    assert hypothesis.refinement_count == 0
    assert hypothesis.independent_evidence_count == 3
    assert hypothesis.independent_support_count == 3
    assert hypothesis.context_diversity >= 2
    assert hypothesis.confidence >= 0.8
    assert hypothesis.status == "ready_for_review"


def test_refines_does_not_promote_current_hypothesis(db):
    from app.evolution.service import EvolutionMemoryService
    from app.hypothesis.service import HypothesisService

    evolution = EvolutionMemoryService(db)
    service = HypothesisService(db)

    def make_reflection(*, interaction_id: int, complexity: str, relation_seed: str):
        experience = evolution.save_experience(
            interaction_id=interaction_id,
            task_type="architecture_design",
            strategy_name="default_conversation_strategy",
            strategy_version=2,
            skills_used=["recall_relevant_memory"],
            result_summary=relation_seed,
            success=True,
            score=0.95,
            context_data={
                "task_type": "architecture_design",
                "complexity": complexity,
                "interaction_mode": "collaborative",
                "project_stage": "design",
            },
        )
        return evolution.save_reflection(
            experience_id=experience.id,
            reflection_type="pattern",
            observation=relation_seed,
            insight="复杂架构设计需要结构化职责边界分析",
            applicable_context={
                "task_type": "architecture_design",
                "complexity": complexity,
                "interaction_mode": "collaborative",
                "project_stage": "design",
            },
            expected_effect="减少返工",
            canonical_signal=relation_seed,
            strategy_signal="执行前分析职责边界和数据流",
            confidence=0.9,
        )

    first = make_reflection(
        interaction_id=101, complexity="high", relation_seed="support-1"
    )
    hypothesis = service.create_from_reflection(
        reflection=first, canonical_signal=first.canonical_signal
    )

    second = make_reflection(
        interaction_id=102, complexity="high", relation_seed="support-2"
    )
    service.add_evidence(
        hypothesis=hypothesis,
        reflection=second,
        relation_type="SUPPORTS",
        relation_confidence=0.95,
        relation_reason="直接支持当前表述",
    )

    refinement = make_reflection(
        interaction_id=103,
        complexity="medium",
        relation_seed="refine-medium-context",
    )
    service.add_evidence(
        hypothesis=hypothesis,
        reflection=refinement,
        relation_type="REFINES",
        relation_confidence=0.95,
        relation_reason="说明规律可能从 high 扩展到 medium",
    )

    updated = service.get(hypothesis.id)
    assert updated is not None
    assert updated.supporting_count == 2
    assert updated.refinement_count == 1
    assert updated.independent_evidence_count == 3
    assert updated.independent_support_count == 2
    assert updated.status == "needs_revision"
    assert updated.status != "ready_for_review"


def test_structural_module_boundary_task_is_not_missed_by_experience_gate(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM(), session_id="test-session")

    result = runtime.run(
        "请和我一起检查 PersonalMemory 和 Experience 两个模块之间的数据边界，"
        "分析各自职责、数据流和最容易混淆的职责边界。"
    )

    assert result["experience_saved"] is True
    assert result["experience_gate_source"] == "deterministic_hard_pass"
    assert result["context_data"]["task_type"] == "architecture_design"
