from __future__ import annotations

from app.evolution.service import EvolutionMemoryService
from app.hypothesis.aggregator import HypothesisAggregator
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.retriever import HypothesisMatch
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService


class StaticRetriever:
    def __init__(self, hypothesis_id: int) -> None:
        self.hypothesis_id = hypothesis_id
        self.indexed = []

    def index(self, hypothesis) -> None:
        self.indexed.append((hypothesis.id, hypothesis.current_revision))

    def search(self, query: str, *, top_k: int = 5, score_threshold: float = 0.7):
        return [HypothesisMatch(self.hypothesis_id, 0.99)]


class RelationLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json_chat(self, messages, temperature=0.0):
        return self.payload


def _make_reflection(db, *, interaction_id: int, complexity: str, seed: str):
    evolution = EvolutionMemoryService(db)
    experience = evolution.save_experience(
        interaction_id=interaction_id,
        task_type="architecture_design",
        strategy_name="default_conversation_strategy",
        strategy_version=2,
        skills_used=["recall_relevant_memory"],
        result_summary=seed,
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
        observation=seed,
        insight="复杂架构设计需要结构化职责边界分析",
        applicable_context={
            "task_type": "architecture_design",
            "complexity": complexity,
            "interaction_mode": "collaborative",
            "project_stage": "design",
        },
        expected_effect="减少返工",
        canonical_signal=seed,
        strategy_signal="执行前分析职责边界和数据流",
        confidence=0.9,
    )


def _create_hypothesis(db):
    service = HypothesisService(db)
    first = _make_reflection(
        db,
        interaction_id=301,
        complexity="high",
        seed="support-r1",
    )
    hypothesis = service.create_from_reflection(
        reflection=first,
        canonical_signal=first.canonical_signal,
    )
    return service, hypothesis


def test_supports_does_not_create_revision_signal(db):
    service, hypothesis = _create_hypothesis(db)
    reflection = _make_reflection(
        db,
        interaction_id=302,
        complexity="high",
        seed="support-r1-second",
    )
    judge = HypothesisRelationJudge(
        RelationLLM(
            {
                "relation": "SUPPORTS",
                "confidence": 0.95,
                "reason": "当前表述已经覆盖新证据",
                "revision_signals": [
                    {
                        "dimension": "CONTEXT",
                        "operation": "EXPAND",
                        "target_field": "complexity",
                        "before_data": ["high"],
                        "observed_data": ["medium"],
                        "canonical_change": "should be ignored",
                        "confidence": 0.9,
                    }
                ],
            }
        )
    )
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=StaticRetriever(hypothesis.id),
        score_threshold=0.0,
    )

    result = aggregator.process(reflection)

    assert result["action"] == "supports"
    assert result["revision_signal_count"] == 0
    assert RevisionSignalService(db).list_all() == []
    assert service.get(hypothesis.id).refinement_count == 0


def test_refines_creates_revision_signal_bound_to_current_revision(db):
    service, hypothesis = _create_hypothesis(db)
    reflection = _make_reflection(
        db,
        interaction_id=303,
        complexity="medium",
        seed="refine-medium",
    )
    judge = HypothesisRelationJudge(
        RelationLLM(
            {
                "relation": "REFINES",
                "confidence": 0.94,
                "reason": "medium complexity 也出现成功证据",
                "revision_signals": [
                    {
                        "dimension": "CONTEXT",
                        "operation": "EXPAND",
                        "target_field": "complexity",
                        "before_data": ["high"],
                        "observed_data": ["medium"],
                        "canonical_change": "complexity: high -> include observed medium",
                        "confidence": 0.91,
                    }
                ],
            }
        )
    )
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=StaticRetriever(hypothesis.id),
        score_threshold=0.0,
    )

    result = aggregator.process(reflection)
    signals = RevisionSignalService(db).list_all()

    assert result["action"] == "refines"
    assert result["revision_signal_count"] == 1
    assert len(signals) == 1
    signal = signals[0]
    assert signal.hypothesis_id == hypothesis.id
    assert signal.base_revision == 1
    assert signal.reflection_id == reflection.id
    assert signal.interaction_id == 303
    assert signal.dimension == "CONTEXT"
    assert signal.operation == "EXPAND"
    assert signal.target_field == "complexity"
    assert signal.before_data == ["high"]
    assert signal.observed_data == ["medium"]
    assert service.get(hypothesis.id).status == "needs_revision"


def test_refines_can_record_multiple_local_signals_without_mutating_hypothesis(db):
    service, hypothesis = _create_hypothesis(db)
    original_signal = hypothesis.canonical_signal
    original_revision = hypothesis.current_revision
    reflection = _make_reflection(
        db,
        interaction_id=304,
        complexity="medium",
        seed="refine-context-and-abstraction",
    )
    judge = HypothesisRelationJudge(
        RelationLLM(
            {
                "relation": "REFINES",
                "confidence": 0.93,
                "reason": "适用边界和抽象层级都需要继续观察",
                "revision_signals": [
                    {
                        "dimension": "CONTEXT",
                        "operation": "EXPAND",
                        "target_field": "complexity",
                        "before_data": ["high"],
                        "observed_data": ["medium"],
                        "canonical_change": "complexity: high -> include observed medium",
                        "confidence": 0.9,
                    },
                    {
                        "dimension": "ABSTRACTION",
                        "operation": "GENERALIZE",
                        "target_field": "action_scope",
                        "before_data": "storage/component responsibility analysis",
                        "observed_data": "cross-module responsibility boundary analysis",
                        "canonical_change": "action scope: storage components -> cross-module boundaries",
                        "confidence": 0.87,
                    },
                ],
            }
        )
    )
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=StaticRetriever(hypothesis.id),
        score_threshold=0.0,
    )

    result = aggregator.process(reflection)
    signals = RevisionSignalService(db).list_all(
        hypothesis_id=hypothesis.id,
        base_revision=1,
    )
    updated = service.get(hypothesis.id)

    assert result["revision_signal_count"] == 2
    assert {(item.dimension, item.operation) for item in signals} == {
        ("CONTEXT", "EXPAND"),
        ("ABSTRACTION", "GENERALIZE"),
    }
    # Phase 2 只记录信号，绝不创建 r2 或改写当前 H#1。
    assert updated.current_revision == original_revision == 1
    assert updated.canonical_signal == original_signal
    assert service.get_revision(hypothesis.id, 2) is None


def test_relation_judge_filters_invalid_revision_signal():
    judge = HypothesisRelationJudge(
        RelationLLM(
            {
                "relation": "REFINES",
                "confidence": 0.9,
                "reason": "test",
                "revision_signals": [
                    {
                        "dimension": "MAGIC",
                        "operation": "EXPAND",
                        "target_field": "complexity",
                        "canonical_change": "invalid dimension",
                    },
                    {
                        "dimension": "CONTEXT",
                        "operation": "EXPAND",
                        "target_field": "complexity",
                        "before_data": ["high"],
                        "observed_data": ["medium"],
                        "canonical_change": "valid",
                        "confidence": 1.2,
                    },
                ],
            }
        )
    )

    result = judge.judge(reflection_payload={}, hypothesis_payload={})

    assert len(result["revision_signals"]) == 1
    assert result["revision_signals"][0]["dimension"] == "CONTEXT"
    assert result["revision_signals"][0]["confidence"] == 1.0


def test_phase21_rejects_generic_target_and_multi_action_signal():
    judge = HypothesisRelationJudge(RelationLLM({
        "relation":"REFINES","confidence":0.9,"reason":"hardening",
        "revision_signals":[
            {"dimension":"CONTEXT","operation":"ADD_CONDITION","target_field":"applicable_context","before_data":False,"observed_data":True,"canonical_change":"too generic","confidence":0.9},
            {"dimension":"ACTION","operation":"EXPAND","target_field":"workflow_structure","before_data":"old","observed_data":["step-a","step-b"],"canonical_change":"two changes in one signal","confidence":0.9},
        ]}))
    result = judge.judge(reflection_payload={}, hypothesis_payload={})
    assert result["revision_signals"] == []
    assert len(result["rejected_revision_signal_reasons"]) == 2


def test_phase21_rejects_execution_context_as_hypothesis_condition():
    judge = HypothesisRelationJudge(RelationLLM({
        "relation":"REFINES","confidence":0.9,"reason":"execution context",
        "revision_signals":[{"dimension":"CONTEXT","operation":"ADD_CONDITION","target_field":"available_skills","before_data":[],"observed_data":["recall_relevant_memory"],"canonical_change":"require recall skill","confidence":0.9}]}))
    result = judge.judge(reflection_payload={}, hypothesis_payload={})
    assert result["revision_signals"] == []


def test_phase21_user_satisfaction_requires_explicit_feedback():
    payload={"relation":"REFINES","confidence":0.9,"reason":"feedback","revision_signals":[{"dimension":"OUTCOME","operation":"EXPAND","target_field":"user_satisfaction","before_data":"unknown","observed_data":"higher","canonical_change":"increase user satisfaction","confidence":0.9}]}
    judge = HypothesisRelationJudge(RelationLLM(payload))
    assert judge.judge(reflection_payload={}, hypothesis_payload={})["revision_signals"] == []
    result=judge.judge(reflection_payload={"user_feedback":"这样更清楚"}, hypothesis_payload={})
    assert result["revision_signals"][0]["target_field"] == "user_satisfaction"


def test_phase21_existing_invalid_signal_is_hidden_not_deleted(db):
    from app.models import RevisionSignal

    _, hypothesis = _create_hypothesis(db)
    reflection = _make_reflection(
        db, interaction_id=399, complexity="medium", seed="legacy-invalid-signal"
    )
    signal = RevisionSignal(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        reflection_id=reflection.id,
        dimension="ACTION",
        operation="EXPAND",
        target_field="do_data.text",
        before_data="old",
        observed_data=["a", "b"],
        canonical_change="legacy broad signal",
        confidence=0.9,
    )
    db.add(signal)
    db.commit()

    service = RevisionSignalService(db)
    assert service.list_all() == []
    assert service.count_invalid() == 1
    assert len(service.list_all(valid_only=False)) == 1


def test_context_expansion_rejects_incidental_complexity_from_real_interaction(db):
    from app.audit.service import AuditService
    from app.evolution.service import EvolutionMemoryService

    audit = AuditService(db)
    interaction = audit.save_interaction(
        session_id="ctx-leak",
        user_input="你以后给我设计架构时，先讲数据流，再讲模块接口，这样我更容易理解。",
        assistant_response="好的",
        strategy_name="default_conversation_strategy_v2",
        record_metadata={"learning_eligibility": {"category": "natural"}},
    )
    evolution = EvolutionMemoryService(db)
    experience = evolution.save_experience(
        interaction_id=interaction.id,
        task_type="architecture_design",
        strategy_name="default_conversation_strategy",
        strategy_version=2,
        skills_used=[],
        result_summary="preference-feedback",
        success=True,
        score=0.95,
        context_data={
            "task_type": "architecture_design",
            "complexity": "low",
            "interaction_mode": "direct",
        },
    )
    reflection = evolution.save_reflection(
        experience_id=experience.id,
        reflection_type="success",
        observation="用户明确给出输出顺序偏好",
        insight="明确输出顺序偏好可改善理解",
        applicable_context={
            "task_type": "architecture_design",
            "complexity": "low",
            "interaction_mode": "direct",
        },
        expected_effect="提高理解度",
        canonical_signal="preference-order",
        strategy_signal="按数据流 -> 模块接口组织回答",
        confidence=0.9,
    )

    service = HypothesisService(db)
    base = _make_reflection(db, interaction_id=901, complexity="high", seed="base-high")
    hypothesis = service.create_from_reflection(
        reflection=base,
        canonical_signal=base.canonical_signal,
    )
    judge = HypothesisRelationJudge(
        RelationLLM(
            {
                "relation": "REFINES",
                "confidence": 0.95,
                "reason": "preference improves response structure",
                "revision_signals": [
                    {
                        "dimension": "CONTEXT",
                        "operation": "EXPAND",
                        "target_field": "complexity",
                        "before_data": ["high"],
                        "observed_data": ["low"],
                        "canonical_change": "complexity: high -> include observed low",
                        "confidence": 0.88,
                    },
                    {
                        "dimension": "CONTEXT",
                        "operation": "EXPAND",
                        "target_field": "interaction_mode",
                        "before_data": ["collaborative"],
                        "observed_data": ["direct"],
                        "canonical_change": "interaction mode: collaborative -> include direct",
                        "confidence": 0.86,
                    },
                    {
                        "dimension": "ACTION",
                        "operation": "EXPAND",
                        "target_field": "response_structure",
                        "before_data": "existing structure",
                        "observed_data": "数据流 -> 模块接口 -> 内部实现",
                        "canonical_change": "add explicit user-preferred response order",
                        "confidence": 0.9,
                    },
                ],
            }
        )
    )
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=StaticRetriever(hypothesis.id),
        score_threshold=0.0,
    )

    result = aggregator.process(reflection)
    signals = RevisionSignalService(db).list_all(hypothesis_id=hypothesis.id)

    assert result["revision_signal_count"] == 1
    assert [(s.dimension, s.target_field) for s in signals] == [
        ("ACTION", "response_structure")
    ]
    assert signals[0].provenance_data["source_interaction_id"] == interaction.id


def test_context_expansion_allows_explicit_user_scope(db):
    from app.audit.service import AuditService
    from app.evolution.service import EvolutionMemoryService

    interaction = AuditService(db).save_interaction(
        session_id="ctx-explicit",
        user_input="无论简单任务还是高复杂度架构设计，都按这个顺序讲。",
        assistant_response="好的",
        strategy_name="default_conversation_strategy_v2",
    )
    evolution = EvolutionMemoryService(db)
    experience = evolution.save_experience(
        interaction_id=interaction.id,
        task_type="architecture_design",
        strategy_name="default_conversation_strategy",
        strategy_version=2,
        skills_used=[],
        result_summary="explicit complexity scope",
        success=True,
        score=0.9,
        context_data={"task_type": "architecture_design", "complexity": "low"},
    )
    reflection = evolution.save_reflection(
        experience_id=experience.id,
        reflection_type="success",
        observation="用户显式扩大适用复杂度",
        insight="该顺序适用于简单任务",
        applicable_context={"task_type": "architecture_design", "complexity": "low"},
        expected_effect="一致讲解顺序",
        canonical_signal="explicit-low",
        strategy_signal="简单和复杂任务都使用该结构",
        confidence=0.9,
    )
    service = HypothesisService(db)
    base = _make_reflection(db, interaction_id=902, complexity="high", seed="base-high-explicit")
    hypothesis = service.create_from_reflection(reflection=base, canonical_signal=base.canonical_signal)
    judge = HypothesisRelationJudge(RelationLLM({
        "relation": "REFINES", "confidence": 0.93, "reason": "explicit scope",
        "revision_signals": [{
            "dimension": "CONTEXT", "operation": "EXPAND", "target_field": "complexity",
            "before_data": ["high"], "observed_data": ["low"],
            "canonical_change": "complexity: high -> include observed low", "confidence": 0.9,
        }],
    }))
    result = HypothesisAggregator(
        db, relation_judge=judge, retriever=StaticRetriever(hypothesis.id), score_threshold=0.0
    ).process(reflection)

    signals = RevisionSignalService(db).list_all(hypothesis_id=hypothesis.id)
    assert result["revision_signal_count"] == 1
    assert signals[0].target_field == "complexity"
    assert signals[0].provenance_data["context_evidence_basis"] == "explicit_user_scope"
