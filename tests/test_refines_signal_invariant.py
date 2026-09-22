from __future__ import annotations

from app.evolution.service import EvolutionMemoryService
from app.hypothesis.aggregator import HypothesisAggregator
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.retriever import HypothesisMatch
from app.hypothesis.service import HypothesisService


class OneHypothesisRetriever:
    def __init__(self, hypothesis_id):
        self.hypothesis_id = hypothesis_id

    def search(self, query, *, top_k=5, score_threshold=0.0):
        return [HypothesisMatch(self.hypothesis_id, 0.99)]

    def index(self, hypothesis):
        pass


class StaticLLM:
    def __init__(self, payload):
        self.payload = payload

    def json_chat(self, messages, temperature=0.0):
        return self.payload


def _reflection(db, interaction_id, seed):
    evolution = EvolutionMemoryService(db)
    experience = evolution.save_experience(
        interaction_id=interaction_id,
        task_type="architecture_design",
        strategy_name="default_conversation_strategy",
        strategy_version=2,
        skills_used=[],
        result_summary=seed,
        success=True,
        score=0.95,
        context_data={"task_type": "architecture_design"},
    )
    return evolution.save_reflection(
        experience_id=experience.id,
        reflection_type="success",
        observation=seed,
        insight="用户偏好稳定的架构讲解顺序",
        applicable_context={"task_type": "architecture_design"},
        expected_effect="提升理解",
        canonical_signal=seed,
        strategy_signal="先讲数据流，再讲模块接口",
        confidence=0.9,
    )


def test_refines_without_signal_is_invalid():
    judge = HypothesisRelationJudge(
        StaticLLM({
            "relation": "REFINES",
            "confidence": 0.9,
            "reason": "needs a local change",
            "revision_signals": [],
        })
    )
    result = judge.judge(
        reflection_payload={"insight": "x"},
        hypothesis_payload={
            "id": 1,
            "revision": 1,
            "when_data": {},
            "do_data": {"text": "old"},
            "expect_data": {"text": "clear"},
        },
    )
    assert result["relation"] == "REFINES"
    assert result["decision_valid"] is False
    assert result["revision_signals"] == []


def test_invalid_refines_does_not_create_evidence_or_duplicate_hypothesis(db):
    service = HypothesisService(db)
    base = _reflection(db, 9101, "base")
    hypothesis = service.create_from_reflection(
        reflection=base,
        canonical_signal=base.canonical_signal,
    )
    incoming = _reflection(db, 9102, "incoming")

    judge = HypothesisRelationJudge(
        StaticLLM({
            "decisions": [{
                "hypothesis_id": hypothesis.id,
                "relation": "REFINES",
                "confidence": 0.91,
                "reason": "same rule but missing atomic change",
                "revision_signals": [],
            }]
        })
    )
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=OneHypothesisRetriever(hypothesis.id),
        score_threshold=0.0,
    )

    result = aggregator.process(incoming)

    assert result["action"] == "skipped_invalid_relation"
    assert result["hypothesis_id"] == hypothesis.id
    assert result["revision_signal_count"] == 0
    assert len(service.list_all(include_inactive=False)) == 1
    assert service.get(hypothesis.id).refinement_count == 0
