from __future__ import annotations

import json

from app.evolution.service import EvolutionMemoryService
from app.hypothesis.aggregator import HypothesisAggregator
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.retriever import HypothesisMatch
from app.hypothesis.service import HypothesisService
from app.models import RevisionSignal


class CountingLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_messages = None

    def json_chat(self, messages, temperature=0.0):
        self.calls += 1
        self.last_messages = messages
        return self.payload


class MultiRetriever:
    def __init__(self, hypothesis_ids):
        self.hypothesis_ids = list(hypothesis_ids)

    def search(self, query, *, top_k=5, score_threshold=0.0):
        return [HypothesisMatch(hid, 0.99 - i * 0.01) for i, hid in enumerate(self.hypothesis_ids)]

    def index(self, hypothesis):
        pass


def _reflection(db, interaction_id, seed):
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
        context_data={"task_type": "architecture_design", "complexity": "high"},
    )
    return evolution.save_reflection(
        experience_id=experience.id,
        reflection_type="pattern",
        observation=seed,
        insight="架构设计需要稳定的数据流与接口边界",
        applicable_context={"task_type": "architecture_design", "complexity": "high"},
        expected_effect="减少职责混淆",
        canonical_signal=seed,
        strategy_signal="先梳理数据流，再定义模块接口",
        confidence=0.9,
    )


def test_relation_judge_batch_uses_one_compact_llm_call():
    llm = CountingLLM(
        {
            "decisions": [
                {"hypothesis_id": 1, "relation": "SUPPORTS", "confidence": 0.91, "reason": "covered", "revision_signals": []},
                {"hypothesis_id": 2, "relation": "UNRELATED", "confidence": 0.88, "reason": "different rule", "revision_signals": []},
            ]
        }
    )
    judge = HypothesisRelationJudge(llm)
    reflection = {
        "insight": "prefer data flow first",
        "applicable_context": {"task_type": "architecture_design"},
        "proposed_change": "data flow before interfaces",
        "expected_effect": "clearer design",
        "canonical_signal": "duplicate semantic text that should not be sent",
        "user_feedback": "explicit preference",
    }
    hypotheses = [
        {
            "id": 1,
            "revision": 2,
            "when_data": {"task_type": "architecture_design"},
            "do_data": {"text": "data flow then interfaces", "source_signal_ids": [1, 2]},
            "expect_data": {"text": "clearer boundaries"},
            "canonical_signal": "duplicate h1 canonical",
            "applicable_context": {"task_type": "architecture_design"},
            "proposed_change": "duplicate h1 action",
            "expected_effect": "duplicate h1 effect",
        },
        {
            "id": 2,
            "revision": 1,
            "when_data": {"task_type": "debugging"},
            "do_data": {"text": "verify errors"},
            "expect_data": {"text": "correctness"},
        },
    ]

    decisions = judge.judge_batch(
        reflection_payload=reflection,
        hypothesis_payloads=hypotheses,
    )

    assert llm.calls == 1
    assert [item["hypothesis_id"] for item in decisions] == [1, 2]
    user_payload = json.loads(llm.last_messages[-1]["content"])
    assert len(user_payload["hypotheses"]) == 2
    assert "canonical_signal" not in user_payload["reflection"]
    assert "source_signal_ids" not in json.dumps(user_payload, ensure_ascii=False)


class BatchOnlyJudge:
    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def judge_batch(self, *, reflection_payload, hypothesis_payloads):
        self.calls += 1
        self.batch_sizes.append(len(hypothesis_payloads))
        return [
            {
                "hypothesis_id": hypothesis_payloads[0]["id"],
                "relation": "SUPPORTS",
                "confidence": 0.95,
                "reason": "first hypothesis matches",
                "revision_signals": [],
            },
            {
                "hypothesis_id": hypothesis_payloads[1]["id"],
                "relation": "UNRELATED",
                "confidence": 0.9,
                "reason": "second hypothesis differs",
                "revision_signals": [],
            },
        ]


def test_aggregator_batches_multiple_retrieved_hypotheses(db):
    service = HypothesisService(db)
    first = _reflection(db, 801, "first-base")
    second = _reflection(db, 802, "second-base")
    h1 = service.create_from_reflection(reflection=first, canonical_signal="h1-base")
    h2 = service.create_from_reflection(reflection=second, canonical_signal="h2-base")
    incoming = _reflection(db, 803, "incoming")

    judge = BatchOnlyJudge()
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=MultiRetriever([h1.id, h2.id]),
        score_threshold=0.0,
    )

    result = aggregator.process(incoming)

    assert judge.calls == 1
    assert judge.batch_sizes == [2]
    assert result["action"] == "supports"
    assert result["hypothesis_id"] == h1.id


def test_missing_batch_decision_fails_closed_without_duplicate(db):
    service = HypothesisService(db)
    base = _reflection(db, 811, "base")
    hypothesis = service.create_from_reflection(
        reflection=base,
        canonical_signal="h1-base",
    )
    incoming = _reflection(db, 812, "incoming")
    judge = HypothesisRelationJudge(CountingLLM({"decisions": []}))
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=MultiRetriever([hypothesis.id]),
        score_threshold=0.0,
    )

    result = aggregator.process(incoming)

    assert result["action"] == "skipped_ambiguous_relation"
    assert len(service.list_all(include_inactive=False)) == 1


def test_low_confidence_unrelated_fails_closed_without_duplicate(db):
    service = HypothesisService(db)
    base = _reflection(db, 821, "base")
    hypothesis = service.create_from_reflection(
        reflection=base,
        canonical_signal="h1-base",
    )
    incoming = _reflection(db, 822, "incoming")
    judge = HypothesisRelationJudge(
        CountingLLM({
            "decisions": [{
                "hypothesis_id": hypothesis.id,
                "relation": "UNRELATED",
                "confidence": 0.2,
                "reason": "uncertain",
                "revision_signals": [],
            }]
        })
    )
    aggregator = HypothesisAggregator(
        db,
        relation_judge=judge,
        retriever=MultiRetriever([hypothesis.id]),
        score_threshold=0.0,
    )

    result = aggregator.process(incoming)

    assert result["action"] == "skipped_ambiguous_relation"
    assert len(service.list_all(include_inactive=False)) == 1


def test_relation_judge_receives_active_pending_signal_context(db):
    service = HypothesisService(db)
    base = _reflection(db, 831, "base")
    hypothesis = service.create_from_reflection(
        reflection=base,
        canonical_signal="h1-base",
    )
    db.add(
        RevisionSignal(
            hypothesis_id=hypothesis.id,
            base_revision=1,
            reflection_id=base.id,
            dimension="ACTION",
            operation="EXPAND",
            target_field="response_structure",
            before_data="先说明职责边界",
            observed_data="数据流 -> 模块接口",
            canonical_change="默认按数据流 -> 模块接口组织回答",
            confidence=0.9,
        )
    )
    db.commit()

    incoming = _reflection(db, 832, "incoming")
    llm = CountingLLM({
        "decisions": [{
            "hypothesis_id": hypothesis.id,
            "relation": "UNRELATED",
            "confidence": 0.9,
            "reason": "model missed pending continuity",
            "revision_signals": [],
        }]
    })
    aggregator = HypothesisAggregator(
        db,
        relation_judge=HypothesisRelationJudge(llm),
        retriever=MultiRetriever([hypothesis.id]),
        score_threshold=0.0,
    )

    result = aggregator.process(incoming)

    payload = json.loads(llm.last_messages[-1]["content"])
    pending = payload["hypotheses"][0]["pending_signals"]
    assert len(pending) == 1
    assert pending[0]["target_field"] == "response_structure"
    assert pending[0]["observed_data"] == "数据流 -> 模块接口"
    assert result["action"] == "skipped_ambiguous_relation"
    assert len(service.list_all(include_inactive=False)) == 1
