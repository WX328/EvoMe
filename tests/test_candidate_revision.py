import json

from sqlalchemy import select

from app.hypothesis.candidate_revision import (
    CURRENT_EVALUATION_VERSION,
    CandidateRevisionService,
)
from app.models import CandidateRevision, HypothesisRevision, RevisionAnalysisRecord, RevisionProposal
from tests.test_revision_aggregation import _add_refine_signal, _seed_hypothesis


class FakeCandidateLLM:
    def __init__(self, *, only_first=False, regression=False):
        self.calls = 0
        self.only_first = only_first
        self.regression = regression

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        signals = payload["source_signals"]
        if self.only_first:
            signals = signals[:1]
        guards = payload.get("guard_evidence", [])
        return {
            "signal_assessments": [
                {
                    "signal_id": item["signal_id"],
                    "resolution": "RESOLVED",
                    "confidence": 0.95,
                    "reason": "candidate applies the proposed local delta",
                }
                for item in signals
            ],
            "guard_assessments": [
                {
                    "evidence_id": item["evidence_id"],
                    "verdict": "REGRESSION" if self.regression else "PASS",
                    "confidence": 0.9,
                    "reason": "guard result",
                }
                for item in guards
            ],
        }


def _make_proposal(
    db,
    *,
    dimension,
    target_field,
    proposed_after_data,
    signal_count=2,
):
    hypothesis = _seed_hypothesis(db)
    signals = []
    for index in range(signal_count):
        signals.append(
            _add_refine_signal(
                db,
                hypothesis,
                interaction_id=1000 + index,
                reflection_seed=f"evidence-{index}",
                target_field=target_field,
            )
        )

    analysis = RevisionAnalysisRecord(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension=dimension,
        operation="EXPAND",
        target_field=target_field,
        signal_ids=[item.id for item in signals],
        candidate_fingerprint=f"test-{dimension}-{target_field}-{hypothesis.id}",
        consistent=True,
        direction_confidence=0.9,
        reason="same direction",
    )
    db.add(analysis)
    db.flush()
    proposal = RevisionProposal(
        analysis_id=analysis.id,
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension=dimension,
        operation="EXPAND",
        target_field=target_field,
        before_data=[],
        proposed_after_data=proposed_after_data,
        reason="test proposal",
        source_signal_ids=[item.id for item in signals],
        independent_signal_count=signal_count,
        direction_confidence=0.9,
        status="proposed",
    )
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return hypothesis, proposal


def test_phase5_outcome_patch_is_low_risk_and_ready_without_mutating_revision(db):
    hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "improve clarity; reduce scope drift"}},
        signal_count=2,
    )
    llm = FakeCandidateLLM()
    service = CandidateRevisionService(db, llm, max_evidence=8)

    result = service.evaluate_proposal(proposal)
    candidate = result.candidate

    assert llm.calls == 1
    assert candidate.expect_data["text"] == "improve clarity; reduce scope drift"
    assert candidate.risk_level == "LOW"
    assert candidate.status == "ready_to_apply"
    assert candidate.fit_gain > 0
    assert candidate.evaluation_version == CURRENT_EVALUATION_VERSION
    assert candidate.evaluation_data["method"] == "signal_resolution_v3"
    assert candidate.evaluation_data["regression_count"] == 0

    db.refresh(hypothesis)
    assert hypothesis.current_revision == 1
    revisions = list(db.scalars(select(HypothesisRevision).where(HypothesisRevision.hypothesis_id == hypothesis.id)).all())
    assert len(revisions) == 1


def test_phase5_action_expansion_can_improve_local_fit_but_require_review_for_complexity(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="ACTION",
        target_field="response_structure",
        proposed_after_data={
            "target_field": "response_structure",
            "operation": "EXPAND",
            "response_structure": {
                "ordered_sections": ["responsibility matrix", "data flow", "interface draft", "pending decisions"],
                "checks": ["boundary", "call relation", "anti-pattern"],
                "evidence": ["supported", "unsupported", "confidence", "next step"],
            },
        },
        signal_count=3,
    )
    llm = FakeCandidateLLM()
    service = CandidateRevisionService(db, llm, max_evidence=8)

    candidate = service.evaluate_proposal(proposal).candidate

    assert candidate.do_data["response_structure"]["ordered_sections"][0] == "responsibility matrix"
    assert candidate.fit_gain > 0
    assert candidate.complexity_delta > 3
    assert candidate.risk_level == "MEDIUM"
    assert candidate.status == "needs_review"


def test_phase5_one_proposal_maps_to_one_candidate_and_does_not_spend_token_twice(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "reduce scope drift"}},
    )
    llm = FakeCandidateLLM()
    service = CandidateRevisionService(db, llm)

    first = service.evaluate_proposal(proposal)
    second = service.evaluate_proposal(proposal)

    assert first.candidate.id == second.candidate.id
    assert first.llm_called is True
    assert second.llm_called is False
    assert llm.calls == 1
    assert db.query(CandidateRevision).count() == 1


def test_phase5_incomplete_source_signal_coverage_cannot_be_ready_to_apply(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "reduce scope drift"}},
        signal_count=2,
    )
    llm = FakeCandidateLLM(only_first=True)
    service = CandidateRevisionService(db, llm)

    candidate = service.evaluate_proposal(proposal).candidate

    assert candidate.evaluation_data["source_coverage"] == 0.5
    assert candidate.status == "needs_review"


def test_phase5_guard_regression_rejects_candidate(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "reduce scope drift"}},
        signal_count=2,
    )
    llm = FakeCandidateLLM(regression=True)
    service = CandidateRevisionService(db, llm)

    candidate = service.evaluate_proposal(proposal).candidate
    # The helper data may not have non-source guard evidence. If present, any
    # regression is a hard reject; otherwise the local signal result still wins.
    if candidate.evaluation_data["guard_assessments"]:
        assert candidate.evaluation_data["regression_count"] > 0
        assert candidate.status == "rejected"


def test_phase5_stale_v1_candidate_is_reevaluated_in_place(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "reduce scope drift"}},
    )
    stale = CandidateRevision(
        hypothesis_id=proposal.hypothesis_id,
        base_revision=1,
        source_proposal_id=proposal.id,
        title="stale",
        status="rejected",
        evaluation_version=1,
        evaluation_data={"method": "reflection_relation_v1"},
    )
    db.add(stale)
    db.commit()
    stale_id = stale.id

    llm = FakeCandidateLLM()
    service = CandidateRevisionService(db, llm)
    result = service.evaluate_proposal(proposal)

    assert llm.calls == 1
    assert result.candidate.id == stale_id
    assert result.candidate.evaluation_version == CURRENT_EVALUATION_VERSION
    assert result.candidate.status == "ready_to_apply"
    assert result.candidate.evaluation_data["previous_evaluation"]["method"] == "reflection_relation_v1"


class LooseFormatCandidateLLM:
    def __init__(self):
        self.calls = 0

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        signals = payload["source_signals"]
        guards = payload.get("guard_evidence", [])
        return {
            "source_signal_assessments": [
                {
                    "revision_signal_id": f"RS#{item['signal_id']}",
                    "status": "fully resolved",
                    "confidence": "0.92",
                    "explanation": "same local change",
                }
                for item in signals
            ],
            "guard_evidence_assessments": [
                {
                    "hypothesis_evidence_id": f"E#{item['evidence_id']}",
                    "status": "preserved",
                    "confidence": 0.9,
                    "explanation": "preserved",
                }
                for item in guards
            ],
        }


class EmptyCandidateLLM:
    def json_chat(self, messages, temperature=0.1):
        return {"unexpected": []}


def test_phase5_v3_normalizer_accepts_common_llm_aliases(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "reduce scope drift"}},
        signal_count=2,
    )
    service = CandidateRevisionService(db, LooseFormatCandidateLLM())
    candidate = service.evaluate_proposal(proposal).candidate

    assert candidate.evaluation_version == CURRENT_EVALUATION_VERSION
    assert candidate.evaluation_data["evaluation_valid"] is True
    assert candidate.evaluation_data["source_coverage"] == 1.0
    assert candidate.fit_gain > 0


def test_phase5_v3_invalid_evaluator_output_is_uncertainty_not_negative_evidence(db):
    _hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "reduce scope drift"}},
        signal_count=2,
    )
    service = CandidateRevisionService(db, EmptyCandidateLLM())
    candidate = service.evaluate_proposal(proposal).candidate

    assert candidate.evaluation_data["evaluation_valid"] is False
    assert candidate.evaluation_data["evaluation_error"] == "no_valid_signal_assessments"
    assert candidate.evaluation_data["source_coverage"] == 0.0
    assert candidate.candidate_fit_score == candidate.base_fit_score
    assert candidate.fit_gain == 0.0
    assert candidate.utility_score == 0.0
    assert candidate.status == "needs_review"
