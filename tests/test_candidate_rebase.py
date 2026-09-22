import json

from app.hypothesis.ancestor_revalidation import AncestorSignalRevalidationService
from app.hypothesis.candidate_rebase import CandidateRebaseService
from app.hypothesis.candidate_revision import CandidateRevisionService
from app.hypothesis.revision_commit import RevisionCommitService
from app.models import (
    Experience,
    HypothesisEvidence,
    ReflectionRecord,
    RevisionAnalysisRecord,
    RevisionProposal,
    RevisionSignal,
)
from tests.test_candidate_revision import FakeCandidateLLM
from tests.test_revision_aggregation import _add_refine_signal
from tests.test_revision_commit import _evaluated_low_candidate
from tests.test_revision_reconciliation import _normalize_outcome_source_signals


class CombinedRebaseLLM:
    """Understands both ancestor revalidation and candidate-local evaluation prompts."""

    def __init__(self):
        self.revalidation_calls = 0
        self.candidate_calls = 0

    def json_chat(self, messages, temperature=0.1):
        payload = json.loads(messages[-1]["content"])
        if "ancestor_signals" in payload:
            self.revalidation_calls += 1
            return {
                "assessments": [
                    {
                        "signal_id": item["signal_id"],
                        "outcome": "STILL_APPLIES",
                        "confidence": 0.93,
                        "reason": "still needed on r2",
                    }
                    for item in payload["ancestor_signals"]
                ]
            }
        self.candidate_calls += 1
        return {
            "signal_assessments": [
                {
                    "signal_id": item["signal_id"],
                    "resolution": "RESOLVED",
                    "confidence": 0.95,
                    "reason": "rebased candidate resolves the local signal",
                }
                for item in payload.get("source_signals", [])
            ],
            "guard_assessments": [
                {
                    "evidence_id": item["evidence_id"],
                    "verdict": "PASS",
                    "confidence": 0.9,
                    "reason": "preserved",
                }
                for item in payload.get("guard_evidence", [])
            ],
        }


def _stale_action_candidate_after_r2(db):
    hypothesis, _low_proposal, low_candidate = _evaluated_low_candidate(db)
    _normalize_outcome_source_signals(db, low_candidate)

    signals = [
        _add_refine_signal(
            db,
            hypothesis,
            interaction_id=9100 + index,
            reflection_seed=f"response structure {index}",
            target_field="response_structure",
        )
        for index in range(3)
    ]
    analysis = RevisionAnalysisRecord(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        signal_ids=[item.id for item in signals],
        candidate_fingerprint=f"stale-action-{hypothesis.id}",
        consistent=True,
        direction_confidence=0.9,
        reason="same response structure direction",
    )
    db.add(analysis)
    db.flush()
    proposal = RevisionProposal(
        analysis_id=analysis.id,
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        before_data=None,
        proposed_after_data={
            "response_structure": {
                "ordered_sections": ["data flow", "interfaces"],
                "checks": ["boundaries", "anti-patterns"],
            }
        },
        reason="stale action proposal",
        source_signal_ids=[item.id for item in signals],
        independent_signal_count=3,
        direction_confidence=0.9,
        status="proposed",
    )
    db.add(proposal)
    db.commit()

    stale_candidate = CandidateRevisionService(db, FakeCandidateLLM()).evaluate_proposal(proposal).candidate
    assert stale_candidate.status == "needs_review"

    RevisionCommitService(db).auto_apply_ready(max_candidates=1)
    db.refresh(hypothesis)
    db.refresh(stale_candidate)
    assert hypothesis.current_revision == 2
    assert stale_candidate.status == "stale"
    return hypothesis, proposal, stale_candidate, signals


def test_phase5_5_clean_candidate_rebase_creates_new_proposal_and_reevaluated_candidate(db):
    hypothesis, old_proposal, stale_candidate, source_signals = _stale_action_candidate_after_r2(db)
    llm = CombinedRebaseLLM()

    revalidation = AncestorSignalRevalidationService(db, llm, max_signals_per_batch=16)
    revalidated = revalidation.revalidate_hypothesis(hypothesis.id)
    assert set(revalidated.rebased_signal_ids)
    assert llm.revalidation_calls == 1

    service = CandidateRebaseService(db, llm)
    result = service.rebase_candidate(stale_candidate.id)

    assert result.record.status == "rebased"
    assert result.candidate is not None
    assert result.candidate.id != stale_candidate.id
    assert result.candidate.base_revision == 2
    assert result.candidate.source_proposal_id == result.record.rebased_proposal_id
    assert result.candidate.evaluation_data["rebase"]["source_candidate_id"] == stale_candidate.id
    assert result.record.source_signal_ids == [item.id for item in source_signals]
    assert len(result.record.target_signal_ids) == len(source_signals)
    assert llm.candidate_calls == 1

    db.refresh(old_proposal)
    assert old_proposal.status == "stale"
    # Rebase never changes the current revision by itself.
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 2

    # Idempotent repeat returns the already rebased Candidate without another LLM call.
    second = service.rebase_candidate(stale_candidate.id)
    assert second.candidate.id == result.candidate.id
    assert second.llm_called is False
    assert llm.candidate_calls == 1


def test_phase5_5_candidate_rebase_waits_until_all_source_signals_are_revalidated(db):
    _hypothesis, _old_proposal, stale_candidate, _signals = _stale_action_candidate_after_r2(db)
    llm = CombinedRebaseLLM()
    service = CandidateRebaseService(db, llm)

    result = service.rebase_candidate(stale_candidate.id)

    assert result.record.status == "not_ready"
    assert result.candidate is None
    assert "not_fully_revalidated" in result.record.reason
    assert llm.candidate_calls == 0


def test_phase5_5_candidate_rebase_defers_when_new_same_target_signal_exists(db):
    hypothesis, _old_proposal, stale_candidate, _signals = _stale_action_candidate_after_r2(db)
    llm = CombinedRebaseLLM()
    revalidated = AncestorSignalRevalidationService(db, llm).revalidate_hypothesis(hypothesis.id)
    assert revalidated.rebased_signal_ids

    # New r2 evidence arrived after the stale r1 proposal was formed. It belongs
    # to the same semantic group, so the old proposal must not be silently reused.
    exp = Experience(
        interaction_id=9999,
        task_type="architecture_design",
        strategy_name="default",
        strategy_version=2,
        skills_used=[],
        result_summary="new current-head response preference",
        success=True,
        score=1.0,
        context_data={"task_type": "architecture_design", "complexity": "high", "interaction_mode": "collaborative"},
    )
    db.add(exp); db.flush()
    reflection = ReflectionRecord(
        experience_id=exp.id,
        reflection_type="success",
        observation="new current-head response preference",
        lesson="new current-head response preference",
        insight="new current-head response preference",
        applicable_context={"task_type": "architecture_design", "complexity": "high", "interaction_mode": "collaborative"},
        strategy_signal="new current-head response preference",
        expected_effect="clarity",
        canonical_signal="new current-head response preference",
        confidence=0.9,
        status="active",
    )
    db.add(reflection); db.flush()
    evidence = HypothesisEvidence(
        hypothesis_id=hypothesis.id,
        hypothesis_revision=2,
        reflection_id=reflection.id,
        relation_type="REFINES",
        relation_confidence=0.9,
        relation_reason="new r2 refine",
        is_independent=True,
    )
    db.add(evidence); db.flush()
    extra = RevisionSignal(
        hypothesis_id=hypothesis.id,
        base_revision=2,
        reflection_id=reflection.id,
        evidence_id=evidence.id,
        interaction_id=9999,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        before_data=None,
        observed_data="new explicit response ordering",
        canonical_change="response_structure: expand",
        confidence=0.9,
        provenance_data={},
        resolution_status="open",
    )
    db.add(extra); db.commit()

    result = CandidateRebaseService(db, llm).rebase_candidate(stale_candidate.id)

    assert result.record.status == "needs_fresh_analysis"
    assert result.candidate is None
    assert extra.id in result.record.rebase_data["extra_current_signal_ids"]
    assert llm.candidate_calls == 0
