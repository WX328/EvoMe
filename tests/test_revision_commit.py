from sqlalchemy import func, select

from app.hypothesis.candidate_revision import CandidateRevisionService
from app.hypothesis.revision_commit import RevisionCommitService
from app.models import (
    CandidateRevision,
    HypothesisEvidence,
    HypothesisRevision,
    RevisionAnalysisRecord,
    RevisionApplicationEvent,
    RevisionProposal,
)
from tests.test_candidate_revision import FakeCandidateLLM, _make_proposal


def _evaluated_low_candidate(db):
    hypothesis, proposal = _make_proposal(
        db,
        dimension="OUTCOME",
        target_field="scope_drift",
        proposed_after_data={"expect": {"text": "improve clarity; reduce scope drift"}},
        signal_count=2,
    )
    candidate = CandidateRevisionService(db, FakeCandidateLLM()).evaluate_proposal(proposal).candidate
    assert candidate.status == "ready_to_apply"
    return hypothesis, proposal, candidate


def _manual_candidate_on_base(db, hypothesis, *, base_revision: int, suffix: str = "manual"):
    analysis = RevisionAnalysisRecord(
        hypothesis_id=hypothesis.id,
        base_revision=base_revision,
        dimension="OUTCOME",
        operation="EXPAND",
        target_field="clarity",
        signal_ids=[],
        candidate_fingerprint=f"{suffix}-{hypothesis.id}-{base_revision}",
        consistent=True,
        direction_confidence=0.9,
        reason="manual test proposal",
    )
    db.add(analysis)
    db.flush()
    proposal = RevisionProposal(
        analysis_id=analysis.id,
        hypothesis_id=hypothesis.id,
        base_revision=base_revision,
        dimension="OUTCOME",
        operation="EXPAND",
        target_field="clarity",
        before_data=None,
        proposed_after_data={"expect": {"text": f"manual change {suffix}"}},
        reason="manual test proposal",
        source_signal_ids=[],
        independent_signal_count=0,
        direction_confidence=0.9,
        status="proposed",
    )
    db.add(proposal)
    db.flush()
    base = db.scalar(
        select(HypothesisRevision).where(
            HypothesisRevision.hypothesis_id == hypothesis.id,
            HypothesisRevision.revision_number == base_revision,
        )
    )
    candidate = CandidateRevision(
        hypothesis_id=hypothesis.id,
        base_revision=base_revision,
        source_proposal_id=proposal.id,
        title=base.title,
        insight=base.insight,
        when_data=base.when_data,
        do_data=base.do_data,
        expect_data={"text": f"manual change {suffix}"},
        canonical_signal=f"manual-{suffix}",
        status="needs_review",
        base_fit_score=0.325,
        candidate_fit_score=1.0,
        fit_gain=0.675,
        base_complexity_score=10,
        complexity_score=12,
        complexity_delta=2,
        risk_level="MEDIUM",
        risk_score=0.4,
        utility_score=0.4,
        evaluation_version=3,
        evidence_ids=[],
        evaluation_data={
            "method": "signal_resolution_v3",
            "evaluation_version": 3,
            "evaluation_valid": True,
            "source_coverage": 1.0,
            "regression_count": 0,
            "source_signal_assessments": [],
        },
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return proposal, candidate


def test_phase5_3_auto_apply_creates_immutable_r2_and_event(db):
    hypothesis, proposal, candidate = _evaluated_low_candidate(db)
    base_evidence_count = db.scalar(
        select(func.count()).select_from(HypothesisEvidence).where(
            HypothesisEvidence.hypothesis_id == hypothesis.id,
            HypothesisEvidence.hypothesis_revision == 1,
        )
    )

    service = RevisionCommitService(db, auto_min_utility=0.20)
    results = service.auto_apply_ready(max_candidates=2)

    assert len(results) == 1
    result = results[0]
    assert result.applied is True
    assert result.event.action == "AUTO_APPLY"
    assert result.event.from_revision == 1
    assert result.event.to_revision == 2
    assert result.event.candidate_id == candidate.id
    assert result.event.proposal_id == proposal.id

    db.refresh(hypothesis)
    db.refresh(candidate)
    db.refresh(proposal)
    assert hypothesis.current_revision == 2
    assert hypothesis.revision == 2
    assert candidate.status == "applied"
    assert proposal.status == "applied"

    revision = db.scalar(
        select(HypothesisRevision).where(
            HypothesisRevision.hypothesis_id == hypothesis.id,
            HypothesisRevision.revision_number == 2,
        )
    )
    assert revision is not None
    assert "scope drift" in revision.expect_data["text"]

    copied_count = db.scalar(
        select(func.count()).select_from(HypothesisEvidence).where(
            HypothesisEvidence.hypothesis_id == hypothesis.id,
            HypothesisEvidence.hypothesis_revision == 2,
        )
    )
    assert copied_count == base_evidence_count


def test_phase5_3_applying_one_candidate_marks_same_base_sibling_stale(db):
    hypothesis, _proposal, ready = _evaluated_low_candidate(db)
    _sibling_proposal, sibling = _manual_candidate_on_base(
        db, hypothesis, base_revision=1, suffix="sibling"
    )

    service = RevisionCommitService(db)
    results = service.auto_apply_ready(max_candidates=2)

    assert len(results) == 1
    db.refresh(ready)
    db.refresh(sibling)
    assert ready.status == "applied"
    assert sibling.status == "stale"
    assert "superseded_by_CR" in sibling.evaluation_data["stale_reason"]


def test_phase5_3_human_approve_can_apply_reviewed_medium_candidate(db):
    hypothesis, _proposal, _ready = _evaluated_low_candidate(db)
    # Keep the low candidate from auto applying in this test; approve a reviewed
    # medium Candidate explicitly instead.
    _medium_proposal, medium = _manual_candidate_on_base(
        db, hypothesis, base_revision=1, suffix="medium"
    )

    service = RevisionCommitService(db)
    result = service.approve_candidate(medium.id, reason="operator reviewed")

    assert result.applied is True
    assert result.event.action == "HUMAN_APPLY"
    assert result.event.reason == "operator reviewed"
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 2


def test_phase5_3_rollback_preserves_history_and_next_commit_uses_max_plus_one(db):
    hypothesis, _proposal, _candidate = _evaluated_low_candidate(db)
    service = RevisionCommitService(db)
    first = service.auto_apply_ready(max_candidates=1)[0]
    assert first.event.to_revision == 2

    rollback = service.rollback(hypothesis.id, 1, reason="test rollback")
    assert rollback.event.action == "ROLLBACK"
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 1
    assert db.scalar(
        select(func.count()).select_from(HypothesisRevision).where(
            HypothesisRevision.hypothesis_id == hypothesis.id
        )
    ) == 2

    _proposal2, candidate2 = _manual_candidate_on_base(
        db, hypothesis, base_revision=1, suffix="after-rollback"
    )
    second = service.approve_candidate(candidate2.id, reason="new revision after rollback")

    assert second.event.from_revision == 1
    assert second.event.to_revision == 3
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 3
    revision_numbers = list(
        db.scalars(
            select(HypothesisRevision.revision_number)
            .where(HypothesisRevision.hypothesis_id == hypothesis.id)
            .order_by(HypothesisRevision.revision_number.asc())
        ).all()
    )
    assert revision_numbers == [1, 2, 3]


def test_phase5_3_auto_apply_budget_blocks_second_auto_revision_within_window(db):
    hypothesis, _proposal, _candidate = _evaluated_low_candidate(db)
    service = RevisionCommitService(db, auto_per_hypothesis=1, auto_budget_hours=24)
    assert len(service.auto_apply_ready(max_candidates=1)) == 1
    service.rollback(hypothesis.id, 1)

    _proposal2, candidate2 = _manual_candidate_on_base(
        db, hypothesis, base_revision=1, suffix="budget"
    )
    # Make it pass the strict deterministic auto gate.
    candidate2.status = "ready_to_apply"
    candidate2.risk_level = "LOW"
    candidate2.risk_score = 0.1
    candidate2.utility_score = 0.5
    db.commit()

    assert service.auto_apply_ready(max_candidates=1) == []
    db.refresh(candidate2)
    assert candidate2.status == "ready_to_apply"
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 1


def test_phase5_3_stale_base_is_never_applied(db):
    hypothesis, _proposal, candidate = _evaluated_low_candidate(db)
    service = RevisionCommitService(db)
    service.auto_apply_ready(max_candidates=1)
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 2

    # Simulate an old ready candidate that still claims r1 after the head moved.
    analysis = RevisionAnalysisRecord(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension="OUTCOME",
        operation="EXPAND",
        target_field="efficiency",
        signal_ids=[],
        candidate_fingerprint=f"stale-{hypothesis.id}",
        consistent=True,
        direction_confidence=0.9,
        reason="stale",
    )
    db.add(analysis)
    db.flush()
    proposal = RevisionProposal(
        analysis_id=analysis.id,
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension="OUTCOME",
        operation="EXPAND",
        target_field="efficiency",
        proposed_after_data={"expect": {"text": "faster"}},
        reason="stale",
        source_signal_ids=[],
        independent_signal_count=0,
        direction_confidence=0.9,
        status="proposed",
    )
    db.add(proposal)
    db.flush()
    stale = CandidateRevision(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        source_proposal_id=proposal.id,
        title="stale",
        when_data={},
        do_data={},
        expect_data={"text": "faster"},
        canonical_signal="stale",
        status="ready_to_apply",
        fit_gain=0.5,
        risk_level="LOW",
        utility_score=0.5,
        evaluation_version=3,
        evaluation_data={
            "evaluation_valid": True,
            "source_coverage": 1.0,
            "regression_count": 0,
        },
    )
    db.add(stale)
    db.commit()

    # Budget is disabled here so only the stale-base guard decides the result.
    stale_service = RevisionCommitService(db, auto_per_hypothesis=10)
    assert stale_service.auto_apply_ready(max_candidates=1) == []
    db.refresh(stale)
    assert stale.status == "stale"
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 2


def test_phase5_3_human_reject_is_audited(db):
    _hypothesis, _proposal, candidate = _evaluated_low_candidate(db)
    service = RevisionCommitService(db)
    result = service.reject_candidate(candidate.id, reason="too broad for now")

    db.refresh(candidate)
    assert candidate.status == "rejected"
    assert result.event.action == "HUMAN_REJECT"
    assert result.event.to_revision is None
    assert result.event.reason == "too broad for now"
    assert db.query(RevisionApplicationEvent).count() == 1
