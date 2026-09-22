from sqlalchemy import select

from app.hypothesis.revision_aggregation import RevisionAggregationService
from app.hypothesis.revision_commit import RevisionCommitService
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.models import HypothesisRevision, RevisionSignal
from tests.test_candidate_revision import _add_refine_signal
from tests.test_revision_commit import _evaluated_low_candidate


def _normalize_outcome_source_signals(db, candidate):
    ids = [int(item["signal_id"]) for item in candidate.evaluation_data["source_signal_assessments"]]
    for signal_id in ids:
        signal = db.get(RevisionSignal, signal_id)
        signal.dimension = "OUTCOME"
        signal.target_field = "scope_drift"
    db.commit()
    return ids


def test_phase5_4_commit_projects_signal_resolution_and_lineage(db):
    hypothesis, _proposal, candidate = _evaluated_low_candidate(db)
    source_signal_ids = _normalize_outcome_source_signals(db, candidate)

    service = RevisionCommitService(db)
    result = service.auto_apply_ready(max_candidates=1)[0]

    revision = db.scalar(
        select(HypothesisRevision).where(
            HypothesisRevision.hypothesis_id == hypothesis.id,
            HypothesisRevision.revision_number == 2,
        )
    )
    assert revision.parent_revision == 1

    rows = list(
        db.scalars(select(RevisionSignal).where(RevisionSignal.id.in_(source_signal_ids))).all()
    )
    assert len(rows) == len(source_signal_ids)
    assert all(item.resolution_status == "resolved" for item in rows)
    assert all(item.resolved_by_candidate_id == candidate.id for item in rows)
    assert all(item.resolved_by_revision == 2 for item in rows)

    pressure = RevisionReconciliationService(db).pressure_summary(hypothesis.id)
    assert pressure.lineage == [2, 1]
    assert pressure.resolved == len(source_signal_ids)
    assert pressure.active_open == 0
    assert result.event.event_data["resolved_signal_ids"] == sorted(source_signal_ids)


def test_phase5_4_ancestor_open_requires_revalidation_not_direct_revision(db):
    hypothesis, _proposal, candidate = _evaluated_low_candidate(db)
    _normalize_outcome_source_signals(db, candidate)
    extra = _add_refine_signal(
        db,
        hypothesis,
        interaction_id=2001,
        reflection_seed="ancestor-open",
        target_field="boundary_handling",
    )

    RevisionCommitService(db).auto_apply_ready(max_candidates=1)
    db.refresh(hypothesis)

    pressure = RevisionReconciliationService(db).pressure_summary(hypothesis.id)
    assert pressure.active_open == 0
    assert extra.id in pressure.ancestor_open_ids
    assert hypothesis.status == "needs_revalidation"

    # Old r1 signals must not be fed directly to the r2 analyzer.
    groups = RevisionAggregationService(db, min_independent_signals=1).group_signals(
        hypothesis_id=hypothesis.id
    )
    assert groups == []


def test_phase5_4_rollback_reopens_signal_when_resolving_revision_is_not_in_lineage(db):
    hypothesis, _proposal, candidate = _evaluated_low_candidate(db)
    source_signal_ids = _normalize_outcome_source_signals(db, candidate)
    commit = RevisionCommitService(db)
    commit.auto_apply_ready(max_candidates=1)

    pressure_r2 = RevisionReconciliationService(db).pressure_summary(hypothesis.id)
    assert set(source_signal_ids).issubset(set(pressure_r2.resolved_ids))

    commit.rollback(hypothesis.id, 1, reason="lineage test")
    db.refresh(hypothesis)
    pressure_r1 = RevisionReconciliationService(db).pressure_summary(hypothesis.id)

    assert pressure_r1.lineage == [1]
    assert set(source_signal_ids).issubset(set(pressure_r1.active_open_ids))
    assert hypothesis.status == "needs_revision"


def test_phase5_4_new_revision_after_rollback_records_true_parent(db):
    hypothesis, _proposal, candidate = _evaluated_low_candidate(db)
    _normalize_outcome_source_signals(db, candidate)
    commit = RevisionCommitService(db)
    commit.auto_apply_ready(max_candidates=1)
    commit.rollback(hypothesis.id, 1)

    # Reuse the Phase 5.3 helper to create a reviewed candidate on current r1.
    from tests.test_revision_commit import _manual_candidate_on_base

    _proposal2, candidate2 = _manual_candidate_on_base(
        db, hypothesis, base_revision=1, suffix="branch-parent"
    )
    result = commit.approve_candidate(candidate2.id)
    assert result.event.to_revision == 3

    r3 = db.scalar(
        select(HypothesisRevision).where(
            HypothesisRevision.hypothesis_id == hypothesis.id,
            HypothesisRevision.revision_number == 3,
        )
    )
    assert r3.parent_revision == 1
    assert RevisionReconciliationService(db).lineage(hypothesis.id, 3) == [3, 1]
