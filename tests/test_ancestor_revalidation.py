import json

from sqlalchemy import select

from app.hypothesis.ancestor_revalidation import AncestorSignalRevalidationService
from app.hypothesis.revision_commit import RevisionCommitService
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.models import RevisionSignal, RevisionSignalRevalidation
from tests.test_revision_aggregation import _add_refine_signal
from tests.test_revision_commit import _evaluated_low_candidate
from tests.test_revision_reconciliation import _normalize_outcome_source_signals


class FakeRevalidationLLM:
    def __init__(self, outcome="STILL_APPLIES"):
        self.outcome = outcome
        self.calls = 0

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        return {
            "assessments": [
                {
                    "signal_id": item["signal_id"],
                    "outcome": self.outcome,
                    "confidence": 0.92,
                    "reason": "test revalidation",
                }
                for item in payload["ancestor_signals"]
            ]
        }


def _r2_with_one_ancestor_signal(db):
    hypothesis, _proposal, low_candidate = _evaluated_low_candidate(db)
    _normalize_outcome_source_signals(db, low_candidate)
    ancestor = _add_refine_signal(
        db,
        hypothesis,
        interaction_id=9001,
        reflection_seed="response structure still needed",
        target_field="response_structure",
    )
    RevisionCommitService(db).auto_apply_ready(max_candidates=1)
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 2
    assert ancestor.id in RevisionReconciliationService(db).pressure_summary(hypothesis.id).ancestor_open_ids
    return hypothesis, ancestor


def test_phase5_5_still_applies_materializes_current_signal_and_supersedes_ancestor_on_branch(db):
    hypothesis, ancestor = _r2_with_one_ancestor_signal(db)
    llm = FakeRevalidationLLM("STILL_APPLIES")
    service = AncestorSignalRevalidationService(db, llm)

    result = service.revalidate_hypothesis(hypothesis.id)

    assert result.llm_called is True
    assert llm.calls == 1
    assert len(result.record_ids) == 1
    assert len(result.rebased_signal_ids) == 1

    db.refresh(ancestor)
    rebased = db.get(RevisionSignal, result.rebased_signal_ids[0])
    assert rebased.base_revision == 2
    assert rebased.target_field == ancestor.target_field
    assert rebased.provenance_data["rebased_from_signal_id"] == ancestor.id
    assert ancestor.resolution_status == "superseded"
    assert ancestor.superseded_by_signal_id == rebased.id
    assert ancestor.superseded_by_revision == 2

    record = db.get(RevisionSignalRevalidation, result.record_ids[0])
    assert record.outcome == "STILL_APPLIES"
    assert record.rebased_signal_id == rebased.id

    pressure = RevisionReconciliationService(db).pressure_summary(hypothesis.id)
    assert pressure.ancestor_open == 0
    assert pressure.active_open == 1
    assert pressure.superseded == 1
    assert rebased.id in pressure.active_open_ids

    # Same source/target is idempotent and spends no second LLM call.
    second = service.revalidate_hypothesis(hypothesis.id)
    assert second.llm_called is False
    assert llm.calls == 1


def test_phase5_5_branch_supersession_reopens_original_after_rollback(db):
    hypothesis, ancestor = _r2_with_one_ancestor_signal(db)
    service = AncestorSignalRevalidationService(db, FakeRevalidationLLM("STILL_APPLIES"))
    result = service.revalidate_hypothesis(hypothesis.id)
    rebased_id = result.rebased_signal_ids[0]

    RevisionCommitService(db).rollback(hypothesis.id, 1, reason="test rollback")
    db.refresh(hypothesis)
    pressure = RevisionReconciliationService(db).pressure_summary(hypothesis.id)

    assert ancestor.id in pressure.active_open_ids
    assert rebased_id in pressure.inactive_branch_open_ids
    assert ancestor.id not in pressure.resolved_ids


def test_phase5_5_resolved_by_current_retires_ancestor_without_copy(db):
    hypothesis, ancestor = _r2_with_one_ancestor_signal(db)
    service = AncestorSignalRevalidationService(db, FakeRevalidationLLM("RESOLVED_BY_CURRENT"))

    result = service.revalidate_hypothesis(hypothesis.id)

    assert result.rebased_signal_ids == []
    assert result.resolved_signal_ids == [ancestor.id]
    db.refresh(ancestor)
    assert ancestor.resolution_status == "resolved"
    assert ancestor.resolved_by_revision == 2
    pressure = RevisionReconciliationService(db).pressure_summary(hypothesis.id)
    assert ancestor.id in pressure.resolved_ids
    assert pressure.ancestor_open == 0
