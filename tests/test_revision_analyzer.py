from app.hypothesis.revision_analyzer import RevisionAnalysisService
from app.models import RevisionAnalysisRecord, RevisionProposal
from tests.test_revision_aggregation import _add_refine_signal, _seed_hypothesis


class FakeAnalysisLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        return self.response


def _eligible_group(db):
    hypothesis = _seed_hypothesis(db)
    _add_refine_signal(
        db,
        hypothesis,
        interaction_id=901,
        reflection_seed="lock scope before architecture work",
    )
    _add_refine_signal(
        db,
        hypothesis,
        interaction_id=902,
        reflection_seed="align scope before architecture work",
    )
    return hypothesis


def test_phase4_consistent_group_creates_proposal_without_mutating_hypothesis(db):
    hypothesis = _eligible_group(db)
    original_signal = hypothesis.canonical_signal
    llm = FakeAnalysisLLM(
        {
            "consistent": True,
            "direction_confidence": 0.91,
            "reason": "Both independent signals request explicit scope alignment.",
            "proposed_after_data": "align scope before architecture work",
        }
    )
    service = RevisionAnalysisService(db, llm, min_independent_signals=2)

    results = service.analyze_pending(max_analyses=2)

    assert len(results) == 1
    assert results[0].analysis.consistent is True
    assert results[0].proposal is not None
    assert results[0].proposal.status == "proposed"
    assert results[0].proposal.independent_signal_count == 2
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 1
    assert hypothesis.canonical_signal == original_signal


def test_phase4_inconsistent_group_records_analysis_without_proposal(db):
    _eligible_group(db)
    llm = FakeAnalysisLLM(
        {
            "consistent": False,
            "direction_confidence": 0.42,
            "reason": "The signals do not identify one coherent change.",
            "proposed_after_data": None,
        }
    )
    service = RevisionAnalysisService(db, llm, min_independent_signals=2)

    results = service.analyze_pending(max_analyses=2)

    assert len(results) == 1
    assert results[0].proposal is None
    assert db.query(RevisionAnalysisRecord).count() == 1
    assert db.query(RevisionProposal).count() == 0


def test_phase4_unchanged_candidate_is_not_analyzed_twice(db):
    _eligible_group(db)
    llm = FakeAnalysisLLM(
        {
            "consistent": True,
            "direction_confidence": 0.9,
            "reason": "same direction",
            "proposed_after_data": "align scope",
        }
    )
    service = RevisionAnalysisService(db, llm, min_independent_signals=2)

    assert len(service.analyze_pending(max_analyses=2)) == 1
    assert service.analyze_pending(max_analyses=2) == []
    assert llm.calls == 1


def test_phase4_ineligible_group_does_not_call_llm(db):
    hypothesis = _seed_hypothesis(db)
    _add_refine_signal(
        db,
        hypothesis,
        interaction_id=903,
        reflection_seed="only one signal",
    )
    llm = FakeAnalysisLLM({})
    service = RevisionAnalysisService(db, llm, min_independent_signals=2)

    assert service.analyze_pending(max_analyses=2) == []
    assert llm.calls == 0
