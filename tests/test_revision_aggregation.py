from app.hypothesis.revision_aggregation import RevisionAggregationService
from app.hypothesis.signal_quality import normalize_revision_signal
from app.hypothesis.service import HypothesisService
from app.models import Experience, HypothesisEvidence, ReflectionRecord, RevisionSignal, StrategyHypothesis


def _seed_hypothesis(db):
    hypothesis = StrategyHypothesis(
        title="architecture boundary analysis",
        canonical_signal="architecture_design high collaborative -> boundary analysis",
        insight="boundary analysis helps architecture design",
        proposed_change="analyze responsibilities and boundaries",
        expected_effect="improve clarity",
        applicable_context={
            "task_type": "architecture_design",
            "complexity": "high",
            "interaction_mode": "collaborative",
        },
        confidence=0.5,
        status="needs_revision",
        revision=1,
        current_revision=1,
    )
    db.add(hypothesis)
    db.flush()
    HypothesisService(db)._snapshot_current_hypothesis(
        hypothesis,
        created_reason="test",
        commit=False,
    )
    db.commit()
    db.refresh(hypothesis)
    return hypothesis


def _add_refine_signal(db, hypothesis, *, interaction_id, reflection_seed, operation="EXPAND", target_field="scope_alignment", confidence=0.9):
    experience = Experience(
        interaction_id=interaction_id,
        task_type="architecture_design",
        strategy_name="default",
        strategy_version=2,
        skills_used=[],
        result_summary=reflection_seed,
        success=True,
        score=1.0,
        context_data={
            "task_type": "architecture_design",
            "complexity": "medium",
            "interaction_mode": "collaborative",
            "scope_bounded": True,
        },
    )
    db.add(experience)
    db.flush()
    reflection = ReflectionRecord(
        experience_id=experience.id,
        reflection_type="success",
        observation=reflection_seed,
        lesson=reflection_seed,
        insight=reflection_seed,
        applicable_context={
            "task_type": "architecture_design",
            "complexity": "medium",
            "interaction_mode": "collaborative",
            "scope_bounded": True,
        },
        strategy_signal=reflection_seed,
        skill_signal="",
        expected_effect="improve clarity",
        canonical_signal=reflection_seed,
        confidence=0.9,
        status="active",
    )
    db.add(reflection)
    db.flush()
    evidence = HypothesisEvidence(
        hypothesis_id=hypothesis.id,
        hypothesis_revision=1,
        reflection_id=reflection.id,
        relation_type="REFINES",
        relation_confidence=0.9,
        relation_reason="test refine",
        is_independent=True,
    )
    db.add(evidence)
    db.flush()
    signal = RevisionSignal(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        reflection_id=reflection.id,
        evidence_id=evidence.id,
        interaction_id=interaction_id,
        dimension="ACTION",
        operation=operation,
        target_field=target_field,
        before_data="existing",
        observed_data=reflection_seed,
        canonical_change=f"{target_field}: expand",
        confidence=confidence,
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal


def test_phase3_operation_dimension_compatibility_matrix():
    invalid, reason = normalize_revision_signal({
        "dimension": "ACTION",
        "operation": "ADD_CONDITION",
        "target_field": "scope_alignment",
        "before_data": [],
        "observed_data": "lock scope",
        "canonical_change": "add scope lock",
        "confidence": 0.9,
    })
    assert invalid is None
    assert "ACTION 不允许 operation=ADD_CONDITION" in reason

    valid, reason = normalize_revision_signal({
        "dimension": "ACTION",
        "operation": "EXPAND",
        "target_field": "scope_alignment",
        "before_data": [],
        "observed_data": "lock scope",
        "canonical_change": "expand action with scope lock",
        "confidence": 0.9,
    })
    assert reason == "valid"
    assert valid["operation"] == "EXPAND"


def test_phase3_groups_signals_deterministically_and_counts_independent_interactions(db):
    hypothesis = _seed_hypothesis(db)
    _add_refine_signal(db, hypothesis, interaction_id=501, reflection_seed="lock scope first", confidence=0.9)
    _add_refine_signal(db, hypothesis, interaction_id=502, reflection_seed="explicit scope alignment", confidence=0.8)

    service = RevisionAggregationService(db, min_independent_signals=2)
    groups = service.group_signals(hypothesis_id=hypothesis.id, base_revision=1)

    assert len(groups) == 1
    group = groups[0]
    assert group.dimension == "ACTION"
    assert group.operation == "EXPAND"
    assert group.target_field == "scope_alignment"
    assert group.signal_count == 2
    assert group.independent_count == 2
    assert group.average_confidence == 0.85
    assert group.eligible_for_analysis is True


def test_phase3_same_interaction_does_not_inflate_independent_count(db):
    hypothesis = _seed_hypothesis(db)
    _add_refine_signal(db, hypothesis, interaction_id=601, reflection_seed="first", target_field="scope_alignment")
    _add_refine_signal(db, hypothesis, interaction_id=601, reflection_seed="second", target_field="response_structure")

    service = RevisionAggregationService(db, min_independent_signals=2)
    groups = service.group_signals(hypothesis_id=hypothesis.id, base_revision=1)

    assert len(groups) == 2
    assert all(group.independent_count == 1 for group in groups)
    assert all(group.eligible_for_analysis is False for group in groups)


def test_phase3_invalid_legacy_signals_are_excluded_from_groups(db):
    hypothesis = _seed_hypothesis(db)
    signal = _add_refine_signal(
        db,
        hypothesis,
        interaction_id=701,
        reflection_seed="legacy action",
        operation="ADD_CONDITION",
    )
    assert signal.id is not None

    service = RevisionAggregationService(db, min_independent_signals=1)
    assert service.group_signals(hypothesis_id=hypothesis.id, base_revision=1) == []


def test_phase3_evidence_summary_is_derived_and_compact(db):
    hypothesis = _seed_hypothesis(db)
    _add_refine_signal(db, hypothesis, interaction_id=801, reflection_seed="scope lock a")
    _add_refine_signal(db, hypothesis, interaction_id=802, reflection_seed="scope lock b")

    service = RevisionAggregationService(db, min_independent_signals=2)
    summary = service.build_evidence_summary(hypothesis_id=hypothesis.id, revision_number=1)

    assert summary["hypothesis_id"] == hypothesis.id
    assert summary["revision"] == 1
    assert summary["evidence_counts"]["refines"] == 2
    assert summary["context_coverage"]["complexity"] == ["medium"]
    assert summary["context_coverage"]["scope_bounded"] == [True]
    assert len(summary["revision_groups"]) == 1
    assert len(summary["analysis_candidates"]) == 1
    candidate = summary["analysis_candidates"][0]
    assert candidate["target_field"] == "scope_alignment"
    assert candidate["independent_count"] == 2
