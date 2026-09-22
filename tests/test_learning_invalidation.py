from app.audit.service import AuditService
from app.evolution.service import EvolutionMemoryService
from app.hypothesis.service import HypothesisService
from app.learning.invalidation import LearningInvalidationService
from app.memory.service import PersonalMemoryService
from app.models import RevisionSignal


def test_invalidate_interaction_retires_derived_learning_without_deleting_audit(db):
    audit = AuditService(db)
    interaction = audit.save_interaction(
        session_id="pollution",
        user_input="revision-signals",
        assistant_response="mistaken natural-language answer",
        strategy_name="default_conversation_strategy_v2",
        record_metadata={
            "learning_eligibility": {
                "category": "natural",
                "allow_personal_memory": True,
                "allow_evolution": True,
            }
        },
    )
    memory = PersonalMemoryService(db).save(
        memory_type="preference",
        content="polluted preference",
        source_interaction_id=interaction.id,
    )
    evolution = EvolutionMemoryService(db)
    experience = evolution.save_experience(
        interaction_id=interaction.id,
        task_type="architecture_design",
        strategy_name="default_conversation_strategy",
        strategy_version=2,
        skills_used=[],
        result_summary="polluted experience",
        success=True,
        score=0.9,
        context_data={"task_type": "architecture_design", "complexity": "high"},
    )
    reflection = evolution.save_reflection(
        experience_id=experience.id,
        reflection_type="success",
        observation="polluted",
        insight="polluted insight",
        applicable_context={"task_type": "architecture_design", "complexity": "high"},
        expected_effect="polluted",
        canonical_signal="polluted signal",
        strategy_signal="polluted change",
        confidence=0.9,
    )
    hypothesis_service = HypothesisService(db)
    hypothesis = hypothesis_service.create_from_reflection(
        reflection=reflection,
        canonical_signal=reflection.canonical_signal,
    )
    evidence = hypothesis_service.add_evidence(
        hypothesis=hypothesis,
        reflection=reflection,
        relation_type="REFINES",
        relation_confidence=0.9,
        relation_reason="polluted refine",
    )
    signal = RevisionSignal(
        hypothesis_id=hypothesis.id,
        base_revision=hypothesis.current_revision,
        reflection_id=reflection.id,
        evidence_id=evidence.id,
        interaction_id=interaction.id,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        before_data="old",
        observed_data="polluted structure",
        canonical_change="polluted expansion",
        confidence=0.9,
    )
    db.add(signal)
    db.commit()

    event = LearningInvalidationService(db).invalidate_interaction(
        interaction.id,
        reason="operational_input_misclassified",
    )

    db.refresh(interaction)
    db.refresh(memory)
    db.refresh(reflection)
    db.refresh(signal)
    updated = hypothesis_service.get(hypothesis.id)

    assert interaction.record_metadata["learning_invalidated"] is True
    assert memory.status == "invalid"
    assert reflection.status == "invalid"
    assert signal.resolution_status == "invalid"
    assert signal.id in event.event_data["revision_signal_ids"]
    assert audit.get(interaction.id) is not None
    # Inactive reflection evidence remains in the audit graph but no longer
    # contributes to current hypothesis statistics.
    assert updated.supporting_count == 0
    assert updated.refinement_count == 0


def test_learning_invalidation_is_idempotent(db):
    interaction = AuditService(db).save_interaction(
        session_id="idempotent",
        user_input="hypothesis",
        assistant_response="mistaken answer",
        strategy_name="default_conversation_strategy_v2",
    )
    service = LearningInvalidationService(db)
    first = service.invalidate_interaction(interaction.id, reason="operator mistake")
    second = service.invalidate_interaction(interaction.id, reason="operator mistake again")
    assert first.id == second.id


def test_invalidation_stales_unapplied_candidate_based_on_polluted_signal(db):
    from app.models import CandidateRevision, RevisionAnalysisRecord, RevisionProposal

    interaction = AuditService(db).save_interaction(
        session_id="candidate-pollution",
        user_input="revision-signals",
        assistant_response="mistaken answer",
        strategy_name="default_conversation_strategy_v2",
    )
    evolution = EvolutionMemoryService(db)
    experience = evolution.save_experience(
        interaction_id=interaction.id,
        task_type="architecture_design",
        strategy_name="default_conversation_strategy",
        strategy_version=2,
        skills_used=[],
        result_summary="polluted",
        success=True,
        score=0.9,
        context_data={"task_type": "architecture_design"},
    )
    reflection = evolution.save_reflection(
        experience_id=experience.id,
        reflection_type="success",
        observation="polluted",
        insight="polluted",
        applicable_context={"task_type": "architecture_design"},
        expected_effect="polluted",
        canonical_signal="polluted-canonical",
        strategy_signal="polluted-action",
        confidence=0.9,
    )
    hs = HypothesisService(db)
    hypothesis = hs.create_from_reflection(reflection=reflection, canonical_signal=reflection.canonical_signal)
    evidence = hs.add_evidence(
        hypothesis=hypothesis,
        reflection=reflection,
        relation_type="REFINES",
        relation_confidence=0.9,
        relation_reason="polluted",
    )
    signal = RevisionSignal(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        reflection_id=reflection.id,
        evidence_id=evidence.id,
        interaction_id=interaction.id,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        before_data="old",
        observed_data="bad",
        canonical_change="bad expansion",
        confidence=0.9,
    )
    db.add(signal)
    db.flush()
    analysis = RevisionAnalysisRecord(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        signal_ids=[signal.id],
        candidate_fingerprint="polluted-fingerprint",
        consistent=True,
        direction_confidence=0.9,
        reason="polluted",
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
        before_data="old",
        proposed_after_data="bad",
        reason="polluted",
        source_signal_ids=[signal.id],
        independent_signal_count=1,
        direction_confidence=0.9,
        status="proposed",
    )
    db.add(proposal)
    db.flush()
    candidate = CandidateRevision(
        hypothesis_id=hypothesis.id,
        base_revision=1,
        source_proposal_id=proposal.id,
        title="polluted",
        insight="polluted",
        when_data={},
        do_data={"text": "bad"},
        expect_data={},
        canonical_signal="bad",
        status="ready_to_apply",
        evaluation_version=3,
        evaluation_data={"evaluation_valid": True},
    )
    db.add(candidate)
    db.commit()

    event = LearningInvalidationService(db).invalidate_interaction(
        interaction.id, reason="operational_input_misclassified"
    )
    db.refresh(proposal)
    db.refresh(candidate)

    assert proposal.status == "invalidated"
    assert candidate.status == "stale"
    assert candidate.evaluation_data["source_invalidated"] is True
    assert candidate.id in event.event_data["stale_candidate_ids"]
