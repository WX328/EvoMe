from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.service import HypothesisService
from app.models import (
    CandidateRevision,
    Experience,
    HypothesisEvidence,
    InteractionRecord,
    LearningInvalidationEvent,
    PersonalMemory,
    ReflectionRecord,
    RevisionProposal,
    RevisionSignal,
)


class LearningInvalidationService:
    """Invalidate learning derived from a misclassified interaction without deleting audit history.

    The source InteractionRecord remains append-only. Derived learning artifacts are marked inactive/
    invalid, current Hypothesis statistics are recalculated, and any not-yet-applied Proposal/Candidate
    that depends on invalidated signals is retired. Already-applied revisions are never auto-rolled back;
    they are reported for explicit human review.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def list_events(self, *, limit: int = 50) -> list[LearningInvalidationEvent]:
        stmt = (
            select(LearningInvalidationEvent)
            .order_by(LearningInvalidationEvent.created_at.desc(), LearningInvalidationEvent.id.desc())
            .limit(max(1, int(limit)))
        )
        return list(self.db.scalars(stmt).all())

    def invalidate_interaction(
        self,
        interaction_id: int,
        *,
        reason: str = "operator_marked_non_learning",
    ) -> LearningInvalidationEvent:
        interaction = self.db.get(InteractionRecord, int(interaction_id))
        if interaction is None:
            raise ValueError(f"Interaction #{interaction_id} 不存在")

        # Idempotent operator UX: do not create repeated invalidation events for
        # an interaction that has already been explicitly invalidated.
        existing = self.db.scalar(
            select(LearningInvalidationEvent)
            .where(LearningInvalidationEvent.interaction_id == int(interaction_id))
            .order_by(LearningInvalidationEvent.id.desc())
        )
        metadata = dict(interaction.record_metadata or {})
        if metadata.get("learning_invalidated") and existing is not None:
            return existing

        experiences = list(
            self.db.scalars(
                select(Experience).where(Experience.interaction_id == int(interaction_id))
            ).all()
        )
        experience_ids = [int(item.id) for item in experiences]

        reflections: list[ReflectionRecord] = []
        if experience_ids:
            reflections = list(
                self.db.scalars(
                    select(ReflectionRecord).where(
                        ReflectionRecord.experience_id.in_(experience_ids)
                    )
                ).all()
            )
        reflection_ids = [int(item.id) for item in reflections]

        signals = list(
            self.db.scalars(
                select(RevisionSignal).where(
                    (RevisionSignal.interaction_id == int(interaction_id))
                    | (
                        RevisionSignal.reflection_id.in_(reflection_ids)
                        if reflection_ids
                        else False
                    )
                )
            ).all()
        )
        signal_ids = sorted({int(item.id) for item in signals})
        hypothesis_ids = sorted({int(item.hypothesis_id) for item in signals})

        # Include hypotheses that have evidence from an invalidated reflection,
        # even when a legacy REFINE relation never produced a valid atomic signal.
        if reflection_ids:
            evidence_hids = self.db.scalars(
                select(HypothesisEvidence.hypothesis_id).where(
                    HypothesisEvidence.reflection_id.in_(reflection_ids)
                )
            ).all()
            hypothesis_ids = sorted(
                set(hypothesis_ids) | {int(item) for item in evidence_hids if item is not None}
            )

        invalidated_memory_ids: list[int] = []
        for memory in self.db.scalars(
            select(PersonalMemory).where(
                PersonalMemory.source_interaction_id == int(interaction_id)
            )
        ).all():
            if memory.status == "active":
                memory.status = "invalid"
            data = dict(memory.memory_data or {})
            data["learning_invalidation"] = {"reason": reason}
            memory.memory_data = data
            invalidated_memory_ids.append(int(memory.id))

        for experience in experiences:
            data = dict(experience.experience_metadata or {})
            data["learning_invalidated"] = True
            data["learning_invalidation_reason"] = reason
            experience.experience_metadata = data

        for reflection in reflections:
            reflection.status = "invalid"

        for signal in signals:
            signal.resolution_status = "invalid"
            signal.resolution_note = f"learning_invalidated: {reason}"

        invalidated_proposal_ids: list[int] = []
        stale_candidate_ids: list[int] = []
        applied_candidate_ids: list[int] = []
        if signal_ids:
            proposals = list(
                self.db.scalars(select(RevisionProposal)).all()
            )
            affected_proposals = [
                proposal
                for proposal in proposals
                if set(int(x) for x in (proposal.source_signal_ids or []) if str(x).isdigit())
                & set(signal_ids)
            ]
            for proposal in affected_proposals:
                if proposal.status not in {"applied", "rejected"}:
                    proposal.status = "invalidated"
                invalidated_proposal_ids.append(int(proposal.id))

            if invalidated_proposal_ids:
                candidates = list(
                    self.db.scalars(
                        select(CandidateRevision).where(
                            CandidateRevision.source_proposal_id.in_(invalidated_proposal_ids)
                        )
                    ).all()
                )
                for candidate in candidates:
                    data = dict(candidate.evaluation_data or {})
                    data["source_invalidated"] = True
                    data["source_invalidation_reason"] = reason
                    if candidate.status == "applied":
                        data["requires_manual_revision_review"] = True
                        applied_candidate_ids.append(int(candidate.id))
                    else:
                        candidate.status = "stale"
                        data["stale_reason"] = f"source_learning_invalidated_interaction_{interaction_id}"
                        stale_candidate_ids.append(int(candidate.id))
                    candidate.evaluation_data = data

        # Flush invalid statuses before recalculating current-state statistics.
        self.db.flush()
        hypothesis_service = HypothesisService(self.db)
        for hypothesis_id in hypothesis_ids:
            try:
                hypothesis_service.recalculate(hypothesis_id, commit=False)
            except ValueError:
                continue

        event = LearningInvalidationEvent(
            interaction_id=int(interaction_id),
            reason=str(reason or "operator_marked_non_learning"),
            event_data={
                "experience_ids": experience_ids,
                "reflection_ids": reflection_ids,
                "revision_signal_ids": signal_ids,
                "personal_memory_ids": sorted(invalidated_memory_ids),
                "invalidated_proposal_ids": sorted(set(invalidated_proposal_ids)),
                "stale_candidate_ids": sorted(set(stale_candidate_ids)),
                "applied_candidate_ids_requiring_manual_review": sorted(set(applied_candidate_ids)),
                "recalculated_hypothesis_ids": hypothesis_ids,
            },
            created_at=datetime.now(),
        )
        self.db.add(event)
        self.db.flush()

        metadata["learning_invalidated"] = True
        metadata["learning_invalidation_reason"] = event.reason
        metadata["learning_invalidation_event_id"] = int(event.id)
        interaction.record_metadata = metadata

        self.db.commit()
        self.db.refresh(event)
        return event
