from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.canonical import build_hypothesis_title
from app.models import (
    Experience,
    HypothesisEvidence,
    ReflectionRecord,
    StrategyHypothesis,
)


ACTIVE_HYPOTHESIS_STATUSES = {"observing", "ready_for_review", "needs_revision"}


class HypothesisService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, hypothesis_id: int) -> StrategyHypothesis | None:
        return self.db.get(StrategyHypothesis, hypothesis_id)

    def list_all(self, *, include_inactive: bool = True) -> list[StrategyHypothesis]:
        stmt = select(StrategyHypothesis)
        if not include_inactive:
            stmt = stmt.where(
                StrategyHypothesis.status.in_(ACTIVE_HYPOTHESIS_STATUSES)
            )
        stmt = stmt.order_by(StrategyHypothesis.updated_at.desc())
        return list(self.db.scalars(stmt).all())

    def create_from_reflection(
        self,
        *,
        reflection: ReflectionRecord,
        canonical_signal: str,
    ) -> StrategyHypothesis:
        hypothesis = StrategyHypothesis(
            title=build_hypothesis_title(reflection.insight, reflection.strategy_signal),
            canonical_signal=canonical_signal,
            insight=reflection.insight,
            proposed_change=reflection.strategy_signal,
            expected_effect=reflection.expected_effect,
            applicable_context=reflection.applicable_context or {},
            confidence=0.5,
            status="observing",
            revision=1,
        )
        self.db.add(hypothesis)
        self.db.commit()
        self.db.refresh(hypothesis)

        self.add_evidence(
            hypothesis=hypothesis,
            reflection=reflection,
            relation_type="SUPPORTS",
            relation_confidence=reflection.confidence,
            relation_reason="创建 Hypothesis 的首条 Reflection 证据",
        )
        return hypothesis

    def _interaction_id_for_reflection(self, reflection_id: int) -> int | None:
        stmt = (
            select(Experience.interaction_id)
            .join(ReflectionRecord, ReflectionRecord.experience_id == Experience.id)
            .where(ReflectionRecord.id == reflection_id)
        )
        return self.db.scalar(stmt)

    def _existing_interaction_ids(self, hypothesis_id: int) -> set[int]:
        stmt = (
            select(Experience.interaction_id)
            .join(ReflectionRecord, ReflectionRecord.experience_id == Experience.id)
            .join(
                HypothesisEvidence,
                HypothesisEvidence.reflection_id == ReflectionRecord.id,
            )
            .where(HypothesisEvidence.hypothesis_id == hypothesis_id)
        )
        return {int(item) for item in self.db.scalars(stmt).all() if item is not None}

    def add_evidence(
        self,
        *,
        hypothesis: StrategyHypothesis,
        reflection: ReflectionRecord,
        relation_type: str,
        relation_confidence: float,
        relation_reason: str = "",
    ) -> HypothesisEvidence:
        existing = self.db.scalar(
            select(HypothesisEvidence).where(
                HypothesisEvidence.hypothesis_id == hypothesis.id,
                HypothesisEvidence.reflection_id == reflection.id,
            )
        )
        if existing is not None:
            return existing

        interaction_id = self._interaction_id_for_reflection(reflection.id)
        previous_interactions = self._existing_interaction_ids(hypothesis.id)
        is_independent = interaction_id is not None and interaction_id not in previous_interactions

        evidence = HypothesisEvidence(
            hypothesis_id=hypothesis.id,
            reflection_id=reflection.id,
            relation_type=relation_type,
            relation_confidence=max(0.0, min(1.0, float(relation_confidence))),
            relation_reason=relation_reason,
            is_independent=is_independent,
        )
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)
        self.recalculate(hypothesis.id)
        return evidence

    @staticmethod
    def _context_signature(context: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(context.get("task_type", "unknown")),
            str(context.get("complexity", "unknown")),
            str(context.get("interaction_mode", "unknown")),
            str(context.get("project_stage", "unknown")),
        )

    def _evidence_rows(
        self, hypothesis_id: int
    ) -> list[tuple[HypothesisEvidence, ReflectionRecord, Experience]]:
        stmt = (
            select(HypothesisEvidence, ReflectionRecord, Experience)
            .join(
                ReflectionRecord,
                ReflectionRecord.id == HypothesisEvidence.reflection_id,
            )
            .join(Experience, Experience.id == ReflectionRecord.experience_id)
            .where(HypothesisEvidence.hypothesis_id == hypothesis_id)
        )
        return list(self.db.execute(stmt).all())

    def recalculate(self, hypothesis_id: int) -> StrategyHypothesis:
        hypothesis = self.db.get(StrategyHypothesis, hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"Hypothesis #{hypothesis_id} 不存在")

        rows = self._evidence_rows(hypothesis_id)
        supporting = [row for row in rows if row[0].relation_type == "SUPPORTS"]
        refinements = [row for row in rows if row[0].relation_type == "REFINES"]
        conflicts = [row for row in rows if row[0].relation_type == "CONFLICTS"]

        independent_count = sum(1 for evidence, _, _ in rows if evidence.is_independent)
        independent_support_count = sum(
            1 for evidence, _, _ in supporting if evidence.is_independent
        )

        # ready_for_review 只能由“直接支持当前表述”的证据推动。
        # REFINES 说明当前 Hypothesis 的边界/表述仍需修订，因此不能扩大
        # support diversity，也不能把当前版本推向成熟。
        support_signatures = {
            self._context_signature(experience.context_data or {})
            for _, _, experience in supporting
        }
        context_diversity = len(support_signatures)

        positive_strength = sum(
            evidence.relation_confidence * reflection.confidence
            for evidence, reflection, _ in supporting
        )
        negative_strength = sum(
            evidence.relation_confidence * reflection.confidence
            for evidence, reflection, _ in conflicts
        )

        confidence = (
            0.48
            + min(0.30, positive_strength * 0.10)
            + min(0.15, max(0, independent_support_count - 1) * 0.05)
            + min(0.10, max(0, context_diversity - 1) * 0.04)
            - min(0.45, negative_strength * 0.12)
        )
        confidence = max(0.0, min(1.0, confidence))

        hypothesis.supporting_count = len(supporting)
        hypothesis.refinement_count = len(refinements)
        hypothesis.counter_count = len(conflicts)
        hypothesis.independent_evidence_count = independent_count
        hypothesis.independent_support_count = independent_support_count
        hypothesis.context_diversity = context_diversity
        hypothesis.confidence = round(confidence, 4)
        hypothesis.last_evidence_at = datetime.now()

        if hypothesis.status in ACTIVE_HYPOTHESIS_STATUSES:
            if hypothesis.refinement_count > 0:
                hypothesis.status = "needs_revision"
            elif (
                hypothesis.confidence >= 0.8
                and hypothesis.independent_support_count >= 3
                and hypothesis.context_diversity >= 2
                and hypothesis.counter_count == 0
            ):
                hypothesis.status = "ready_for_review"
            else:
                hypothesis.status = "observing"

        self.db.commit()
        self.db.refresh(hypothesis)
        return hypothesis

    def recalculate_all(self) -> list[StrategyHypothesis]:
        """按现有 Evidence 重新计算所有活动 Hypothesis。

        用于规则修正后纠正旧数据库里把 REFINES 计入 support 的统计。
        """
        hypotheses = self.list_all(include_inactive=False)
        return [self.recalculate(hypothesis.id) for hypothesis in hypotheses]
