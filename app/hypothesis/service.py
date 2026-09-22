from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.canonical import build_hypothesis_title
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.models import (
    Experience,
    HypothesisEvidence,
    HypothesisRevision,
    ReflectionRecord,
    StrategyHypothesis,
)


ACTIVE_HYPOTHESIS_STATUSES = {"observing", "ready_for_review", "needs_revision", "needs_revalidation"}


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

    def get_revision(
        self,
        hypothesis_id: int,
        revision_number: int,
    ) -> HypothesisRevision | None:
        stmt = select(HypothesisRevision).where(
            HypothesisRevision.hypothesis_id == hypothesis_id,
            HypothesisRevision.revision_number == revision_number,
        )
        return self.db.scalar(stmt)

    def list_revisions(self, hypothesis_id: int) -> list[HypothesisRevision]:
        stmt = (
            select(HypothesisRevision)
            .where(HypothesisRevision.hypothesis_id == hypothesis_id)
            .order_by(HypothesisRevision.revision_number.asc())
        )
        return list(self.db.scalars(stmt).all())

    def create_revision_snapshot(
        self,
        *,
        hypothesis_id: int,
        revision_number: int,
        title: str,
        canonical_signal: str,
        insight: str,
        when_data: dict[str, Any],
        do_data: dict[str, Any],
        expect_data: dict[str, Any],
        created_reason: str,
        parent_revision: int | None = None,
        commit: bool = True,
    ) -> HypothesisRevision:
        """创建不可变 Revision 快照。

        已存在的 (hypothesis_id, revision_number) 永远不覆盖；调用方必须创建新版本。
        """
        hypothesis = self.get(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"Hypothesis #{hypothesis_id} 不存在")

        if self.get_revision(hypothesis_id, revision_number) is not None:
            raise ValueError(
                f"Hypothesis #{hypothesis_id}:r{revision_number} 已存在；"
                "Revision 是不可变快照，不能原地覆盖"
            )

        revision = HypothesisRevision(
            hypothesis_id=hypothesis_id,
            revision_number=revision_number,
            parent_revision=parent_revision,
            title=title,
            canonical_signal=canonical_signal,
            insight=insight,
            when_data=when_data or {},
            do_data=do_data or {},
            expect_data=expect_data or {},
            created_reason=created_reason,
        )
        self.db.add(revision)
        if commit:
            self.db.commit()
            self.db.refresh(revision)
        else:
            self.db.flush()
        return revision

    def _snapshot_current_hypothesis(
        self,
        hypothesis: StrategyHypothesis,
        *,
        created_reason: str,
        commit: bool = True,
    ) -> HypothesisRevision:
        return self.create_revision_snapshot(
            hypothesis_id=hypothesis.id,
            revision_number=hypothesis.current_revision,
            title=hypothesis.title,
            canonical_signal=hypothesis.canonical_signal,
            insight=hypothesis.insight,
            when_data=hypothesis.applicable_context or {},
            do_data={"text": hypothesis.proposed_change},
            expect_data={"text": hypothesis.expected_effect},
            created_reason=created_reason,
            parent_revision=None,
            commit=commit,
        )

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
            current_revision=1,
        )
        self.db.add(hypothesis)
        self.db.flush()

        self._snapshot_current_hypothesis(
            hypothesis,
            created_reason="created_from_first_supporting_reflection",
            commit=False,
        )
        self.db.commit()
        self.db.refresh(hypothesis)

        self.add_evidence(
            hypothesis=hypothesis,
            reflection=reflection,
            relation_type="SUPPORTS",
            relation_confidence=reflection.confidence,
            relation_reason="创建 Hypothesis 的首条 Reflection 证据",
            revision_number=1,
        )
        return hypothesis

    def _interaction_id_for_reflection(self, reflection_id: int) -> int | None:
        stmt = (
            select(Experience.interaction_id)
            .join(ReflectionRecord, ReflectionRecord.experience_id == Experience.id)
            .where(ReflectionRecord.id == reflection_id)
        )
        return self.db.scalar(stmt)

    def _existing_interaction_ids(
        self,
        hypothesis_id: int,
        revision_number: int,
    ) -> set[int]:
        stmt = (
            select(Experience.interaction_id)
            .join(ReflectionRecord, ReflectionRecord.experience_id == Experience.id)
            .join(
                HypothesisEvidence,
                HypothesisEvidence.reflection_id == ReflectionRecord.id,
            )
            .where(
                HypothesisEvidence.hypothesis_id == hypothesis_id,
                HypothesisEvidence.hypothesis_revision == revision_number,
                ReflectionRecord.status == "active",
            )
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
        revision_number: int | None = None,
        recalculate_current: bool = True,
    ) -> HypothesisEvidence:
        target_revision = int(revision_number or hypothesis.current_revision or 1)
        if self.get_revision(hypothesis.id, target_revision) is None:
            raise ValueError(
                f"Hypothesis #{hypothesis.id}:r{target_revision} 不存在，"
                "不能绑定 Evidence"
            )

        existing = self.db.scalar(
            select(HypothesisEvidence).where(
                HypothesisEvidence.hypothesis_id == hypothesis.id,
                HypothesisEvidence.hypothesis_revision == target_revision,
                HypothesisEvidence.reflection_id == reflection.id,
            )
        )
        if existing is not None:
            return existing

        interaction_id = self._interaction_id_for_reflection(reflection.id)
        previous_interactions = self._existing_interaction_ids(
            hypothesis.id,
            target_revision,
        )
        is_independent = (
            interaction_id is not None and interaction_id not in previous_interactions
        )

        evidence = HypothesisEvidence(
            hypothesis_id=hypothesis.id,
            hypothesis_revision=target_revision,
            reflection_id=reflection.id,
            relation_type=relation_type,
            relation_confidence=max(0.0, min(1.0, float(relation_confidence))),
            relation_reason=relation_reason,
            is_independent=is_independent,
        )
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)

        if recalculate_current and target_revision == hypothesis.current_revision:
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
        self,
        hypothesis_id: int,
        revision_number: int,
    ) -> list[tuple[HypothesisEvidence, ReflectionRecord, Experience]]:
        stmt = (
            select(HypothesisEvidence, ReflectionRecord, Experience)
            .join(
                ReflectionRecord,
                ReflectionRecord.id == HypothesisEvidence.reflection_id,
            )
            .join(Experience, Experience.id == ReflectionRecord.experience_id)
            .where(
                HypothesisEvidence.hypothesis_id == hypothesis_id,
                HypothesisEvidence.hypothesis_revision == revision_number,
                ReflectionRecord.status == "active",
            )
        )
        return list(self.db.execute(stmt).all())

    def get_revision_stats(
        self,
        hypothesis_id: int,
        revision_number: int,
    ) -> dict[str, Any]:
        if self.get_revision(hypothesis_id, revision_number) is None:
            raise ValueError(
                f"Hypothesis #{hypothesis_id}:r{revision_number} 不存在"
            )

        rows = self._evidence_rows(hypothesis_id, revision_number)
        supporting = [row for row in rows if row[0].relation_type == "SUPPORTS"]
        refinements = [row for row in rows if row[0].relation_type == "REFINES"]
        conflicts = [row for row in rows if row[0].relation_type == "CONFLICTS"]

        independent_count = sum(1 for evidence, _, _ in rows if evidence.is_independent)
        independent_support_count = sum(
            1 for evidence, _, _ in supporting if evidence.is_independent
        )

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

        return {
            "supporting_count": len(supporting),
            "refinement_count": len(refinements),
            "counter_count": len(conflicts),
            "independent_evidence_count": independent_count,
            "independent_support_count": independent_support_count,
            "context_diversity": context_diversity,
            "confidence": round(confidence, 4),
        }

    def recalculate(
        self,
        hypothesis_id: int,
        *,
        commit: bool = True,
    ) -> StrategyHypothesis:
        hypothesis = self.db.get(StrategyHypothesis, hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"Hypothesis #{hypothesis_id} 不存在")

        current_revision = int(hypothesis.current_revision or hypothesis.revision or 1)
        stats = self.get_revision_stats(hypothesis_id, current_revision)

        hypothesis.supporting_count = stats["supporting_count"]
        hypothesis.refinement_count = stats["refinement_count"]
        hypothesis.counter_count = stats["counter_count"]
        hypothesis.independent_evidence_count = stats["independent_evidence_count"]
        hypothesis.independent_support_count = stats["independent_support_count"]
        hypothesis.context_diversity = stats["context_diversity"]
        hypothesis.confidence = stats["confidence"]
        hypothesis.last_evidence_at = datetime.now()

        if hypothesis.status in ACTIVE_HYPOTHESIS_STATUSES:
            pressure = RevisionReconciliationService(self.db).pressure_summary(
                hypothesis_id, current_revision
            )
            # Current-state semantics: historical REFINES evidence is audit history.
            # Only open signals on the active head mean "needs_revision". Signals
            # inherited from an ancestor must first be revalidated/rebased.
            if pressure.active_open > 0:
                hypothesis.status = "needs_revision"
            elif pressure.ancestor_open > 0:
                hypothesis.status = "needs_revalidation"
            elif (
                hypothesis.refinement_count > 0
                and pressure.resolved == 0
                and pressure.superseded == 0
                and pressure.inactive_branch_open == 0
            ):
                # Compatibility/failsafe: a REFINE relation may exist without an
                # atomic signal (legacy data or a judge that emitted no valid
                # signal). Do not silently downgrade that unresolved evidence.
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

        if commit:
            self.db.commit()
            self.db.refresh(hypothesis)
        else:
            self.db.flush()
        return hypothesis

    def activate_revision(
        self,
        hypothesis_id: int,
        revision_number: int,
        *,
        commit: bool = True,
    ) -> StrategyHypothesis:
        """切换 current revision 的受控入口。

        V0.2.1 第一阶段只提供版本切换地基；Risk Gate / EvolutionEvent 将在后续阶段接入。
        """
        hypothesis = self.get(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"Hypothesis #{hypothesis_id} 不存在")
        revision = self.get_revision(hypothesis_id, revision_number)
        if revision is None:
            raise ValueError(
                f"Hypothesis #{hypothesis_id}:r{revision_number} 不存在"
            )

        hypothesis.current_revision = revision_number
        hypothesis.revision = revision_number  # V0.2 compatibility mirror

        # StrategyHypothesis 暂时保留 current-head 内容，避免上层 Retriever/Aggregator
        # 在第一阶段同时大改；内容来源始终是不可变 Revision snapshot。
        hypothesis.title = revision.title
        hypothesis.canonical_signal = revision.canonical_signal
        hypothesis.insight = revision.insight
        hypothesis.applicable_context = revision.when_data or {}
        hypothesis.proposed_change = str((revision.do_data or {}).get("text", ""))
        hypothesis.expected_effect = str((revision.expect_data or {}).get("text", ""))

        if commit:
            self.db.commit()
            self.db.refresh(hypothesis)
        else:
            self.db.flush()
        return self.recalculate(hypothesis_id, commit=commit)

    def recalculate_all(self) -> list[StrategyHypothesis]:
        """按各 Hypothesis 的 current revision 重新计算活动 Hypothesis。"""
        hypotheses = self.list_all(include_inactive=False)
        return [self.recalculate(hypothesis.id) for hypothesis in hypotheses]
