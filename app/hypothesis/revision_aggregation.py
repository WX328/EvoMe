from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context.schema import sanitize_applicable_context
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService
from app.models import Experience, HypothesisEvidence, ReflectionRecord, RevisionSignal


@dataclass(frozen=True)
class RevisionSignalGroup:
    hypothesis_id: int
    base_revision: int
    dimension: str
    operation: str
    target_field: str
    signal_count: int
    independent_count: int
    average_confidence: float
    signal_ids: list[int]
    reflection_ids: list[int]
    interaction_ids: list[int]
    observed_values: list[Any]
    eligible_for_analysis: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RevisionAggregationService:
    """Phase 3 deterministic aggregation.

    这里只做可复现的计数、分组与压缩，不做语义判断，不调用 LLM，
    也不会创建 RevisionProposal 或修改 Hypothesis。
    """

    def __init__(
        self,
        db: Session,
        *,
        min_independent_signals: int = 2,
    ) -> None:
        self.db = db
        self.min_independent_signals = max(1, int(min_independent_signals))
        self.signal_service = RevisionSignalService(db)
        self.hypothesis_service = HypothesisService(db)

    @staticmethod
    def _stable_key(value: Any) -> str:
        if isinstance(value, dict):
            return repr(sorted((str(k), RevisionAggregationService._stable_key(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set)):
            return repr([RevisionAggregationService._stable_key(v) for v in value])
        return repr(value)

    @classmethod
    def _unique_values(cls, values: list[Any], *, limit: int = 6) -> list[Any]:
        seen: set[str] = set()
        result: list[Any] = []
        for value in values:
            key = cls._stable_key(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return result

    def group_signals(
        self,
        *,
        hypothesis_id: int | None = None,
        base_revision: int | None = None,
    ) -> list[RevisionSignalGroup]:
        signals = self.signal_service.list_all(
            hypothesis_id=hypothesis_id,
            base_revision=base_revision,
            valid_only=True,
        )

        buckets: dict[tuple[int, int, str, str, str], list[RevisionSignal]] = {}
        for signal in signals:
            key = (
                int(signal.hypothesis_id),
                int(signal.base_revision),
                str(signal.dimension),
                str(signal.operation),
                str(signal.target_field),
            )
            buckets.setdefault(key, []).append(signal)

        groups: list[RevisionSignalGroup] = []
        for key, items in buckets.items():
            hid, revision, dimension, operation, target_field = key
            interaction_ids = sorted(
                {int(item.interaction_id) for item in items if item.interaction_id is not None}
            )
            reflection_ids = sorted({int(item.reflection_id) for item in items})
            signal_ids = sorted(int(item.id) for item in items)
            confidences = [float(item.confidence) for item in items]
            observed_values = self._unique_values([item.observed_data for item in items])
            independent_count = len(interaction_ids)

            groups.append(
                RevisionSignalGroup(
                    hypothesis_id=hid,
                    base_revision=revision,
                    dimension=dimension,
                    operation=operation,
                    target_field=target_field,
                    signal_count=len(items),
                    independent_count=independent_count,
                    average_confidence=round(mean(confidences), 4) if confidences else 0.0,
                    signal_ids=signal_ids,
                    reflection_ids=reflection_ids,
                    interaction_ids=interaction_ids,
                    observed_values=observed_values,
                    eligible_for_analysis=(independent_count >= self.min_independent_signals),
                )
            )

        groups.sort(
            key=lambda group: (
                group.hypothesis_id,
                group.base_revision,
                -group.independent_count,
                -group.signal_count,
                group.dimension,
                group.operation,
                group.target_field,
            )
        )
        return groups

    def _evidence_contexts(
        self,
        hypothesis_id: int,
        revision_number: int,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(ReflectionRecord, Experience)
            .join(
                HypothesisEvidence,
                HypothesisEvidence.reflection_id == ReflectionRecord.id,
            )
            .join(Experience, Experience.id == ReflectionRecord.experience_id)
            .where(
                HypothesisEvidence.hypothesis_id == hypothesis_id,
                HypothesisEvidence.hypothesis_revision == revision_number,
            )
        )
        contexts: list[dict[str, Any]] = []
        for reflection, experience in self.db.execute(stmt).all():
            reflected = sanitize_applicable_context(reflection.applicable_context or {})
            if reflected:
                contexts.append(reflected)
                continue
            contexts.append(sanitize_applicable_context(experience.context_data or {}))
        return contexts

    def _context_coverage(
        self,
        hypothesis_id: int,
        revision_number: int,
    ) -> dict[str, list[Any]]:
        coverage: dict[str, list[Any]] = {}
        for context in self._evidence_contexts(hypothesis_id, revision_number):
            for field, value in context.items():
                coverage.setdefault(field, []).append(value)
        return {
            field: self._unique_values(values)
            for field, values in sorted(coverage.items())
            if values
        }

    def build_evidence_summary(
        self,
        *,
        hypothesis_id: int,
        revision_number: int | None = None,
    ) -> dict[str, Any]:
        hypothesis = self.hypothesis_service.get(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"Hypothesis #{hypothesis_id} 不存在")

        target_revision = int(revision_number or hypothesis.current_revision or 1)
        stats = self.hypothesis_service.get_revision_stats(hypothesis_id, target_revision)
        groups = self.group_signals(
            hypothesis_id=hypothesis_id,
            base_revision=target_revision,
        )

        return {
            "hypothesis_id": hypothesis_id,
            "revision": target_revision,
            "evidence_counts": {
                "supports": stats["supporting_count"],
                "refines": stats["refinement_count"],
                "conflicts": stats["counter_count"],
                "independent_evidence": stats["independent_evidence_count"],
                "independent_support": stats["independent_support_count"],
                "context_diversity": stats["context_diversity"],
            },
            "confidence": stats["confidence"],
            "context_coverage": self._context_coverage(hypothesis_id, target_revision),
            "revision_groups": [group.to_dict() for group in groups],
            "analysis_candidates": [
                {
                    "dimension": group.dimension,
                    "operation": group.operation,
                    "target_field": group.target_field,
                    "signal_count": group.signal_count,
                    "independent_count": group.independent_count,
                    "average_confidence": group.average_confidence,
                    "signal_ids": group.signal_ids,
                }
                for group in groups
                if group.eligible_for_analysis
            ],
        }
