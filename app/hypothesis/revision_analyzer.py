from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.revision_aggregation import (
    RevisionAggregationService,
    RevisionSignalGroup,
)
from app.hypothesis.service import HypothesisService
from app.models import (
    RevisionAnalysisRecord,
    RevisionProposal,
    RevisionSignal,
)


MIN_PROPOSAL_CONFIDENCE = 0.70


@dataclass(frozen=True)
class RevisionAnalysisResult:
    analysis: RevisionAnalysisRecord
    proposal: RevisionProposal | None


class RevisionAnalysisService:
    """Analyze deterministic revision-signal groups without mutating hypotheses.

    Phase 4 may persist an analysis and a proposal. It must not create a new
    HypothesisRevision or change the active StrategyHypothesis.
    """

    def __init__(
        self,
        db: Session,
        llm,
        *,
        min_independent_signals: int = 2,
    ) -> None:
        self.db = db
        self.llm = llm
        self.aggregation_service = RevisionAggregationService(
            db,
            min_independent_signals=min_independent_signals,
        )
        self.hypothesis_service = HypothesisService(db)

    @staticmethod
    def _fingerprint(group: RevisionSignalGroup) -> str:
        payload = {
            "hypothesis_id": group.hypothesis_id,
            "base_revision": group.base_revision,
            "dimension": group.dimension,
            "operation": group.operation,
            "target_field": group.target_field,
            "signal_ids": sorted(group.signal_ids),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "consistent"}
        return bool(value)

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _unique(values: list[Any]) -> list[Any]:
        seen: set[str] = set()
        result: list[Any] = []
        for value in values:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    def _signals(self, group: RevisionSignalGroup) -> list[RevisionSignal]:
        if not group.signal_ids:
            return []
        stmt = (
            select(RevisionSignal)
            .where(RevisionSignal.id.in_(group.signal_ids))
            .order_by(RevisionSignal.id)
        )
        return list(self.db.scalars(stmt).all())

    def _llm_payload(
        self,
        group: RevisionSignalGroup,
        signals: list[RevisionSignal],
    ) -> dict[str, Any]:
        revision = self.hypothesis_service.get_revision(
            group.hypothesis_id,
            group.base_revision,
        )
        return {
            "hypothesis": {
                "id": group.hypothesis_id,
                "revision": group.base_revision,
                "when": revision.when_data if revision else {},
                "do": revision.do_data if revision else {},
                "expect": revision.expect_data if revision else {},
            },
            "candidate": group.to_dict(),
            "signals": [
                {
                    "id": signal.id,
                    "before_data": signal.before_data,
                    "observed_data": signal.observed_data,
                    "canonical_change": signal.canonical_change,
                    "confidence": signal.confidence,
                }
                for signal in signals
            ],
        }

    def _judge(
        self,
        group: RevisionSignalGroup,
        signals: list[RevisionSignal],
    ) -> dict[str, Any]:
        payload = self._llm_payload(group, signals)
        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze evidence for a proposed hypothesis revision. "
                    "Decide whether all signals point to one coherent local change. "
                    "Do not broaden the change beyond the supplied evidence. Return JSON "
                    "with: consistent (boolean), direction_confidence (0..1), reason "
                    "(string), and proposed_after_data (JSON value or null)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]
        result = self.llm.json_chat(messages, temperature=0.1)
        return result if isinstance(result, dict) else {}

    def analyze_group(
        self,
        group: RevisionSignalGroup,
    ) -> RevisionAnalysisResult | None:
        if not group.eligible_for_analysis:
            return None

        fingerprint = self._fingerprint(group)
        existing = self.db.scalar(
            select(RevisionAnalysisRecord).where(
                RevisionAnalysisRecord.candidate_fingerprint == fingerprint
            )
        )
        if existing is not None:
            proposal = self.db.scalar(
                select(RevisionProposal).where(
                    RevisionProposal.analysis_id == existing.id
                )
            )
            return RevisionAnalysisResult(existing, proposal)

        signals = self._signals(group)
        raw = self._judge(group, signals)
        consistent = self._as_bool(raw.get("consistent", False))
        confidence = self._confidence(raw.get("direction_confidence", 0.0))
        reason = str(raw.get("reason", "")).strip()

        analysis = RevisionAnalysisRecord(
            hypothesis_id=group.hypothesis_id,
            base_revision=group.base_revision,
            dimension=group.dimension,
            operation=group.operation,
            target_field=group.target_field,
            signal_ids=sorted(group.signal_ids),
            candidate_fingerprint=fingerprint,
            consistent=consistent,
            direction_confidence=confidence,
            reason=reason,
        )
        self.db.add(analysis)
        self.db.flush()

        proposed_after_data = raw.get("proposed_after_data")
        proposal = None
        if (
            consistent
            and confidence >= MIN_PROPOSAL_CONFIDENCE
            and proposed_after_data is not None
        ):
            before_values = self._unique([signal.before_data for signal in signals])
            before_data: Any = before_values[0] if len(before_values) == 1 else before_values
            proposal = RevisionProposal(
                analysis_id=analysis.id,
                hypothesis_id=group.hypothesis_id,
                base_revision=group.base_revision,
                dimension=group.dimension,
                operation=group.operation,
                target_field=group.target_field,
                before_data=before_data,
                proposed_after_data=proposed_after_data,
                reason=reason,
                source_signal_ids=sorted(group.signal_ids),
                independent_signal_count=group.independent_count,
                direction_confidence=confidence,
                status="proposed",
            )
            self.db.add(proposal)

        self.db.commit()
        self.db.refresh(analysis)
        if proposal is not None:
            self.db.refresh(proposal)
        return RevisionAnalysisResult(analysis, proposal)

    def analyze_pending(self, *, max_analyses: int = 2) -> list[RevisionAnalysisResult]:
        limit = max(0, int(max_analyses))
        if limit == 0:
            return []

        results: list[RevisionAnalysisResult] = []
        for group in self.aggregation_service.group_signals():
            if not group.eligible_for_analysis:
                continue
            fingerprint = self._fingerprint(group)
            exists = self.db.scalar(
                select(RevisionAnalysisRecord.id).where(
                    RevisionAnalysisRecord.candidate_fingerprint == fingerprint
                )
            )
            if exists is not None:
                continue
            result = self.analyze_group(group)
            if result is not None:
                results.append(result)
            if len(results) >= limit:
                break
        return results

    def list_analyses(self) -> list[RevisionAnalysisRecord]:
        stmt = select(RevisionAnalysisRecord).order_by(RevisionAnalysisRecord.id.desc())
        return list(self.db.scalars(stmt).all())

    def list_proposals(self) -> list[RevisionProposal]:
        stmt = select(RevisionProposal).order_by(RevisionProposal.id.desc())
        return list(self.db.scalars(stmt).all())
