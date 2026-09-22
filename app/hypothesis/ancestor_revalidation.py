from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context.schema import sanitize_applicable_context
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService
from app.models import (
    Experience,
    HypothesisEvidence,
    HypothesisRevision,
    InteractionRecord,
    ReflectionRecord,
    RevisionSignal,
    RevisionSignalRevalidation,
    StrategyHypothesis,
)


CURRENT_REVALIDATION_VERSION = 1

OUTCOMES = {
    "STILL_APPLIES",
    "RESOLVED_BY_CURRENT",
    "NO_LONGER_APPLIES",
    "UNCERTAIN",
}


@dataclass(frozen=True)
class AncestorRevalidationBatchResult:
    hypothesis_id: int
    target_revision: int
    record_ids: list[int]
    rebased_signal_ids: list[int]
    resolved_signal_ids: list[int]
    superseded_signal_ids: list[int]
    uncertain_signal_ids: list[int]
    llm_called: bool


class AncestorSignalRevalidationService:
    """Re-check ancestor-open signals against the current Hypothesis revision.

    The service never reads raw conversation history. It sends only the current
    immutable revision plus compact signal/reflection/provenance summaries to one
    batch LLM call. A STILL_APPLIES result carries the signal forward to the
    current revision; RESOLVED/NO_LONGER_APPLIES retire the ancestor pressure for
    the current branch; UNCERTAIN leaves it open for later review.
    """

    def __init__(
        self,
        db: Session,
        llm,
        *,
        max_signals_per_batch: int = 16,
    ) -> None:
        self.db = db
        self.llm = llm
        self.max_signals_per_batch = max(1, int(max_signals_per_batch))
        self.signal_service = RevisionSignalService(db)
        self.reconciliation = RevisionReconciliationService(db)

    @staticmethod
    def _compact_text(value: Any, limit: int = 360) -> str:
        text = str(value or "").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _parse_numeric_id(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        matches = re.findall(r"\d+", text)
        return int(matches[0]) if len(matches) == 1 else None

    @staticmethod
    def _normalize_outcome(value: Any) -> str | None:
        text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "STILL_APPLIES": "STILL_APPLIES",
            "APPLIES": "STILL_APPLIES",
            "STILL_VALID": "STILL_APPLIES",
            "REBASE": "STILL_APPLIES",
            "仍适用": "STILL_APPLIES",
            "仍然适用": "STILL_APPLIES",
            "RESOLVED_BY_CURRENT": "RESOLVED_BY_CURRENT",
            "RESOLVED": "RESOLVED_BY_CURRENT",
            "ALREADY_RESOLVED": "RESOLVED_BY_CURRENT",
            "SATISFIED": "RESOLVED_BY_CURRENT",
            "已解决": "RESOLVED_BY_CURRENT",
            "NO_LONGER_APPLIES": "NO_LONGER_APPLIES",
            "OBSOLETE": "NO_LONGER_APPLIES",
            "SUPERSEDED": "NO_LONGER_APPLIES",
            "NOT_APPLICABLE": "NO_LONGER_APPLIES",
            "不再适用": "NO_LONGER_APPLIES",
            "UNCERTAIN": "UNCERTAIN",
            "UNKNOWN": "UNCERTAIN",
            "REVIEW": "UNCERTAIN",
            "不确定": "UNCERTAIN",
        }
        outcome = aliases.get(text)
        return outcome if outcome in OUTCOMES else None

    def _existing_record(
        self,
        source_signal_id: int,
        target_revision: int,
    ) -> RevisionSignalRevalidation | None:
        return self.db.scalar(
            select(RevisionSignalRevalidation).where(
                RevisionSignalRevalidation.source_signal_id == int(source_signal_id),
                RevisionSignalRevalidation.target_revision == int(target_revision),
                RevisionSignalRevalidation.evaluation_version == CURRENT_REVALIDATION_VERSION,
            )
        )

    def _ancestor_signals(
        self,
        hypothesis: StrategyHypothesis,
        *,
        max_signals: int,
    ) -> list[RevisionSignal]:
        current = int(hypothesis.current_revision or hypothesis.revision or 1)
        pressure = self.reconciliation.pressure_summary(hypothesis.id, current)
        ids = set(pressure.ancestor_open_ids)
        if not ids:
            return []
        rows = [
            item
            for item in self.signal_service.list_all(hypothesis_id=hypothesis.id, valid_only=True)
            if int(item.id) in ids
            and self._existing_record(int(item.id), current) is None
        ]
        rows.sort(key=lambda item: (int(item.base_revision), int(item.id)))
        return rows[: max(1, int(max_signals))]

    def _compact_signal(self, signal: RevisionSignal) -> dict[str, Any]:
        reflection = self.db.get(ReflectionRecord, int(signal.reflection_id))
        experience = None
        interaction = None
        if reflection is not None and reflection.experience_id is not None:
            experience = self.db.get(Experience, int(reflection.experience_id))
        if experience is not None and experience.interaction_id is not None:
            interaction = self.db.get(InteractionRecord, int(experience.interaction_id))

        context = {}
        if reflection is not None:
            context = sanitize_applicable_context(reflection.applicable_context or {})
        if not context and experience is not None:
            context = sanitize_applicable_context(experience.context_data or {})

        return {
            "signal_id": int(signal.id),
            "base_revision": int(signal.base_revision),
            "dimension": str(signal.dimension).upper(),
            "operation": str(signal.operation).upper(),
            "target_field": str(signal.target_field),
            "before": signal.before_data,
            "observed": signal.observed_data,
            "canonical_change": self._compact_text(signal.canonical_change),
            "confidence": float(signal.confidence),
            "context": context,
            "reflection_signal": self._compact_text(
                (reflection.canonical_signal if reflection is not None else "")
                or (reflection.strategy_signal if reflection is not None else "")
                or (reflection.insight if reflection is not None else "")
            ),
            "source_user_excerpt": self._compact_text(
                (signal.provenance_data or {}).get("source_user_excerpt")
                or (interaction.user_input if interaction is not None else ""),
                240,
            ),
        }

    def _call_llm(
        self,
        *,
        revision: HypothesisRevision,
        signals: list[RevisionSignal],
    ) -> dict[str, Any]:
        compact = [self._compact_signal(item) for item in signals]
        messages = [
            {
                "role": "system",
                "content": (
                    "Revalidate ancestor RevisionSignals against the CURRENT immutable Hypothesis revision. "
                    "Judge each signal independently and only from the supplied current revision and compact "
                    "evidence. Outcomes: STILL_APPLIES when the refinement remains needed on the current head; "
                    "RESOLVED_BY_CURRENT when the descendant revision already implements/satisfies it; "
                    "NO_LONGER_APPLIES when the refinement became obsolete or incompatible with the current "
                    "branch; UNCERTAIN when evidence is insufficient. Do not reinterpret unrelated signals, "
                    "do not invent context, and do not treat a changed revision number alone as resolution. "
                    "Return JSON only with key `assessments`. Every input signal_id must appear once. Example: "
                    "{\"assessments\":[{\"signal_id\":7,\"outcome\":\"STILL_APPLIES\","
                    "\"confidence\":0.9,\"reason\":\"...\"}]}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_revision": {
                            "hypothesis_id": int(revision.hypothesis_id),
                            "revision": int(revision.revision_number),
                            "when": revision.when_data or {},
                            "do": revision.do_data or {},
                            "expect": revision.expect_data or {},
                        },
                        "ancestor_signals": compact,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        raw = self.llm.json_chat(messages, temperature=0.1)
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def _normalize_assessments(
        cls,
        raw: dict[str, Any],
        expected_ids: set[int],
    ) -> list[dict[str, Any]]:
        rows = raw.get("assessments")
        if rows is None:
            rows = raw.get("signal_assessments")
        if rows is None:
            rows = raw.get("revalidations")
        if not isinstance(rows, list):
            return []

        result: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in rows:
            if not isinstance(item, dict):
                continue
            signal_id = cls._parse_numeric_id(
                item.get("signal_id", item.get("revision_signal_id", item.get("id")))
            )
            if signal_id is None or signal_id not in expected_ids or signal_id in seen:
                continue
            outcome = cls._normalize_outcome(
                item.get("outcome", item.get("status", item.get("verdict")))
            )
            if outcome is None:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            seen.add(signal_id)
            result.append(
                {
                    "signal_id": signal_id,
                    "outcome": outcome,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": str(item.get("reason", item.get("explanation", ""))).strip(),
                }
            )
        return result

    @staticmethod
    def _target_before_data(revision: HypothesisRevision, signal: RevisionSignal) -> Any:
        dimension = str(signal.dimension).upper()
        field = str(signal.target_field)
        if dimension == "CONTEXT":
            return copy.deepcopy((revision.when_data or {}).get(field))
        if dimension == "ACTION" and field in (revision.do_data or {}):
            return copy.deepcopy((revision.do_data or {}).get(field))
        if dimension == "OUTCOME" and field in (revision.expect_data or {}):
            return copy.deepcopy((revision.expect_data or {}).get(field))
        return copy.deepcopy(signal.before_data)

    def _materialize_rebased_signal(
        self,
        *,
        source: RevisionSignal,
        target_revision: HypothesisRevision,
        record: RevisionSignalRevalidation,
    ) -> RevisionSignal:
        existing = self.db.scalar(
            select(RevisionSignal).where(
                RevisionSignal.hypothesis_id == int(source.hypothesis_id),
                RevisionSignal.base_revision == int(target_revision.revision_number),
                RevisionSignal.reflection_id == int(source.reflection_id),
                RevisionSignal.dimension == str(source.dimension),
                RevisionSignal.operation == str(source.operation),
                RevisionSignal.target_field == str(source.target_field),
            )
        )
        if existing is not None:
            return existing

        evidence = self.db.scalar(
            select(HypothesisEvidence).where(
                HypothesisEvidence.hypothesis_id == int(source.hypothesis_id),
                HypothesisEvidence.hypothesis_revision == int(target_revision.revision_number),
                HypothesisEvidence.reflection_id == int(source.reflection_id),
            )
        )
        provenance = dict(source.provenance_data or {})
        provenance.update(
            {
                "rebased_from_signal_id": int(source.id),
                "revalidated_from_revision": int(source.base_revision),
                "revalidated_to_revision": int(target_revision.revision_number),
                "revalidation_record_id": int(record.id),
                "revalidation_confidence": float(record.confidence),
                "revalidation_outcome": "STILL_APPLIES",
            }
        )
        signal = RevisionSignal(
            hypothesis_id=int(source.hypothesis_id),
            base_revision=int(target_revision.revision_number),
            reflection_id=int(source.reflection_id),
            evidence_id=(int(evidence.id) if evidence is not None else None),
            interaction_id=source.interaction_id,
            dimension=str(source.dimension),
            operation=str(source.operation),
            target_field=str(source.target_field),
            before_data=self._target_before_data(target_revision, source),
            observed_data=copy.deepcopy(source.observed_data),
            canonical_change=str(source.canonical_change),
            confidence=min(float(source.confidence), max(0.5, float(record.confidence))),
            provenance_data=provenance,
            resolution_status="open",
        )
        self.db.add(signal)
        self.db.flush()
        return signal

    def _apply_assessment(
        self,
        *,
        source: RevisionSignal,
        target_revision: HypothesisRevision,
        assessment: dict[str, Any],
        raw: dict[str, Any],
    ) -> RevisionSignalRevalidation:
        existing = self._existing_record(int(source.id), int(target_revision.revision_number))
        if existing is not None:
            return existing

        outcome = str(assessment["outcome"])
        record = RevisionSignalRevalidation(
            hypothesis_id=int(source.hypothesis_id),
            source_signal_id=int(source.id),
            source_revision=int(source.base_revision),
            target_revision=int(target_revision.revision_number),
            evaluation_version=CURRENT_REVALIDATION_VERSION,
            outcome=outcome,
            confidence=float(assessment["confidence"]),
            reason=str(assessment.get("reason", "")),
            evaluation_data={"raw_batch_output": raw},
        )
        self.db.add(record)
        self.db.flush()

        now = datetime.now()
        if outcome == "STILL_APPLIES":
            rebased = self._materialize_rebased_signal(
                source=source,
                target_revision=target_revision,
                record=record,
            )
            record.rebased_signal_id = int(rebased.id)
            source.resolution_status = "superseded"
            source.superseded_by_signal_id = int(rebased.id)
            source.superseded_by_revision = int(target_revision.revision_number)
            source.superseded_at = now
            source.resolution_note = (
                f"revalidated_still_applies_as_RS#{rebased.id}_on_r{target_revision.revision_number}"
            )
        elif outcome == "RESOLVED_BY_CURRENT":
            source.resolution_status = "resolved"
            source.resolved_by_candidate_id = None
            source.resolved_by_revision = int(target_revision.revision_number)
            source.resolved_at = now
            source.resolution_note = (
                f"ancestor_revalidation_resolved_by_r{target_revision.revision_number}"
            )
        elif outcome == "NO_LONGER_APPLIES":
            source.resolution_status = "superseded"
            source.superseded_by_signal_id = None
            source.superseded_by_revision = int(target_revision.revision_number)
            source.superseded_at = now
            source.resolution_note = (
                f"ancestor_revalidation_obsolete_on_r{target_revision.revision_number}"
            )
        # UNCERTAIN deliberately leaves the source signal open.

        self.db.flush()
        return record

    def revalidate_hypothesis(
        self,
        hypothesis_id: int,
        *,
        max_signals: int | None = None,
    ) -> AncestorRevalidationBatchResult:
        hypothesis = self.db.get(StrategyHypothesis, int(hypothesis_id))
        if hypothesis is None:
            raise ValueError(f"H#{hypothesis_id} 不存在")
        current = int(hypothesis.current_revision or hypothesis.revision or 1)
        revision = self.db.scalar(
            select(HypothesisRevision).where(
                HypothesisRevision.hypothesis_id == int(hypothesis_id),
                HypothesisRevision.revision_number == current,
            )
        )
        if revision is None:
            raise ValueError(f"H#{hypothesis_id}:r{current} 不存在")

        signals = self._ancestor_signals(
            hypothesis,
            max_signals=(max_signals or self.max_signals_per_batch),
        )
        if not signals:
            return AncestorRevalidationBatchResult(
                hypothesis_id=int(hypothesis_id),
                target_revision=current,
                record_ids=[],
                rebased_signal_ids=[],
                resolved_signal_ids=[],
                superseded_signal_ids=[],
                uncertain_signal_ids=[],
                llm_called=False,
            )

        raw = self._call_llm(revision=revision, signals=signals)
        expected_ids = {int(item.id) for item in signals}
        assessments = self._normalize_assessments(raw, expected_ids)
        by_id = {int(item.id): item for item in signals}

        record_ids: list[int] = []
        rebased_signal_ids: list[int] = []
        resolved_signal_ids: list[int] = []
        superseded_signal_ids: list[int] = []
        uncertain_signal_ids: list[int] = []

        for assessment in assessments:
            source = by_id[int(assessment["signal_id"])]
            record = self._apply_assessment(
                source=source,
                target_revision=revision,
                assessment=assessment,
                raw=raw,
            )
            record_ids.append(int(record.id))
            if record.outcome == "STILL_APPLIES" and record.rebased_signal_id is not None:
                rebased_signal_ids.append(int(record.rebased_signal_id))
            elif record.outcome == "RESOLVED_BY_CURRENT":
                resolved_signal_ids.append(int(source.id))
            elif record.outcome == "NO_LONGER_APPLIES":
                superseded_signal_ids.append(int(source.id))
            elif record.outcome == "UNCERTAIN":
                uncertain_signal_ids.append(int(source.id))

        # Recalculate the hypothesis only after branch-local signal state has
        # been projected. This updates needs_revalidation/needs_revision without
        # rewriting historical evidence.
        HypothesisService(self.db).recalculate(int(hypothesis_id), commit=False)
        self.db.commit()
        return AncestorRevalidationBatchResult(
            hypothesis_id=int(hypothesis_id),
            target_revision=current,
            record_ids=record_ids,
            rebased_signal_ids=rebased_signal_ids,
            resolved_signal_ids=resolved_signal_ids,
            superseded_signal_ids=superseded_signal_ids,
            uncertain_signal_ids=uncertain_signal_ids,
            llm_called=True,
        )

    def revalidate_pending(self, *, max_hypotheses: int = 2) -> list[AncestorRevalidationBatchResult]:
        results: list[AncestorRevalidationBatchResult] = []
        hypotheses = list(
            self.db.scalars(
                select(StrategyHypothesis).order_by(StrategyHypothesis.id.asc())
            ).all()
        )
        for hypothesis in hypotheses:
            if len(results) >= max(0, int(max_hypotheses)):
                break
            pressure = self.reconciliation.pressure_summary(int(hypothesis.id))
            if pressure.ancestor_open <= 0:
                continue
            result = self.revalidate_hypothesis(int(hypothesis.id))
            if result.llm_called or result.record_ids:
                results.append(result)
        return results

    def list_records(self, *, limit: int = 100) -> list[RevisionSignalRevalidation]:
        stmt = (
            select(RevisionSignalRevalidation)
            .order_by(RevisionSignalRevalidation.created_at.desc(), RevisionSignalRevalidation.id.desc())
            .limit(max(1, int(limit)))
        )
        return list(self.db.scalars(stmt).all())
