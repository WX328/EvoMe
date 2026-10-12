from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.canonical import build_canonical_signal
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.retriever import HypothesisRetriever
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService
from app.models import (
    Experience,
    HypothesisRevision,
    ReflectionRecord,
    RevisionSignal,
    StrategyHypothesis,
)


class HypothesisAggregator:
    def __init__(
        self,
        db: Session,
        *,
        relation_judge: HypothesisRelationJudge,
        retriever: HypothesisRetriever,
        top_k: int = 5,
        score_threshold: float = 0.60,
        min_relation_confidence: float = 0.60,
    ) -> None:
        self.db = db
        self.service = HypothesisService(db)
        self.revision_signal_service = RevisionSignalService(db)
        self.revision_reconciliation_service = RevisionReconciliationService(db)
        self.relation_judge = relation_judge
        self.retriever = retriever
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.min_relation_confidence = min_relation_confidence

    def sync_index(self) -> None:
        # 启动时先按当前 Evidence 规则重算，修正旧 V0.2 数据中
        # “REFINES 被当作 SUPPORTS”造成的错误成熟度。
        self.service.recalculate_all()
        for hypothesis in self.service.list_all(include_inactive=False):
            self.retriever.index(hypothesis)

    def _reflection_payload(self, reflection: ReflectionRecord) -> dict[str, Any]:
        user_feedback = ""
        if reflection.experience_id is not None:
            experience = self.db.get(Experience, reflection.experience_id)
            if experience is not None and experience.user_feedback:
                user_feedback = str(experience.user_feedback).strip()
        return {
            "insight": reflection.insight,
            "applicable_context": reflection.applicable_context or {},
            "proposed_change": reflection.strategy_signal,
            "expected_effect": reflection.expected_effect,
            "canonical_signal": reflection.canonical_signal,
            "user_feedback": user_feedback,
        }

    def _hypothesis_payload(self, hypothesis: StrategyHypothesis) -> dict[str, Any]:
        revision = self.service.get_revision(
            hypothesis.id,
            int(hypothesis.current_revision or hypothesis.revision or 1),
        )
        if revision is None:
            payload = {
                "id": hypothesis.id,
                "revision": hypothesis.current_revision,
                "insight": hypothesis.insight,
                "applicable_context": hypothesis.applicable_context or {},
                "proposed_change": hypothesis.proposed_change,
                "expected_effect": hypothesis.expected_effect,
                "canonical_signal": hypothesis.canonical_signal,
                "confidence": hypothesis.confidence,
            }
        else:
            payload = self._revision_payload(hypothesis, revision)

        pending_signals = self._active_pending_signal_payloads(hypothesis)
        if pending_signals:
            payload["pending_signals"] = pending_signals
        return payload

    def _active_pending_signal_payloads(
        self,
        hypothesis: StrategyHypothesis,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return a tiny active-open signal view for semantic continuity only."""
        current_revision = int(
            hypothesis.current_revision or hypothesis.revision or 1
        )
        pressure = self.revision_reconciliation_service.pressure_summary(
            hypothesis.id,
            current_revision,
        )
        if not pressure.active_open_ids:
            return []

        signals = list(
            self.db.scalars(
                select(RevisionSignal)
                .where(RevisionSignal.id.in_(pressure.active_open_ids))
                .order_by(RevisionSignal.confidence.desc(), RevisionSignal.id.desc())
                .limit(max(0, int(limit)))
            ).all()
        )
        return [
            {
                "signal_id": int(signal.id),
                "dimension": signal.dimension,
                "operation": signal.operation,
                "target_field": signal.target_field,
                "observed_data": signal.observed_data,
                "canonical_change": signal.canonical_change,
                "confidence": round(float(signal.confidence or 0.0), 4),
            }
            for signal in signals
        ]

    @staticmethod
    def _continuity_text(value: Any) -> str:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())

    @staticmethod
    def _continuity_segments(value: Any) -> list[str]:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return [
            part
            for part in re.split(
                r"[^0-9a-z\u4e00-\u9fff]+",
                str(value or "").lower(),
            )
            if len(part) >= 2
        ]

    @classmethod
    def _matches_pending_signal(
        cls,
        *,
        reflection_payload: dict[str, Any],
        hypothesis_payload: dict[str, Any],
    ) -> bool:
        """High-precision duplicate guard without a fuzzy similarity threshold."""
        reflection_text = cls._continuity_text(
            " ".join(
                str(reflection_payload.get(key) or "")
                for key in (
                    "insight",
                    "proposed_change",
                    "expected_effect",
                    "canonical_signal",
                )
            )
        )
        if not reflection_text:
            return False
        for signal in hypothesis_payload.get("pending_signals") or []:
            if not isinstance(signal, dict):
                continue
            for key in ("observed_data", "canonical_change"):
                pending_text = cls._continuity_text(signal.get(key))
                if len(pending_text) >= 6 and pending_text in reflection_text:
                    return True
                segments = cls._continuity_segments(signal.get(key))
                if len(segments) >= 2 and all(
                    segment in reflection_text for segment in segments
                ):
                    return True
        return False

    @staticmethod
    def _revision_payload(
        hypothesis: StrategyHypothesis,
        revision: HypothesisRevision,
    ) -> dict[str, Any]:
        return {
            "id": hypothesis.id,
            "revision": revision.revision_number,
            "insight": revision.insight,
            "when_data": revision.when_data or {},
            "do_data": revision.do_data or {},
            "expect_data": revision.expect_data or {},
            # 同时保留 Relation Judge 已熟悉的语义键名，减少提示词漂移。
            "applicable_context": revision.when_data or {},
            "proposed_change": (revision.do_data or {}).get("text", revision.do_data or {}),
            "expected_effect": (revision.expect_data or {}).get(
                "text", revision.expect_data or {}
            ),
            "canonical_signal": revision.canonical_signal,
            "confidence": hypothesis.confidence,
        }

    def process(self, reflection: ReflectionRecord) -> dict[str, Any]:
        if not reflection.strategy_signal.strip() or not reflection.insight.strip():
            return {
                "action": "skipped",
                "hypothesis_id": None,
                "reason": "Reflection 没有可执行 strategy_signal 或 insight",
                "revision_signal_count": 0,
            }

        canonical_signal = reflection.canonical_signal.strip() or build_canonical_signal(
            applicable_context=reflection.applicable_context,
            proposed_change=reflection.strategy_signal,
            expected_effect=reflection.expected_effect,
            insight=reflection.insight,
        )
        if not reflection.canonical_signal:
            reflection.canonical_signal = canonical_signal
            self.db.commit()

        matches = self.retriever.search(
            canonical_signal,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        judged: list[
            tuple[
                float,
                str,
                str,
                list[dict[str, Any]],
                StrategyHypothesis,
            ]
        ] = []

        # Phase 5.7B.1: retrieve first, then judge all viable candidates in one
        # compact LLM call. This replaces the old O(N) per-hypothesis loop.
        candidates: list[tuple[StrategyHypothesis, dict[str, Any]]] = []
        for match in matches:
            hypothesis = self.service.get(match.hypothesis_id)
            if hypothesis is None or hypothesis.status not in {
                "observing", "ready_for_review", "needs_revision", "needs_revalidation"
            }:
                continue
            candidates.append((hypothesis, self._hypothesis_payload(hypothesis)))

        invalid_related_decisions: list[tuple[StrategyHypothesis, dict[str, Any]]] = []
        ambiguous_decisions: list[tuple[StrategyHypothesis, str]] = []
        explicit_unrelated_ids: set[int] = set()
        if candidates:
            reflection_payload = self._reflection_payload(reflection)
            decisions = self.relation_judge.judge_batch(
                reflection_payload=reflection_payload,
                hypothesis_payloads=[payload for _, payload in candidates],
            )
            decision_by_id = {
                int(item["hypothesis_id"]): item
                for item in decisions
                if item.get("hypothesis_id") is not None
            }

            for hypothesis, hypothesis_payload in candidates:
                decision = decision_by_id.get(hypothesis.id)
                if decision is None:
                    ambiguous_decisions.append(
                        (hypothesis, "relation judge omitted this candidate")
                    )
                    continue
                relation = decision["relation"]
                confidence = float(decision["confidence"])
                if not bool(decision.get("decision_valid", True)):
                    if relation == "REFINES":
                        invalid_related_decisions.append((hypothesis, decision))
                    else:
                        ambiguous_decisions.append(
                            (
                                hypothesis,
                                str(
                                    decision.get("invalid_reason")
                                    or "invalid relation decision"
                                ),
                            )
                        )
                    continue
                if confidence < self.min_relation_confidence:
                    ambiguous_decisions.append(
                        (
                            hypothesis,
                            f"low-confidence {relation} decision ({confidence:.3f})",
                        )
                    )
                    continue
                if relation == "UNRELATED":
                    if self._matches_pending_signal(
                        reflection_payload=reflection_payload,
                        hypothesis_payload=hypothesis_payload,
                    ):
                        ambiguous_decisions.append(
                            (
                                hypothesis,
                                "UNRELATED conflicts with matching active pending signal",
                            )
                        )
                        continue
                    explicit_unrelated_ids.add(int(hypothesis.id))
                    continue
                judged.append(
                    (
                        confidence,
                        relation,
                        decision["reason"],
                        list(decision.get("revision_signals") or []),
                        hypothesis,
                    )
                )

        if judged:
            confidence, relation, reason, raw_revision_signals, hypothesis = max(
                judged, key=lambda item: item[0]
            )
            evidence = self.service.add_evidence(
                hypothesis=hypothesis,
                reflection=reflection,
                relation_type=relation,
                relation_confidence=confidence,
                relation_reason=reason,
            )
            created_signals = self.revision_signal_service.create_from_refines(
                hypothesis=hypothesis,
                reflection=reflection,
                evidence=evidence,
                raw_signals=raw_revision_signals,
            )
            # add_evidence() recalculates before REFINE signals exist. Phase 5.4
            # status semantics are signal-pressure based, so recalculate once more
            # after the atomic signals have been persisted.
            self.service.recalculate(hypothesis.id)
            updated = self.service.get(hypothesis.id)
            if updated is not None:
                self.retriever.index(updated)
            return {
                "action": relation.lower(),
                "hypothesis_id": hypothesis.id,
                "relation": relation,
                "relation_confidence": confidence,
                "revision_signal_count": len(created_signals),
            }

        # Fail closed when the judge says an existing hypothesis needs
        # refinement but cannot produce a valid atomic RevisionSignal. Creating
        # a duplicate hypothesis or recording REFINES-without-signal would both
        # corrupt evidence semantics. The Reflection remains persisted and can
        # be re-evaluated later.
        if invalid_related_decisions:
            hypothesis, decision = max(
                invalid_related_decisions,
                key=lambda item: float(item[1].get("confidence", 0.0)),
            )
            return {
                "action": "skipped_invalid_relation",
                "hypothesis_id": hypothesis.id,
                "relation": decision.get("relation", "REFINES"),
                "relation_confidence": float(decision.get("confidence", 0.0)),
                "revision_signal_count": 0,
                "reason": str(
                    decision.get("invalid_reason")
                    or "related relation decision failed semantic validation"
                ),
            }

        # Creating a new hypothesis is the only irreversible choice in this
        # matching step. Require every retrieved candidate to explicitly and
        # confidently say UNRELATED; missing/truncated/uncertain batch output
        # must fail closed instead of fragmenting the hypothesis set.
        if candidates and len(explicit_unrelated_ids) != len(candidates):
            reason = "; ".join(
                f"H#{hypothesis.id}: {item_reason}"
                for hypothesis, item_reason in ambiguous_decisions[:3]
            ) or "not every candidate was confidently unrelated"
            return {
                "action": "skipped_ambiguous_relation",
                "hypothesis_id": None,
                "relation": "AMBIGUOUS",
                "relation_confidence": 0.0,
                "revision_signal_count": 0,
                "reason": reason,
            }

        hypothesis = self.service.create_from_reflection(
            reflection=reflection,
            canonical_signal=canonical_signal,
        )
        self.retriever.index(hypothesis)
        return {
            "action": "created",
            "hypothesis_id": hypothesis.id,
            "relation": "SUPPORTS",
            "relation_confidence": reflection.confidence,
            "revision_signal_count": 0,
        }
