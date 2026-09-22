from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.hypothesis.canonical import build_canonical_signal
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.retriever import HypothesisRetriever
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService
from app.models import Experience, HypothesisRevision, ReflectionRecord, StrategyHypothesis


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
            return {
                "id": hypothesis.id,
                "revision": hypothesis.current_revision,
                "insight": hypothesis.insight,
                "applicable_context": hypothesis.applicable_context or {},
                "proposed_change": hypothesis.proposed_change,
                "expected_effect": hypothesis.expected_effect,
                "canonical_signal": hypothesis.canonical_signal,
                "confidence": hypothesis.confidence,
            }
        return self._revision_payload(hypothesis, revision)

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

            for hypothesis, _ in candidates:
                decision = decision_by_id.get(hypothesis.id)
                if decision is None:
                    continue
                relation = decision["relation"]
                confidence = float(decision["confidence"])
                if not bool(decision.get("decision_valid", True)):
                    if relation != "UNRELATED":
                        invalid_related_decisions.append((hypothesis, decision))
                    continue
                if relation == "UNRELATED" or confidence < self.min_relation_confidence:
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
