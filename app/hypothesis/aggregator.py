from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.hypothesis.canonical import build_canonical_signal
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.retriever import HypothesisRetriever
from app.hypothesis.service import HypothesisService
from app.models import ReflectionRecord, StrategyHypothesis


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

    @staticmethod
    def _reflection_payload(reflection: ReflectionRecord) -> dict[str, Any]:
        return {
            "insight": reflection.insight,
            "applicable_context": reflection.applicable_context or {},
            "proposed_change": reflection.strategy_signal,
            "expected_effect": reflection.expected_effect,
            "canonical_signal": reflection.canonical_signal,
        }

    @staticmethod
    def _hypothesis_payload(hypothesis: StrategyHypothesis) -> dict[str, Any]:
        return {
            "id": hypothesis.id,
            "insight": hypothesis.insight,
            "applicable_context": hypothesis.applicable_context or {},
            "proposed_change": hypothesis.proposed_change,
            "expected_effect": hypothesis.expected_effect,
            "canonical_signal": hypothesis.canonical_signal,
            "confidence": hypothesis.confidence,
        }

    def process(self, reflection: ReflectionRecord) -> dict[str, Any]:
        if not reflection.strategy_signal.strip() or not reflection.insight.strip():
            return {
                "action": "skipped",
                "hypothesis_id": None,
                "reason": "Reflection 没有可执行 strategy_signal 或 insight",
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

        judged: list[tuple[float, str, str, StrategyHypothesis]] = []
        for match in matches:
            hypothesis = self.service.get(match.hypothesis_id)
            if hypothesis is None or hypothesis.status not in {
                "observing", "ready_for_review", "needs_revision"
            }:
                continue
            decision = self.relation_judge.judge(
                reflection_payload=self._reflection_payload(reflection),
                hypothesis_payload=self._hypothesis_payload(hypothesis),
            )
            relation = decision["relation"]
            confidence = float(decision["confidence"])
            if relation == "UNRELATED" or confidence < self.min_relation_confidence:
                continue
            judged.append((confidence, relation, decision["reason"], hypothesis))

        if judged:
            confidence, relation, reason, hypothesis = max(
                judged, key=lambda item: item[0]
            )
            self.service.add_evidence(
                hypothesis=hypothesis,
                reflection=reflection,
                relation_type=relation,
                relation_confidence=confidence,
                relation_reason=reason,
            )
            updated = self.service.get(hypothesis.id)
            if updated is not None:
                self.retriever.index(updated)
            return {
                "action": relation.lower(),
                "hypothesis_id": hypothesis.id,
                "relation": relation,
                "relation_confidence": confidence,
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
        }
