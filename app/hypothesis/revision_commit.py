from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.hypothesis.candidate_revision import CURRENT_EVALUATION_VERSION
from app.hypothesis.service import HypothesisService
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.models import (
    CandidateRevision,
    HypothesisEvidence,
    HypothesisRevision,
    RevisionApplicationEvent,
    RevisionProposal,
    StrategyHypothesis,
)


AUTO_APPLY_ACTIONS = {"AUTO_APPLY"}
APPLY_ACTIONS = {"AUTO_APPLY", "HUMAN_APPLY"}
ACTIONABLE_CANDIDATE_STATUSES = {"validating", "ready_to_apply", "needs_review"}


@dataclass(frozen=True)
class RevisionCommitResult:
    event: RevisionApplicationEvent | None
    candidate: CandidateRevision | None
    applied: bool
    reason: str


class RevisionCommitService:
    """Phase 5.3: Mixed Approval + immutable Revision commit.

    The Candidate is an experiment. Only this service may turn a validated
    Candidate into a formal HypothesisRevision. Auto-apply is deterministic and
    conservative; human approval may accept reviewed risk, but it still cannot
    bypass invalid evaluation, missing source coverage, known regression, or a
    stale base revision.
    """

    def __init__(
        self,
        db: Session,
        *,
        auto_min_utility: float = 0.20,
        auto_budget_hours: int = 24,
        auto_per_hypothesis: int = 1,
    ) -> None:
        self.db = db
        self.hypothesis_service = HypothesisService(db)
        self.reconciliation_service = RevisionReconciliationService(db)
        self.auto_min_utility = float(auto_min_utility)
        self.auto_budget_hours = max(1, int(auto_budget_hours))
        self.auto_per_hypothesis = max(0, int(auto_per_hypothesis))

    @staticmethod
    def _evaluation(candidate: CandidateRevision) -> dict[str, Any]:
        value = candidate.evaluation_data or {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _source_coverage(cls, candidate: CandidateRevision) -> float:
        data = cls._evaluation(candidate)
        try:
            return float(data.get("source_coverage", data.get("coverage", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _regression_count(cls, candidate: CandidateRevision) -> int:
        data = cls._evaluation(candidate)
        try:
            return int(data.get("regression_count", 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _evaluation_valid(cls, candidate: CandidateRevision) -> bool:
        return bool(cls._evaluation(candidate).get("evaluation_valid", False))

    def _existing_apply_event(self, candidate_id: int) -> RevisionApplicationEvent | None:
        stmt = select(RevisionApplicationEvent).where(
            RevisionApplicationEvent.candidate_id == candidate_id,
            RevisionApplicationEvent.action.in_(APPLY_ACTIONS),
        )
        return self.db.scalar(stmt)

    def _next_revision_number(self, hypothesis_id: int) -> int:
        maximum = self.db.scalar(
            select(func.max(HypothesisRevision.revision_number)).where(
                HypothesisRevision.hypothesis_id == hypothesis_id
            )
        )
        return int(maximum or 0) + 1

    def _auto_budget_used(self, hypothesis_id: int) -> int:
        if self.auto_per_hypothesis <= 0:
            return self.auto_per_hypothesis
        cutoff = datetime.now() - timedelta(hours=self.auto_budget_hours)
        count = self.db.scalar(
            select(func.count())
            .select_from(RevisionApplicationEvent)
            .where(
                RevisionApplicationEvent.hypothesis_id == hypothesis_id,
                RevisionApplicationEvent.action == "AUTO_APPLY",
                RevisionApplicationEvent.created_at >= cutoff,
            )
        )
        return int(count or 0)

    def _mark_stale(self, candidate: CandidateRevision, reason: str) -> None:
        if candidate.status in {"applied", "rejected", "stale"}:
            return
        candidate.status = "stale"
        data = copy.deepcopy(self._evaluation(candidate))
        data["stale_reason"] = reason
        data["stale_at"] = datetime.now().isoformat(timespec="seconds")
        candidate.evaluation_data = data
        proposal = self.db.get(RevisionProposal, candidate.source_proposal_id)
        if proposal is not None and proposal.status == "proposed":
            proposal.status = "stale"

    def mark_stale_candidates(self) -> int:
        """Invalidate Candidates whose base revision is no longer current."""
        rows = list(
            self.db.scalars(
                select(CandidateRevision).where(
                    CandidateRevision.status.in_(ACTIONABLE_CANDIDATE_STATUSES)
                )
            ).all()
        )
        changed = 0
        for candidate in rows:
            hypothesis = self.db.get(StrategyHypothesis, candidate.hypothesis_id)
            if hypothesis is None:
                self._mark_stale(candidate, "hypothesis_missing")
                changed += 1
                continue
            if int(hypothesis.current_revision or 1) != int(candidate.base_revision):
                self._mark_stale(
                    candidate,
                    f"base_revision_mismatch: current=r{hypothesis.current_revision}",
                )
                changed += 1
        if changed:
            self.db.commit()
        return changed

    def _auto_gate(self, candidate: CandidateRevision) -> tuple[bool, str]:
        if candidate.status != "ready_to_apply":
            return False, f"status={candidate.status}"
        proposal = self.db.get(RevisionProposal, int(candidate.source_proposal_id))
        if proposal is not None and str(proposal.proposal_kind or "raw") == "conditional_policy":
            return False, "conditional_policy_requires_manual_review"
        if int(candidate.evaluation_version or 0) != CURRENT_EVALUATION_VERSION:
            return False, "evaluation_version_not_current"
        if not self._evaluation_valid(candidate):
            return False, "evaluation_invalid"
        if self._source_coverage(candidate) < 0.999:
            return False, "source_coverage_incomplete"
        if self._regression_count(candidate) > 0:
            return False, "known_regression"
        if float(candidate.fit_gain or 0.0) <= 0:
            return False, "fit_gain_not_positive"
        if float(candidate.utility_score or 0.0) < self.auto_min_utility:
            return False, "utility_below_auto_threshold"
        if str(candidate.risk_level).upper() != "LOW":
            return False, "risk_not_low"

        hypothesis = self.db.get(StrategyHypothesis, candidate.hypothesis_id)
        if hypothesis is None:
            return False, "hypothesis_missing"
        if int(hypothesis.current_revision or 1) != int(candidate.base_revision):
            return False, "stale_base_revision"
        return True, "eligible"

    def _human_gate(self, candidate: CandidateRevision) -> tuple[bool, str]:
        if candidate.status not in {"ready_to_apply", "needs_review"}:
            return False, f"status={candidate.status}"
        if int(candidate.evaluation_version or 0) != CURRENT_EVALUATION_VERSION:
            return False, "evaluation_version_not_current"
        if not self._evaluation_valid(candidate):
            return False, "evaluation_invalid"
        if self._source_coverage(candidate) < 0.999:
            return False, "source_coverage_incomplete"
        if self._regression_count(candidate) > 0:
            return False, "known_regression"
        if float(candidate.fit_gain or 0.0) <= 0:
            return False, "fit_gain_not_positive"

        hypothesis = self.db.get(StrategyHypothesis, candidate.hypothesis_id)
        if hypothesis is None:
            return False, "hypothesis_missing"
        if int(hypothesis.current_revision or 1) != int(candidate.base_revision):
            return False, "stale_base_revision"
        return True, "eligible"

    def _clone_evidence(self, hypothesis_id: int, from_revision: int, to_revision: int) -> int:
        rows = list(
            self.db.scalars(
                select(HypothesisEvidence)
                .where(
                    HypothesisEvidence.hypothesis_id == hypothesis_id,
                    HypothesisEvidence.hypothesis_revision == from_revision,
                )
                .order_by(HypothesisEvidence.id.asc())
            ).all()
        )
        for item in rows:
            self.db.add(
                HypothesisEvidence(
                    hypothesis_id=hypothesis_id,
                    hypothesis_revision=to_revision,
                    reflection_id=item.reflection_id,
                    relation_type=item.relation_type,
                    relation_confidence=item.relation_confidence,
                    relation_reason=item.relation_reason,
                    is_independent=item.is_independent,
                    created_at=item.created_at,
                )
            )
        self.db.flush()
        return len(rows)

    def _stale_siblings(self, candidate: CandidateRevision, new_revision: int) -> list[int]:
        rows = list(
            self.db.scalars(
                select(CandidateRevision).where(
                    CandidateRevision.hypothesis_id == candidate.hypothesis_id,
                    CandidateRevision.base_revision == candidate.base_revision,
                    CandidateRevision.id != candidate.id,
                    CandidateRevision.status.in_(ACTIONABLE_CANDIDATE_STATUSES),
                )
            ).all()
        )
        ids: list[int] = []
        for sibling in rows:
            self._mark_stale(
                sibling,
                f"superseded_by_CR#{candidate.id}_creating_r{new_revision}",
            )
            ids.append(int(sibling.id))
        return ids

    @classmethod
    def _resolved_signal_ids(cls, candidate: CandidateRevision) -> list[int]:
        data = cls._evaluation(candidate)
        result: list[int] = []
        rows = data.get("source_signal_assessments", [])
        if not isinstance(rows, list):
            return result
        for item in rows:
            if not isinstance(item, dict):
                continue
            if str(item.get("resolution", "")).upper() != "RESOLVED":
                continue
            try:
                result.append(int(item["signal_id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(set(result))

    def _apply_candidate(
        self,
        candidate: CandidateRevision,
        *,
        action: str,
        reason: str,
    ) -> RevisionCommitResult:
        existing_event = self._existing_apply_event(int(candidate.id))
        if existing_event is not None:
            return RevisionCommitResult(existing_event, candidate, False, "already_applied")

        hypothesis = self.db.get(StrategyHypothesis, candidate.hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"H#{candidate.hypothesis_id} 不存在")
        current_revision = int(hypothesis.current_revision or hypothesis.revision or 1)
        if current_revision != int(candidate.base_revision):
            self._mark_stale(candidate, f"base_revision_mismatch: current=r{current_revision}")
            self.db.commit()
            return RevisionCommitResult(None, candidate, False, "stale_base_revision")

        proposal = self.db.get(RevisionProposal, candidate.source_proposal_id)
        if proposal is None:
            raise ValueError(f"RP#{candidate.source_proposal_id} 不存在")

        next_revision = self._next_revision_number(candidate.hypothesis_id)
        try:
            revision = self.hypothesis_service.create_revision_snapshot(
                hypothesis_id=candidate.hypothesis_id,
                revision_number=next_revision,
                title=candidate.title or hypothesis.title,
                canonical_signal=candidate.canonical_signal,
                insight=candidate.insight or hypothesis.insight,
                when_data=copy.deepcopy(candidate.when_data or {}),
                do_data=copy.deepcopy(candidate.do_data or {}),
                expect_data=copy.deepcopy(candidate.expect_data or {}),
                created_reason=(
                    f"{action.lower()}_candidate_CR#{candidate.id}_from_RP#{proposal.id}"
                ),
                parent_revision=current_revision,
                commit=False,
            )
            evidence_count = self._clone_evidence(
                candidate.hypothesis_id,
                current_revision,
                next_revision,
            )

            candidate.status = "applied"
            proposal.status = "applied"
            stale_candidate_ids = self._stale_siblings(candidate, next_revision)

            # Activate only after the immutable snapshot and carried evidence exist.
            self.hypothesis_service.activate_revision(
                candidate.hypothesis_id,
                revision.revision_number,
                commit=False,
            )

            event = RevisionApplicationEvent(
                hypothesis_id=candidate.hypothesis_id,
                from_revision=current_revision,
                to_revision=next_revision,
                candidate_id=candidate.id,
                proposal_id=proposal.id,
                action=action,
                risk_level=str(candidate.risk_level or ""),
                utility_score=float(candidate.utility_score or 0.0),
                reason=reason,
                event_data={
                    "evaluation_version": int(candidate.evaluation_version or 0),
                    "source_coverage": self._source_coverage(candidate),
                    "regression_count": self._regression_count(candidate),
                    "fit_gain": float(candidate.fit_gain or 0.0),
                    "resolved_signal_ids": self._resolved_signal_ids(candidate),
                    "carried_evidence_count": evidence_count,
                    "stale_candidate_ids": stale_candidate_ids,
                },
            )
            self.db.add(event)
            self.db.flush()
            # Project local source-signal resolution only after the immutable
            # revision/event exist, then recalculate current-state semantics.
            self.reconciliation_service.reconcile_application_event(event, commit=False)
            self.hypothesis_service.recalculate(candidate.hypothesis_id, commit=False)
            self.db.commit()
            self.db.refresh(event)
            self.db.refresh(candidate)
            return RevisionCommitResult(event, candidate, True, "applied")
        except Exception:
            self.db.rollback()
            raise

    def auto_apply_ready(self, *, max_candidates: int = 2) -> list[RevisionCommitResult]:
        """Apply LOW-risk candidates that satisfy every deterministic hard gate.

        At most one competing Candidate can advance a given current base revision;
        once it commits, siblings are stale and must later be rebased/re-evaluated.
        """
        limit = max(0, int(max_candidates))
        if limit == 0:
            return []

        self.mark_stale_candidates()
        rows = list(
            self.db.scalars(
                select(CandidateRevision)
                .where(CandidateRevision.status == "ready_to_apply")
                .order_by(
                    CandidateRevision.hypothesis_id.asc(),
                    CandidateRevision.utility_score.desc(),
                    CandidateRevision.id.asc(),
                )
            ).all()
        )

        results: list[RevisionCommitResult] = []
        applied_hypotheses: set[int] = set()
        for candidate in rows:
            if len(results) >= limit:
                break
            if int(candidate.hypothesis_id) in applied_hypotheses:
                continue

            eligible, reason = self._auto_gate(candidate)
            if not eligible:
                if reason == "stale_base_revision":
                    self._mark_stale(candidate, reason)
                    self.db.commit()
                continue

            if self._auto_budget_used(candidate.hypothesis_id) >= self.auto_per_hypothesis:
                continue

            result = self._apply_candidate(
                candidate,
                action="AUTO_APPLY",
                reason=(
                    "deterministic_auto_gate_passed: "
                    f"utility>={self.auto_min_utility:.2f}, LOW risk, full source coverage, no regression"
                ),
            )
            if result.applied:
                applied_hypotheses.add(int(candidate.hypothesis_id))
                results.append(result)
        return results

    def approve_candidate(self, candidate_id: int, *, reason: str = "human_approved") -> RevisionCommitResult:
        candidate = self.db.get(CandidateRevision, int(candidate_id))
        if candidate is None:
            raise ValueError(f"CR#{candidate_id} 不存在")
        eligible, gate_reason = self._human_gate(candidate)
        if not eligible:
            if gate_reason == "stale_base_revision":
                self._mark_stale(candidate, gate_reason)
                self.db.commit()
            raise ValueError(f"CR#{candidate_id} 不能人工批准：{gate_reason}")
        return self._apply_candidate(
            candidate,
            action="HUMAN_APPLY",
            reason=reason or "human_approved",
        )

    def reject_candidate(self, candidate_id: int, *, reason: str = "human_rejected") -> RevisionCommitResult:
        candidate = self.db.get(CandidateRevision, int(candidate_id))
        if candidate is None:
            raise ValueError(f"CR#{candidate_id} 不存在")
        if candidate.status in {"applied", "stale"}:
            raise ValueError(f"CR#{candidate_id} 当前状态为 {candidate.status}，不能拒绝")
        if candidate.status == "rejected":
            event = self.db.scalar(
                select(RevisionApplicationEvent).where(
                    RevisionApplicationEvent.candidate_id == candidate.id,
                    RevisionApplicationEvent.action == "HUMAN_REJECT",
                )
            )
            return RevisionCommitResult(event, candidate, False, "already_rejected")

        proposal = self.db.get(RevisionProposal, candidate.source_proposal_id)
        candidate.status = "rejected"
        if proposal is not None and proposal.status == "proposed":
            proposal.status = "rejected"
        event = RevisionApplicationEvent(
            hypothesis_id=candidate.hypothesis_id,
            from_revision=candidate.base_revision,
            to_revision=None,
            candidate_id=candidate.id,
            proposal_id=candidate.source_proposal_id,
            action="HUMAN_REJECT",
            risk_level=str(candidate.risk_level or ""),
            utility_score=float(candidate.utility_score or 0.0),
            reason=reason or "human_rejected",
            event_data={
                "evaluation_version": int(candidate.evaluation_version or 0),
                "fit_gain": float(candidate.fit_gain or 0.0),
            },
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return RevisionCommitResult(event, candidate, False, "rejected")

    def rollback(
        self,
        hypothesis_id: int,
        target_revision: int,
        *,
        reason: str = "human_rollback",
    ) -> RevisionCommitResult:
        hypothesis = self.db.get(StrategyHypothesis, int(hypothesis_id))
        if hypothesis is None:
            raise ValueError(f"H#{hypothesis_id} 不存在")
        target = self.hypothesis_service.get_revision(int(hypothesis_id), int(target_revision))
        if target is None:
            raise ValueError(f"H#{hypothesis_id}:r{target_revision} 不存在")

        from_revision = int(hypothesis.current_revision or hypothesis.revision or 1)
        if from_revision == int(target_revision):
            raise ValueError(f"H#{hypothesis_id} 当前已经是 r{target_revision}")

        try:
            self.hypothesis_service.activate_revision(
                int(hypothesis_id),
                int(target_revision),
                commit=False,
            )
            event = RevisionApplicationEvent(
                hypothesis_id=int(hypothesis_id),
                from_revision=from_revision,
                to_revision=int(target_revision),
                candidate_id=None,
                proposal_id=None,
                action="ROLLBACK",
                risk_level="",
                utility_score=0.0,
                reason=reason or "human_rollback",
                event_data={"immutable_history_preserved": True},
            )
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            return RevisionCommitResult(event, None, True, "rolled_back")
        except Exception:
            self.db.rollback()
            raise

    def list_events(self, *, limit: int = 50) -> list[RevisionApplicationEvent]:
        stmt = (
            select(RevisionApplicationEvent)
            .order_by(RevisionApplicationEvent.id.desc())
            .limit(max(1, int(limit)))
        )
        return list(self.db.scalars(stmt).all())
