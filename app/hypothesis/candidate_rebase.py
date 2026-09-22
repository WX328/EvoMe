from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.ancestor_revalidation import CURRENT_REVALIDATION_VERSION
from app.hypothesis.candidate_revision import CandidateRevisionService
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.hypothesis.revision_signal import RevisionSignalService
from app.models import (
    CandidateRebaseRecord,
    CandidateRevision,
    HypothesisRevision,
    RevisionAnalysisRecord,
    RevisionProposal,
    RevisionSignal,
    RevisionSignalRevalidation,
    StrategyHypothesis,
)


@dataclass(frozen=True)
class CandidateRebaseResult:
    record: CandidateRebaseRecord
    candidate: CandidateRevision | None
    llm_called: bool


class CandidateRebaseService:
    """Three-way rebase for a stale CandidateRevision.

    The old semantic direction is reused only when:
    1) every source signal was revalidated as STILL_APPLIES on the current head;
    2) the proposal's target field is unchanged between the stale base and head.

    A clean rebase creates a fresh analysis/proposal on the current head and then
    reruns CandidateRevision validation. Same-field changes are treated as a
    deterministic conflict and are left for normal current-head analysis.
    """

    def __init__(
        self,
        db: Session,
        llm,
        *,
        max_evidence: int = 8,
    ) -> None:
        self.db = db
        self.llm = llm
        self.max_evidence = max(1, int(max_evidence))
        self.reconciliation = RevisionReconciliationService(db)
        self.signal_service = RevisionSignalService(db)

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _field_value(revision: HypothesisRevision, proposal: RevisionProposal) -> tuple[bool, Any]:
        dimension = str(proposal.dimension).upper()
        field = str(proposal.target_field)
        if dimension == "CONTEXT":
            data = revision.when_data or {}
        elif dimension == "ACTION":
            data = revision.do_data or {}
        elif dimension == "OUTCOME":
            data = revision.expect_data or {}
        else:
            return False, None
        return field in data, copy.deepcopy(data.get(field))

    def _existing_record(
        self,
        source_candidate_id: int,
        target_revision: int,
    ) -> CandidateRebaseRecord | None:
        return self.db.scalar(
            select(CandidateRebaseRecord).where(
                CandidateRebaseRecord.source_candidate_id == int(source_candidate_id),
                CandidateRebaseRecord.target_revision == int(target_revision),
            )
        )

    def _source_revalidations(
        self,
        proposal: RevisionProposal,
        target_revision: int,
    ) -> tuple[list[int], list[RevisionSignalRevalidation], list[int]]:
        source_ids = [int(x) for x in (proposal.source_signal_ids or []) if x is not None]
        if not source_ids:
            return [], [], []
        rows = list(
            self.db.scalars(
                select(RevisionSignalRevalidation).where(
                    RevisionSignalRevalidation.source_signal_id.in_(source_ids),
                    RevisionSignalRevalidation.target_revision == int(target_revision),
                    RevisionSignalRevalidation.evaluation_version == CURRENT_REVALIDATION_VERSION,
                )
            ).all()
        )
        by_source = {int(item.source_signal_id): item for item in rows}
        ordered = [by_source[item] for item in source_ids if item in by_source]
        missing = [item for item in source_ids if item not in by_source]
        return source_ids, ordered, missing

    @staticmethod
    def _target_signal_ids(records: list[RevisionSignalRevalidation]) -> list[int]:
        result: list[int] = []
        for item in records:
            if str(item.outcome) != "STILL_APPLIES" or item.rebased_signal_id is None:
                continue
            result.append(int(item.rebased_signal_id))
        return result

    def _upsert_record(
        self,
        *,
        candidate: CandidateRevision,
        proposal: RevisionProposal,
        target_revision: int,
        status: str,
        source_signal_ids: list[int],
        target_signal_ids: list[int],
        reason: str,
        rebase_data: dict[str, Any] | None = None,
    ) -> CandidateRebaseRecord:
        record = self._existing_record(int(candidate.id), int(target_revision))
        if record is None:
            record = CandidateRebaseRecord(
                hypothesis_id=int(candidate.hypothesis_id),
                source_candidate_id=int(candidate.id),
                source_proposal_id=int(proposal.id),
                from_revision=int(candidate.base_revision),
                target_revision=int(target_revision),
            )
            self.db.add(record)
        record.status = str(status)
        record.source_signal_ids = list(source_signal_ids)
        record.target_signal_ids = list(target_signal_ids)
        record.reason = str(reason)
        record.rebase_data = dict(rebase_data or {})
        self.db.flush()
        return record

    def rebase_candidate(self, candidate_id: int) -> CandidateRebaseResult:
        candidate = self.db.get(CandidateRevision, int(candidate_id))
        if candidate is None:
            raise ValueError(f"CR#{candidate_id} 不存在")
        if str(candidate.status) != "stale":
            raise ValueError(f"CR#{candidate_id} 当前状态为 {candidate.status}，只有 stale Candidate 可 rebase")

        hypothesis = self.db.get(StrategyHypothesis, int(candidate.hypothesis_id))
        if hypothesis is None:
            raise ValueError(f"H#{candidate.hypothesis_id} 不存在")
        target_revision_number = int(hypothesis.current_revision or hypothesis.revision or 1)
        if target_revision_number == int(candidate.base_revision):
            raise ValueError("Candidate base 已是 current revision，无需 rebase")

        lineage = self.reconciliation.lineage(int(hypothesis.id), target_revision_number)
        if int(candidate.base_revision) not in set(lineage):
            raise ValueError(
                f"CR#{candidate.id} base r{candidate.base_revision} 不在 current r{target_revision_number} lineage 中"
            )

        old_proposal = self.db.get(RevisionProposal, int(candidate.source_proposal_id))
        if old_proposal is None:
            raise ValueError(f"RP#{candidate.source_proposal_id} 不存在")

        existing_record = self._existing_record(int(candidate.id), target_revision_number)
        if (
            existing_record is not None
            and existing_record.status == "rebased"
            and existing_record.rebased_candidate_id is not None
        ):
            rebased_candidate = self.db.get(CandidateRevision, int(existing_record.rebased_candidate_id))
            return CandidateRebaseResult(existing_record, rebased_candidate, llm_called=False)

        source_ids, revalidations, missing = self._source_revalidations(
            old_proposal, target_revision_number
        )
        if not source_ids:
            record = self._upsert_record(
                candidate=candidate,
                proposal=old_proposal,
                target_revision=target_revision_number,
                status="not_ready",
                source_signal_ids=[],
                target_signal_ids=[],
                reason="source_proposal_has_no_signals",
            )
            self.db.commit()
            return CandidateRebaseResult(record, None, llm_called=False)

        non_applying = [
            item
            for item in revalidations
            if str(item.outcome) != "STILL_APPLIES" or item.rebased_signal_id is None
        ]
        target_signal_ids = self._target_signal_ids(revalidations)
        if missing or non_applying or len(target_signal_ids) != len(source_ids):
            record = self._upsert_record(
                candidate=candidate,
                proposal=old_proposal,
                target_revision=target_revision_number,
                status="not_ready",
                source_signal_ids=source_ids,
                target_signal_ids=target_signal_ids,
                reason="source_signals_not_fully_revalidated_as_still_applies",
                rebase_data={
                    "missing_revalidation_signal_ids": missing,
                    "non_applying": [
                        {
                            "source_signal_id": int(item.source_signal_id),
                            "outcome": str(item.outcome),
                        }
                        for item in non_applying
                    ],
                },
            )
            self.db.commit()
            return CandidateRebaseResult(record, None, llm_called=False)

        # If new current-head evidence has appeared in the same semantic group,
        # the stale proposal is no longer the whole story. Do not silently reuse
        # its old semantic analysis; let the normal current-head Analyzer judge
        # the complete group instead.
        current_group_signals = [
            item
            for item in self.signal_service.list_all(
                hypothesis_id=int(hypothesis.id),
                base_revision=target_revision_number,
                valid_only=True,
            )
            if str(item.dimension) == str(old_proposal.dimension)
            and str(item.operation) == str(old_proposal.operation)
            and str(item.target_field) == str(old_proposal.target_field)
            and str(item.resolution_status or "open").lower() == "open"
        ]
        current_group_ids = sorted(int(item.id) for item in current_group_signals)
        extra_group_ids = sorted(set(current_group_ids) - set(target_signal_ids))
        if extra_group_ids:
            record = self._upsert_record(
                candidate=candidate,
                proposal=old_proposal,
                target_revision=target_revision_number,
                status="needs_fresh_analysis",
                source_signal_ids=source_ids,
                target_signal_ids=target_signal_ids,
                reason="new_same_target_signals_require_fresh_analysis",
                rebase_data={
                    "current_group_signal_ids": current_group_ids,
                    "extra_current_signal_ids": extra_group_ids,
                },
            )
            self.db.commit()
            return CandidateRebaseResult(record, None, llm_called=False)

        old_base = self.db.scalar(
            select(HypothesisRevision).where(
                HypothesisRevision.hypothesis_id == int(hypothesis.id),
                HypothesisRevision.revision_number == int(candidate.base_revision),
            )
        )
        current = self.db.scalar(
            select(HypothesisRevision).where(
                HypothesisRevision.hypothesis_id == int(hypothesis.id),
                HypothesisRevision.revision_number == target_revision_number,
            )
        )
        if old_base is None or current is None:
            raise ValueError("缺少 Candidate rebase 所需的 revision snapshot")

        if str(old_proposal.dimension).upper() == "ABSTRACTION":
            record = self._upsert_record(
                candidate=candidate,
                proposal=old_proposal,
                target_revision=target_revision_number,
                status="conflict",
                source_signal_ids=source_ids,
                target_signal_ids=target_signal_ids,
                reason="abstraction_rebase_requires_fresh_semantic_analysis",
            )
            self.db.commit()
            return CandidateRebaseResult(record, None, llm_called=False)

        old_present, old_value = self._field_value(old_base, old_proposal)
        current_present, current_value = self._field_value(current, old_proposal)
        if old_present != current_present or old_value != current_value:
            record = self._upsert_record(
                candidate=candidate,
                proposal=old_proposal,
                target_revision=target_revision_number,
                status="conflict",
                source_signal_ids=source_ids,
                target_signal_ids=target_signal_ids,
                reason="target_field_changed_since_stale_base",
                rebase_data={
                    "target_field": str(old_proposal.target_field),
                    "old_base_value": old_value,
                    "current_value": current_value,
                },
            )
            self.db.commit()
            return CandidateRebaseResult(record, None, llm_called=False)

        target_signals = list(
            self.db.scalars(select(RevisionSignal).where(RevisionSignal.id.in_(target_signal_ids))).all()
        )
        independent_count = len(
            {int(item.interaction_id) for item in target_signals if item.interaction_id is not None}
        )
        revalidation_confidences = [float(item.confidence) for item in revalidations]
        direction_confidence = min(
            float(old_proposal.direction_confidence),
            mean(revalidation_confidences) if revalidation_confidences else 0.5,
        )

        fingerprint = self._fingerprint(
            {
                "kind": "candidate_rebase",
                "source_candidate_id": int(candidate.id),
                "source_proposal_id": int(old_proposal.id),
                "from_revision": int(candidate.base_revision),
                "target_revision": target_revision_number,
                "target_signal_ids": sorted(target_signal_ids),
                "dimension": str(old_proposal.dimension),
                "operation": str(old_proposal.operation),
                "target_field": str(old_proposal.target_field),
            }
        )
        analysis = self.db.scalar(
            select(RevisionAnalysisRecord).where(
                RevisionAnalysisRecord.candidate_fingerprint == fingerprint
            )
        )
        if analysis is None:
            analysis = RevisionAnalysisRecord(
                hypothesis_id=int(hypothesis.id),
                base_revision=target_revision_number,
                dimension=str(old_proposal.dimension),
                operation=str(old_proposal.operation),
                target_field=str(old_proposal.target_field),
                signal_ids=sorted(target_signal_ids),
                candidate_fingerprint=fingerprint,
                consistent=True,
                direction_confidence=float(direction_confidence),
                reason=(
                    f"clean_rebase_from_CR#{candidate.id}/RP#{old_proposal.id}: all source signals "
                    f"revalidated STILL_APPLIES on r{target_revision_number}; target field unchanged"
                ),
            )
            self.db.add(analysis)
            self.db.flush()

        new_proposal = self.db.scalar(
            select(RevisionProposal).where(RevisionProposal.analysis_id == int(analysis.id))
        )
        if new_proposal is None:
            new_proposal = RevisionProposal(
                analysis_id=int(analysis.id),
                hypothesis_id=int(hypothesis.id),
                base_revision=target_revision_number,
                dimension=str(old_proposal.dimension),
                operation=str(old_proposal.operation),
                target_field=str(old_proposal.target_field),
                before_data=(copy.deepcopy(current_value) if current_present else None),
                proposed_after_data=copy.deepcopy(old_proposal.proposed_after_data),
                reason=(
                    f"rebased_from_RP#{old_proposal.id} after ancestor signal revalidation; "
                    "semantic direction inherited, candidate must be re-evaluated"
                ),
                source_signal_ids=sorted(target_signal_ids),
                independent_signal_count=independent_count,
                direction_confidence=float(direction_confidence),
                status="proposed",
            )
            self.db.add(new_proposal)
            self.db.flush()

        record = self._upsert_record(
            candidate=candidate,
            proposal=old_proposal,
            target_revision=target_revision_number,
            status="rebasing",
            source_signal_ids=source_ids,
            target_signal_ids=target_signal_ids,
            reason="clean_three_way_rebase_ready_for_candidate_validation",
            rebase_data={
                "target_field": str(old_proposal.target_field),
                "source_analysis_id": int(old_proposal.analysis_id),
                "revalidation_record_ids": [int(item.id) for item in revalidations],
            },
        )
        record.rebased_analysis_id = int(analysis.id)
        record.rebased_proposal_id = int(new_proposal.id)
        self.db.flush()

        candidate_service = CandidateRevisionService(
            self.db,
            self.llm,
            max_evidence=self.max_evidence,
        )
        evaluated = candidate_service.evaluate_proposal(new_proposal)
        rebased_candidate = evaluated.candidate
        data = dict(rebased_candidate.evaluation_data or {})
        data["rebase"] = {
            "source_candidate_id": int(candidate.id),
            "source_proposal_id": int(old_proposal.id),
            "candidate_rebase_record_id": int(record.id),
            "from_revision": int(candidate.base_revision),
            "target_revision": target_revision_number,
            "source_signal_ids": source_ids,
            "target_signal_ids": target_signal_ids,
        }
        rebased_candidate.evaluation_data = data

        record.status = "rebased"
        record.rebased_candidate_id = int(rebased_candidate.id)
        record.reason = "clean_rebase_completed_and_candidate_re_evaluated"
        old_proposal.status = "stale"
        self.db.commit()
        self.db.refresh(record)
        self.db.refresh(rebased_candidate)
        return CandidateRebaseResult(record, rebased_candidate, llm_called=evaluated.llm_called)

    def list_records(self, *, limit: int = 100) -> list[CandidateRebaseRecord]:
        stmt = (
            select(CandidateRebaseRecord)
            .order_by(CandidateRebaseRecord.created_at.desc(), CandidateRebaseRecord.id.desc())
            .limit(max(1, int(limit)))
        )
        return list(self.db.scalars(stmt).all())
