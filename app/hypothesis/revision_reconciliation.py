from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hypothesis.revision_signal import RevisionSignalService
from app.models import (
    HypothesisRevision,
    RevisionApplicationEvent,
    RevisionSignal,
    StrategyHypothesis,
)


@dataclass(frozen=True)
class RevisionPressureSummary:
    hypothesis_id: int
    current_revision: int
    lineage: list[int]
    active_open: int
    ancestor_open: int
    resolved: int
    superseded: int
    inactive_branch_open: int
    invalid: int
    active_open_ids: list[int]
    ancestor_open_ids: list[int]
    resolved_ids: list[int]
    inactive_branch_open_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RevisionReconciliationService:
    """Phase 5.4 current-state semantics for RevisionSignals.

    RevisionSignal.resolution_status is historical metadata. A signal is only
    considered resolved for a *current head* when the revision that resolved it
    belongs to that head's lineage. This matters after rollback/branching.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.signal_service = RevisionSignalService(db)

    def lineage(self, hypothesis_id: int, revision_number: int) -> list[int]:
        revisions = list(
            self.db.scalars(
                select(HypothesisRevision).where(
                    HypothesisRevision.hypothesis_id == int(hypothesis_id)
                )
            ).all()
        )
        by_number = {int(item.revision_number): item for item in revisions}
        result: list[int] = []
        seen: set[int] = set()
        current = int(revision_number)
        while current not in seen and current in by_number:
            seen.add(current)
            result.append(current)
            parent = by_number[current].parent_revision
            if parent is None:
                break
            current = int(parent)
        return result

    def _valid_signal_ids(self, hypothesis_id: int) -> set[int]:
        return {
            int(item.id)
            for item in self.signal_service.list_all(
                hypothesis_id=int(hypothesis_id), valid_only=True
            )
        }

    def classify_signal(
        self,
        signal: RevisionSignal,
        *,
        current_revision: int,
        lineage: set[int],
        valid_signal_ids: set[int],
    ) -> str:
        if int(signal.id) not in valid_signal_ids:
            return "invalid"
        stored = str(signal.resolution_status or "open").lower()

        # Resolution/supersession are branch-local facts. If the revision that
        # retired the signal is not in the current lineage (for example after a
        # rollback to the ancestor), the original signal becomes open again.
        if stored == "superseded":
            superseded_by = signal.superseded_by_revision
            if superseded_by is not None and int(superseded_by) in lineage:
                return "superseded"

        resolved_by = signal.resolved_by_revision
        if stored == "resolved" and resolved_by is not None and int(resolved_by) in lineage:
            return "resolved"

        base = int(signal.base_revision or 1)
        if base == int(current_revision):
            return "active_open"
        if base in lineage:
            return "ancestor_open"
        return "inactive_branch_open"

    def pressure_summary(
        self,
        hypothesis_id: int,
        revision_number: int | None = None,
    ) -> RevisionPressureSummary:
        hypothesis = self.db.get(StrategyHypothesis, int(hypothesis_id))
        if hypothesis is None:
            raise ValueError(f"H#{hypothesis_id} 不存在")
        current = int(revision_number or hypothesis.current_revision or hypothesis.revision or 1)
        lineage_list = self.lineage(int(hypothesis_id), current)
        lineage = set(lineage_list)
        valid_ids = self._valid_signal_ids(int(hypothesis_id))
        signals = self.signal_service.list_all(
            hypothesis_id=int(hypothesis_id), valid_only=False
        )

        buckets: dict[str, list[int]] = {
            "active_open": [],
            "ancestor_open": [],
            "resolved": [],
            "superseded": [],
            "inactive_branch_open": [],
            "invalid": [],
        }
        for signal in signals:
            state = self.classify_signal(
                signal,
                current_revision=current,
                lineage=lineage,
                valid_signal_ids=valid_ids,
            )
            buckets[state].append(int(signal.id))

        return RevisionPressureSummary(
            hypothesis_id=int(hypothesis_id),
            current_revision=current,
            lineage=lineage_list,
            active_open=len(buckets["active_open"]),
            ancestor_open=len(buckets["ancestor_open"]),
            resolved=len(buckets["resolved"]),
            superseded=len(buckets["superseded"]),
            inactive_branch_open=len(buckets["inactive_branch_open"]),
            invalid=len(buckets["invalid"]),
            active_open_ids=buckets["active_open"],
            ancestor_open_ids=buckets["ancestor_open"],
            resolved_ids=buckets["resolved"],
            inactive_branch_open_ids=buckets["inactive_branch_open"],
        )

    def reconcile_application_event(
        self,
        event: RevisionApplicationEvent,
        *,
        commit: bool = True,
    ) -> list[int]:
        """Project one APPLY event onto source RevisionSignal resolution metadata."""
        if str(event.action) not in {"AUTO_APPLY", "HUMAN_APPLY"} or event.to_revision is None:
            return []
        event_data = event.event_data or {}
        signal_ids = event_data.get("resolved_signal_ids", []) if isinstance(event_data, dict) else []
        changed: list[int] = []
        for raw_id in signal_ids or []:
            try:
                signal_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            signal = self.db.get(RevisionSignal, signal_id)
            if signal is None or int(signal.hypothesis_id) != int(event.hypothesis_id):
                continue
            signal.resolution_status = "resolved"
            signal.resolved_by_candidate_id = event.candidate_id
            signal.resolved_by_revision = int(event.to_revision)
            signal.resolved_at = event.created_at or datetime.now()
            signal.resolution_note = f"resolved_by_RE#{event.id or 'pending'}_{event.action}"
            changed.append(signal_id)
        if commit and changed:
            self.db.commit()
        elif changed:
            self.db.flush()
        return sorted(set(changed))

    def reconcile_existing_events(self, *, commit: bool = True) -> int:
        """Idempotently project existing Phase 5.3 events after upgrading an old DB."""
        events = list(
            self.db.scalars(
                select(RevisionApplicationEvent)
                .where(RevisionApplicationEvent.action.in_({"AUTO_APPLY", "HUMAN_APPLY"}))
                .order_by(RevisionApplicationEvent.id.asc())
            ).all()
        )
        changed = 0
        for event in events:
            for signal_id in (event.event_data or {}).get("resolved_signal_ids", []) or []:
                try:
                    signal = self.db.get(RevisionSignal, int(signal_id))
                except (TypeError, ValueError):
                    continue
                if signal is None or int(signal.hypothesis_id) != int(event.hypothesis_id):
                    continue
                expected_revision = int(event.to_revision) if event.to_revision is not None else None
                if (
                    str(signal.resolution_status or "open") == "resolved"
                    and signal.resolved_by_revision == expected_revision
                    and signal.resolved_by_candidate_id == event.candidate_id
                ):
                    continue
                self.reconcile_application_event(event, commit=False)
                changed += 1
                break
        if commit and changed:
            self.db.commit()
        return changed
