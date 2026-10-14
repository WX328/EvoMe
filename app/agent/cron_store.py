"""S12: durable, opt-in schedule definitions, without model-controlled execution.

Supported UTC-aware one-shot timestamps and a bounded 5-field cron subset:
  minute hour * * weekday       e.g. "*/5 * * * *", "30 9 * * 1-5".
Cron clocks use a fixed explicit UTC offset, not DST-aware named time zones.
Schedule files are private, atomically written, and never used as code.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


_ID = re.compile(r"cron_[0-9a-f]{8}\Z")
_OFFSET = re.compile(r"([+-])(\d{2}):(\d{2})\Z")
_MAX_SCHEDULES = 50
_MAX_BYTES = 8000


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("S12 requires a timezone-aware timestamp")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="seconds")


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("invalid timestamp")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (TypeError, ValueError) as exc:
        raise ValueError("use ISO-8601 timestamp with UTC offset, e.g. 2026-10-15T09:00:00+08:00") from exc


def _fixed_offset(value: str) -> timezone:
    if value == "Z" or value == "+00:00":
        return timezone.utc
    match = _OFFSET.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("UTC offset must be +HH:MM or -HH:MM")
    sign, hours, minutes = match.groups()
    hour, minute = int(hours), int(minutes)
    if hour > 14 or minute > 59 or hour == 14 and minute != 0:
        raise ValueError("invalid UTC offset")
    seconds = (hour * 60 + minute) * 60 * (1 if sign == "+" else -1)
    return timezone(timedelta(seconds=seconds))


def _field(text: str, maximum: int, *, minimum: int = 0) -> frozenset[int]:
    if not text or len(text) > 50:
        raise ValueError("invalid cron field")
    allowed: set[int] = set()
    for segment in text.split(","):
        if not segment:
            raise ValueError("empty cron component")
        step = 1
        if "/" in segment:
            base, divisor = segment.split("/", 1)
            if not divisor.isdigit() or not 1 <= int(divisor) <= maximum + 1:
                raise ValueError("invalid cron step")
            step = int(divisor)
            if base not in {"*"} and "-" not in base:
                raise ValueError("cron step must use * or a range")
        else:
            base = segment
        if base == "*":
            low, high = minimum, maximum
        elif "-" in base:
            parts = base.split("-", 1)
            if len(parts) != 2 or not all(x.isdigit() for x in parts):
                raise ValueError("invalid cron range")
            low, high = map(int, parts)
        elif base.isdigit():
            low = high = int(base)
        else:
            raise ValueError("invalid cron value")
        if not (minimum <= low <= high <= maximum):
            raise ValueError("cron field out of range")
        allowed.update(range(low, high + 1, step))
    return frozenset(allowed)


def _cron_spec(expression: str) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    if not isinstance(expression, str):
        raise ValueError("cron expression must be text")
    parts = expression.split()
    if len(parts) != 5 or parts[2:4] != ["*", "*"]:
        raise ValueError("S12 cron supports: minute hour * * weekday (UTC offset separately)")
    return _field(parts[0], 59), _field(parts[1], 23), _field(parts[4], 6)


def next_cron(expression: str, after: datetime, offset: str) -> datetime:
    minute, hour, weekday = _cron_spec(expression)
    zone = _fixed_offset(offset)
    local = _utc(after).astimezone(zone)
    local = local.replace(second=0, microsecond=0) + timedelta(minutes=1)
    # At most 366 days; very sparse expressions get a finite failure.
    for _ in range(366 * 24 * 60):
        cron_weekday = (local.weekday() + 1) % 7  # Sunday=0, Monday=1
        if local.minute in minute and local.hour in hour and cron_weekday in weekday:
            return local.astimezone(timezone.utc)
        local += timedelta(minutes=1)
    raise ValueError("no cron occurrence in next 366 days")


@dataclass(frozen=True)
class ScheduleRecord:
    id: str
    kind: str                       # at / cron
    title: str
    expression: str                 # ISO-8601 for "at"; cron for recurring
    utc_offset: str
    next_run_at: str
    status: str                     # active / paused / completed / missed / canceled
    max_runs: int
    run_count: int
    last_run_at: str = ""
    last_task_id: str = ""
    last_error: str = ""
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()

    @classmethod
    def parse(cls, obj: Any, *, expected_id: str) -> "ScheduleRecord":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(obj, dict) or set(obj) != fields:
            raise ValueError("corrupt S12 schedule schema")
        if obj["id"] != expected_id or not _ID.fullmatch(expected_id):
            raise ValueError("schedule ID/filename mismatch")
        if obj["kind"] not in {"at", "cron"} or obj["status"] not in {
            "active", "paused", "completed", "missed", "canceled",
        }:
            raise ValueError("invalid schedule kind/status")
        title = obj["title"]
        if not isinstance(title, str) or not 1 <= len(title) <= 200:
            raise ValueError("invalid schedule task title")
        for key in ("expression", "utc_offset", "next_run_at", "last_run_at", "last_task_id", "last_error"):
            if not isinstance(obj[key], str) or len(obj[key]) > 700:
                raise ValueError("invalid schedule string")
        if type(obj["max_runs"]) is not int or type(obj["run_count"]) is not int:
            raise ValueError("invalid run count")
        if not 1 <= obj["max_runs"] <= 30 or not 0 <= obj["run_count"] <= obj["max_runs"]:
            raise ValueError("invalid schedule run bounds")
        if type(obj["version"]) is not int or obj["version"] < 1:
            raise ValueError("invalid schedule version")
        _fixed_offset(obj["utc_offset"])
        if obj["kind"] == "cron":
            _cron_spec(obj["expression"])
        else:
            _parse_time(obj["expression"])
        _parse_time(obj["next_run_at"])
        if obj["last_run_at"]:
            _parse_time(obj["last_run_at"])
        if obj["status"] == "completed" and obj["run_count"] != obj["max_runs"]:
            raise ValueError("completed schedule with remaining executions")
        return cls(**obj)


class ScheduleStore:
    """Single-process local schedule store; replay of overdue triggers is forbidden."""

    def __init__(self, *, workspace_root: str | Path, directory: str = ".schedules") -> None:
        self.workspace = Path(workspace_root).resolve()
        folder = Path(directory)
        if not directory or folder.is_absolute() or ".." in folder.parts or folder == Path("."):
            raise ValueError("schedule directory must be workspace-relative")
        self.root = self.workspace / folder
        self._lock = RLock()
        self._guard()

    def _guard(self) -> None:
        part = self.root
        while part != self.workspace:
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                raise PermissionError("S12 schedule store must not contain symlinks or junctions")
            part = part.parent
            if not part.is_relative_to(self.workspace):
                raise PermissionError("S12 schedule store escaped workspace")
        if not self.root.resolve().is_relative_to(self.workspace):
            raise PermissionError("S12 schedule store escaped workspace")

    def _path(self, id_: str) -> Path:
        if not isinstance(id_, str) or not _ID.fullmatch(id_):
            raise ValueError("schedule id must be cron_XXXXXXXX")
        self._guard()
        path = self.root / f"{id_}.json"
        if path.is_symlink():
            raise PermissionError("symlinked schedule record forbidden")
        return path

    def _all(self) -> dict[str, ScheduleRecord]:
        self._guard()
        if not self.root.is_dir():
            return {}
        rows: dict[str, ScheduleRecord] = {}
        for path in sorted(self.root.iterdir()):
            if path.suffix != ".json":
                continue
            if not _ID.fullmatch(path.stem) or path.is_symlink() or path.stat().st_size > _MAX_BYTES:
                raise ValueError(f"unsafe schedule record {path.name}")
            if len(rows) >= _MAX_SCHEDULES:
                raise ValueError("S12 maximum schedule count exceeded")
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError(f"corrupt schedule JSON: {path.name}") from exc
            rows[path.stem] = ScheduleRecord.parse(doc, expected_id=path.stem)
        return rows

    def _save(self, record: ScheduleRecord) -> None:
        self._guard()
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(record.id)
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n"
        if len(payload.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("schedule document exceeds byte limit")
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.root,
                prefix=".s12-", suffix=".tmp", delete=False,
            ) as handle:
                tmp = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._guard()
            if target.is_symlink():
                raise PermissionError("schedule record became symlink")
            os.replace(tmp, target)
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [row.to_dict() for row in self._all().values()]

    def get(self, schedule_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get(schedule_id).to_dict()

    def _get(self, schedule_id: str) -> ScheduleRecord:
        rows = self._all()
        if schedule_id not in rows:
            raise KeyError(f"unknown schedule: {schedule_id}")
        return rows[schedule_id]

    def create(
        self, *, title: str, kind: str, expression: str,
        utc_offset: str = "+00:00", max_runs: int = 1,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
            raise ValueError("task title must be 1-200 characters")
        if type(max_runs) is not int or not 1 <= max_runs <= 30:
            raise ValueError("max_runs must be between 1 and 30")
        now = _utc(now or datetime.now(timezone.utc))
        if kind == "at":
            if max_runs != 1:
                raise ValueError("one-time schedule must have max_runs=1")
            _fixed_offset(utc_offset)
            due = _parse_time(expression)
            if due <= now:
                raise ValueError("one-time schedule must be in the future")
        elif kind == "cron":
            _cron_spec(expression)
            _fixed_offset(utc_offset)
            due = next_cron(expression, now, utc_offset)
        else:
            raise ValueError("kind must be at or cron")
        with self._lock:
            rows = self._all()
            if len(rows) >= _MAX_SCHEDULES:
                raise ValueError("schedule capacity exhausted")
            key = f"cron_{uuid4().hex[:8]}"
            while key in rows:
                key = f"cron_{uuid4().hex[:8]}"
            record = ScheduleRecord(
                id=key, title=title.strip(), kind=kind,
                expression=expression.strip(), utc_offset=utc_offset,
                next_run_at=_stamp(due), status="active",
                max_runs=max_runs, run_count=0,
            )
            self._save(record)
            return record.to_dict()

    def due(self, now: datetime | None = None) -> list[ScheduleRecord]:
        now = _utc(now or datetime.now(timezone.utc))
        with self._lock:
            return sorted(
                [
                    row for row in self._all().values()
                    if row.status == "active" and _parse_time(row.next_run_at) <= now
                ],
                key=lambda row: (row.next_run_at, row.id),
            )[:10]

    def consume(self, schedule_id: str, *, now: datetime) -> dict[str, Any]:
        """Advance the durable cursor BEFORE any external job side effect."""
        now = _utc(now)
        with self._lock:
            old = self._get(schedule_id)
            if old.status != "active" or _parse_time(old.next_run_at) > now:
                raise ValueError("schedule is not due")
            run_count = old.run_count + 1
            finished = run_count == old.max_runs
            next_time = (
                old.next_run_at if finished
                else _stamp(next_cron(old.expression, now, old.utc_offset))
            )
            updated = replace(
                old, run_count=run_count,
                status="completed" if finished else "active",
                next_run_at=next_time, last_run_at=_stamp(now),
                last_error="", version=old.version + 1,
            )
            self._save(updated)
            return updated.to_dict()

    def record_result(self, schedule_id: str, *, task_id: str = "", error: str = "") -> None:
        if len(task_id) > 80 or len(error) > 650:
            raise ValueError("schedule result overflow")
        with self._lock:
            old = self._get(schedule_id)
            self._save(replace(
                old, last_task_id=task_id, last_error=error,
                version=old.version + 1,
            ))

    def pause(self, schedule_id: str) -> dict[str, Any]:
        with self._lock:
            old = self._get(schedule_id)
            if old.status != "active":
                raise ValueError("only active schedules can be paused")
            new = replace(old, status="paused", version=old.version + 1)
            self._save(new)
            return new.to_dict()

    def resume(self, schedule_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        now = _utc(now or datetime.now(timezone.utc))
        with self._lock:
            old = self._get(schedule_id)
            if old.status != "paused":
                raise ValueError("only paused schedules can resume")
            if old.kind == "at":
                when = _parse_time(old.expression)
                if when <= now:
                    raise ValueError("missed one-time schedule cannot resume")
            else:
                when = next_cron(old.expression, now, old.utc_offset)
            new = replace(old, status="active", next_run_at=_stamp(when), version=old.version+1)
            self._save(new)
            return new.to_dict()

    def cancel(self, schedule_id: str) -> dict[str, Any]:
        with self._lock:
            old = self._get(schedule_id)
            if old.status not in {"active", "paused"}:
                raise ValueError("schedule already terminal")
            new = replace(old, status="canceled", version=old.version + 1)
            self._save(new)
            return new.to_dict()

    def skip_missed_on_start(self, now: datetime | None = None) -> int:
        """Never auto-replay triggers whose scheduled time passed while offline."""
        now = _utc(now or datetime.now(timezone.utc))
        skipped = 0
        with self._lock:
            for old in self._all().values():
                if old.status != "active" or _parse_time(old.next_run_at) > now:
                    continue
                if old.kind == "at":
                    new = replace(old, status="missed", last_error="missed_while_offline",
                                  version=old.version + 1)
                else:
                    next_time = next_cron(old.expression, now, old.utc_offset)
                    new = replace(old, next_run_at=_stamp(next_time),
                                  last_error="older_ticks_skipped_after_restart",
                                  version=old.version+1)
                self._save(new)
                skipped += 1
        return skipped
