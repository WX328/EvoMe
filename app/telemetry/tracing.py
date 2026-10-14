"""Bounded metadata-only tracing. No prompts, tool arguments or outputs are saved."""
from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import RLock
from uuid import uuid4

from app.path_safety import resolved_path

_SAFE = re.compile(r"[A-Za-z0-9_.:/-]{1,100}\Z")
_CURRENT: ContextVar[TraceContext | None] = ContextVar("evome_trace", default=None)


@dataclass(frozen=True)
class TraceContext:
    recorder: TraceRecorder
    trace_id: str
    span_id: str


def current_trace() -> TraceContext | None:
    return _CURRENT.get()


def _label(value: object) -> str:
    text = str(value)
    return text if _SAFE.fullmatch(text) else "redacted"


class TraceRecorder:
    def __init__(self, workspace: str | Path, *, prices: dict | None = None,
                 read_only: bool = False, max_events: int = 20000, cleanup_interval: int = 256):
        self.workspace = resolved_path(Path(workspace))
        self.root = self.workspace / ".traces"
        self.path = self.root / "events.sqlite3"
        self._lock = RLock()
        self._writer = None
        self._writer_identity = None
        self._closed = False
        self.read_only = read_only
        self.max_events = max(1, min(20000, int(max_events)))
        self.cleanup_interval = max(1, min(256, int(cleanup_interval)))
        self._writes_since_cleanup = 0
        self.dropped_events = 0
        self.prices = {}
        for model, rates in (prices or {}).items():
            if (not isinstance(rates, (list, tuple)) or len(rates) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in rates)):
                raise ValueError("token prices require three nonnegative USD-per-million rates")
            self.prices[str(model)] = tuple(rates)

    def _guard(self):
        for path in (self.root, self.path, Path(str(self.path) + "-journal"),
                     Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise PermissionError("unsafe trace storage path")
            if not resolved_path(path).is_relative_to(self.workspace):
                raise PermissionError("trace storage escaped workspace")

    def _connect(self, *, write=False):
        self._guard()
        if write:
            self.root.mkdir(exist_ok=True)
            self._guard()
            conn = sqlite3.connect(self.path, timeout=1, check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, timestamp REAL, pid INTEGER,
                trace_id TEXT, span_id TEXT, parent_id TEXT, component TEXT,
                operation TEXT, reference TEXT, phase TEXT, status TEXT,
                elapsed_ms REAL, error_type TEXT, prompt_tokens INTEGER,
                completion_tokens INTEGER, cached_tokens INTEGER, cost_usd REAL)""")
                conn.execute("CREATE INDEX IF NOT EXISTS trace_events ON events(trace_id, seq)")
                with conn:
                    self._prune(conn)
            except BaseException:
                conn.close()
                raise
        else:
            if not self.path.is_file():
                return None
            conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        return conn

    def _prune(self, conn):
        # OFFSET also works when event sequence numbers have gaps.
        conn.execute("DELETE FROM events WHERE seq < (SELECT seq FROM events "
                     "ORDER BY seq DESC LIMIT 1 OFFSET ?)", (self.max_events - 1,))

    def _discard_writer(self):
        conn, self._writer = self._writer, None
        self._writer_identity = None
        if conn is not None:
            conn.close()

    def _write_connection(self):
        # Revalidate on every write, including cached connections. A replaced
        # database must not leave us writing into an unlinked/stale handle.
        self._guard()
        if self._writer is not None:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self._writer_identity:
                self._discard_writer()
        if self._writer is None:
            self._writer = self._connect(write=True)
            stat = self.path.stat()
            self._writer_identity = (stat.st_dev, stat.st_ino)
            self._writes_since_cleanup = 0
        return self._writer

    def close(self):
        """Release file handles after workers stop; repeated calls are safe."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._writer is not None:
                    self._guard()
                    stat = self.path.stat()
                    if (stat.st_dev, stat.st_ino) == self._writer_identity:
                        with self._writer:
                            self._prune(self._writer)
            except (OSError, ValueError, sqlite3.Error):
                self.dropped_events += 1
            finally:
                self._discard_writer()

    def _event(self, context, *, parent_id="", component, operation, reference="",
               phase, status="", elapsed_ms=0, error_type="", prompt_tokens=0,
               completion_tokens=0, cached_tokens=0, cost_usd=None):
        # A failed diagnostic sink must never replay an operation or approve it.
        try:
            with self._lock:
                if self._closed or self.read_only:
                    self.dropped_events += 1
                    return
                conn = self._write_connection()
                with conn:
                    conn.execute("INSERT INTO events VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                            time.time(), os.getpid(), context.trace_id, context.span_id,
                            parent_id, _label(component), _label(operation), _label(reference) if reference else "",
                            phase, _label(status) if status else "", round(elapsed_ms, 3),
                            _label(error_type) if error_type else "", int(prompt_tokens),
                            int(completion_tokens), int(cached_tokens), cost_usd))
                    if self._writes_since_cleanup + 1 >= self.cleanup_interval:
                        self._prune(conn)
                        self._writes_since_cleanup = 0
                    else:
                        self._writes_since_cleanup += 1
        except (OSError, ValueError, sqlite3.Error):
            with self._lock:
                self.dropped_events += 1
                self._discard_writer()

    @contextmanager
    def span(self, component, operation, *, reference="", trace_id="", parent=None):
        previous = parent or current_trace()
        context = TraceContext(self, trace_id or (previous.trace_id if previous else uuid4().hex), uuid4().hex)
        parent_id = previous.span_id if previous and previous.trace_id == context.trace_id else ""
        token = _CURRENT.set(context)
        started = time.perf_counter()
        outcome = {"status": "completed"}
        self._event(context, parent_id=parent_id, component=component, operation=operation,
                    reference=reference, phase="start", status="running")
        try:
            yield outcome
        except BaseException as exc:
            outcome.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            self._event(context, parent_id=parent_id, component=component, operation=operation,
                        reference=reference, phase="end", elapsed_ms=(time.perf_counter()-started)*1000,
                        status=outcome["status"], error_type=outcome.get("error_type", ""))
            _CURRENT.reset(token)

    def usage(self, *, purpose, model, prompt_tokens, completion_tokens, cached_tokens):
        parent = current_trace()
        if parent is None:
            return
        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        cached_tokens = min(prompt_tokens, max(0, int(cached_tokens)))
        rates = self.prices.get(model)
        cost = (((prompt_tokens-cached_tokens)*rates[0] + cached_tokens*rates[1]
                 + completion_tokens*rates[2]) / 1_000_000) if rates is not None else None
        context = TraceContext(self, parent.trace_id, uuid4().hex)
        self._event(context, parent_id=parent.span_id, component="llm", operation=purpose,
                    reference=model, phase="usage", status="recorded", prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, cached_tokens=cached_tokens, cost_usd=cost)

    def goal_state(self, status):
        parent = current_trace()
        if parent is not None:
            self._event(TraceContext(self, parent.trace_id, uuid4().hex),
                        parent_id=parent.span_id, component="goal", operation="state",
                        phase="goal", status=status)

    def events(self, trace_id="", *, limit=500):
        if trace_id and not re.fullmatch(r"[0-9a-f]{32}", trace_id):
            raise ValueError("trace id must contain 32 lowercase hexadecimal characters")
        with self._lock:
            conn = self._connect()
            if conn is None:
                return []
            try:
                where, args = ("WHERE trace_id=?", [trace_id]) if trace_id else ("", [])
                rows = conn.execute(f"SELECT * FROM events {where} ORDER BY seq DESC LIMIT ?",
                                    [*args, max(1, min(self.max_events, int(limit)))])
                return [dict(row) for row in reversed(rows.fetchall())]
            finally:
                conn.close()

    def traces(self, limit=30):
        events = self.events(limit=20000)
        grouped = {}
        for row in events:
            item = grouped.setdefault(row["trace_id"], {"trace_id": row["trace_id"],
                "timestamp": row["timestamp"], "status": "unfinished", "tokens": 0,
                "estimated_usd": 0.0, "cost_known": True, "failures": 0, "usage_calls": 0,
                "spans": {}, "goal_status": None, "root_status": None,
                "subtask_failures": 0, "parents": set()})
            item["timestamp"] = row["timestamp"]
            if row["parent_id"]:
                item["parents"].add(row["parent_id"])
            if row["phase"] == "start":
                item["spans"][row["span_id"]] = {"start": row, "end": None}
            elif row["phase"] == "end":
                item["spans"].setdefault(row["span_id"], {"start": None})["end"] = row
                failed = row["status"] in {"error", "failed", "goal_failed", "goal_error"}
                item["failures"] += int(failed)
                item["subtask_failures"] += int(failed and bool(row["parent_id"]))
                # Only the controlling runtime/agent result can confirm a
                # Goal outcome. The evaluator's provisional 'achieved' and
                # successful Workflow/approval are insufficient evidence.
                if row["component"] in {"runtime", "agent"}:
                    if row["status"].startswith("goal_"):
                        item["goal_status"] = row["status"].removeprefix("goal_")
            elif row["phase"] == "goal":
                item["goal_status"] = row["status"].removeprefix("goal_")
            elif row["phase"] == "usage":
                item["usage_calls"] += 1
                item["tokens"] += row["prompt_tokens"] + row["completion_tokens"]
                item["cost_known"] &= row["cost_usd"] is not None
                item["estimated_usd"] += row["cost_usd"] or 0
        for item in grouped.values():
            spans = item.pop("spans")
            starts = {key for key, span in spans.items() if span["start"]}
            roots = [span for span in spans.values() if
                     (span["start"] or span["end"])["parent_id"] == ""]
            if roots:
                latest = max(roots, key=lambda span: (span["start"] or span["end"])["seq"])
                item["root_status"] = latest["end"]["status"] if latest["end"] else None
            item["open_spans"] = sum(span["start"] is not None and span["end"] is None
                                     for span in spans.values())
            item["partial"] = (any(span["start"] is None for span in spans.values()) or
                               bool(item.pop("parents") - starts) or not roots)
            item["execution_status"] = ("unfinished" if item["open_spans"] else
                "partial" if item["partial"] else "finished")
            # Keep the controlling result separate from background activity.
            # A completed child must never upgrade an awaiting Goal to success.
            item["status"] = ("goal_" + item["goal_status"] if item["goal_status"] else
                              item["root_status"] or "unfinished")
            if item["partial"]:
                item["status"] = "partial"
            elif item["open_spans"] and item["goal_status"] is None:
                item["status"] = "unfinished"
            if not item.pop("cost_known") or not item["usage_calls"]:
                item["estimated_usd"] = None
        return sorted(grouped.values(), key=lambda row: row["timestamp"], reverse=True)[:max(1, min(100, limit))]


def traced(component, operation):
    """Join an active trace; inactive embedders/tests keep existing behavior."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            active = current_trace()
            if active is None:
                return fn(*args, **kwargs)
            reference = kwargs.get("run_id", "") or getattr(args[0], "run_id", "")
            if component == "tool" and len(args) > 1:
                reference = args[1]
            if component == "workflow" and operation == "run" and len(args) > 1:
                reference = args[1]
            with active.recorder.span(component, operation, reference=reference) as outcome:
                result = fn(*args, **kwargs)
                outcome["status"] = getattr(result, "termination_reason", "") or getattr(result, "status", "") or getattr(result, "action", "") or (
                    result.get("agent_termination_reason") or result.get("status", "completed")
                    if isinstance(result, dict) else "completed")
                goal = getattr(args[0], "goal_controller", None)
                if component == "agent" and operation == "run" and goal and goal.trace_id == active.trace_id:
                    active.recorder.goal_state(goal.last_status)
                return result
        return wrapped
    return decorate


def traced_async(component, operation):
    def decorate(fn):
        @wraps(fn)
        async def wrapped(*args, **kwargs):
            active = current_trace()
            if active is None:
                return await fn(*args, **kwargs)
            with active.recorder.span(component, operation,
                    reference=getattr(args[0], "run_id", "")) as outcome:
                result = await fn(*args, **kwargs)
                if isinstance(result, dict):
                    outcome["status"] = result.get("status", "completed")
                else:
                    outcome["status"] = getattr(result, "termination_reason", "") or getattr(result, "status", "completed")
                return result
        return wrapped
    return decorate


def runtime_traced(operation):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            recorder = getattr(self, "trace_recorder", None)
            if recorder is None:
                return fn(self, *args, **kwargs)
            goal = getattr(self, "goal_controller", None)
            active = current_trace()
            trace_id = (active.trace_id if active is not None else
                getattr(goal, "trace_id", "") if goal and goal.condition is not None else "")
            with recorder.span("runtime", operation, trace_id=trace_id) as outcome:
                result = fn(self, *args, **kwargs)
                if isinstance(result, dict):
                    outcome["status"] = result.get("agent_termination_reason") or result.get("status", "completed")
                else:
                    outcome["status"] = getattr(result, "status", "completed")
                if goal and goal.trace_id == trace_id:
                    recorder.goal_state(goal.last_status)
                return result
        return wrapped
    return decorate
