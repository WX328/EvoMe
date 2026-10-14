"""S12 Cron Scheduler: durable cursor + session-only delays, S11 read-only dispatch.

Operational guarantees:
- In this *single process*, serialized ticks and atomic advance-before-submit.
- An overdue trigger on process start is skipped, not replayed.
- Triggering creates a NEW S10 task and requests S11's restricted worker.
- No implicit retries or tool escalation. Not a distributed exactly-once queue.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Event, RLock, Thread
from typing import Any, Callable
from uuid import uuid4

from app.agent.background_tasks import BackgroundTaskService
from app.agent.cron_store import ScheduleStore, _utc, _stamp


class CronScheduler:
    def __init__(
        self,
        *,
        store: ScheduleStore,
        worker: BackgroundTaskService,
        poll_seconds: float = 1.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.worker = worker
        self.poll_seconds = max(0.2, min(60.0, float(poll_seconds)))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()
        self._session: dict[str, dict[str, Any]] = {}
        self._watch: dict[str, tuple[str, str]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=100)
        self._stop = Event()
        self._thread: Thread | None = None
        self._started = False

    def _now(self) -> datetime:
        return _utc(self.clock())

    def start(self) -> int:
        """Perform a skip-missed recovery *before* the first polling thread."""
        with self._lock:
            if self._started:
                return 0
            skipped = self.store.skip_missed_on_start(self._now())
            self._stop.clear()
            self._started = True
            self._thread = Thread(
                target=self._run,
                name="evome-s12-cron",
                daemon=True,
            )
            self._thread.start()
            return skipped

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception as exc:
                # Keep the scheduler alive; expose bounded diagnostics via
                # operator /cron-events. Do not silently turn errors into jobs.
                with self._lock:
                    self._events.append({
                        "type": "scheduler_error",
                        "at": _stamp(self._now()),
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    })

    def _ensure_active(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("S12 scheduler is shut down")

    def create_at(self, *, at: str, title: str) -> dict[str, Any]:
        self._ensure_active()
        self.worker._task_prompt({"title": title, "description": ""})
        return self.store.create(
            title=title, kind="at", expression=at, now=self._now(),
        )

    def create_cron(
        self, *,
        expression: str, utc_offset: str, title: str, max_runs: int = 3,
    ) -> dict[str, Any]:
        self._ensure_active()
        self.worker._task_prompt({"title": title, "description": ""})
        return self.store.create(
            title=title, kind="cron", expression=expression,
            utc_offset=utc_offset, max_runs=max_runs, now=self._now(),
        )

    def create_session(self, *, delay_seconds: float, title: str) -> dict[str, Any]:
        self._ensure_active()
        self.worker._task_prompt({"title": title, "description": ""})
        if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, (int, float)):
            raise ValueError("session delay must be numeric")
        if not 5 <= delay_seconds <= 86400:
            raise ValueError("session delay must be 5-86400 seconds")
        with self._lock:
            if len(self._session) >= 20:
                raise ValueError("S12 session trigger limit reached")
            id_ = f"sess_{uuid4().hex[:8]}"
            target = self._now() + timedelta(seconds=delay_seconds)
            row = {
                "id": id_, "kind": "session", "title": title.strip(),
                "next_run_at": _stamp(target), "status": "active",
                "persisted": False,
            }
            self._session[id_] = row
            return dict(row)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            durable = [
                {**row, "persisted": True} for row in self.store.list()
            ]
            return durable + [dict(row) for row in self._session.values()]

    def get(self, schedule_id: str) -> dict[str, Any]:
        with self._lock:
            if schedule_id.startswith("sess_"):
                if schedule_id not in self._session:
                    raise KeyError("unknown session trigger")
                return dict(self._session[schedule_id])
            return {**self.store.get(schedule_id), "persisted": True}

    def pause(self, schedule_id: str) -> dict[str, Any]:
        self._ensure_active()
        return self.store.pause(schedule_id)

    def resume(self, schedule_id: str) -> dict[str, Any]:
        self._ensure_active()
        return self.store.resume(schedule_id, now=self._now())

    def cancel(self, schedule_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_active()
            if schedule_id.startswith("sess_"):
                row = self._session.pop(schedule_id, None)
                if row is None:
                    raise KeyError("unknown session trigger")
                return {**row, "status": "canceled"}
            return self.store.cancel(schedule_id)

    def _deliver(self, *, schedule_id: str, title: str, at: datetime) -> dict[str, Any]:
        task_id = ""
        try:
            task = self.worker.store.create_task(title)
            task_id = task["id"]
            job = self.worker.submit(task_id)
            event = {
                "type": "dispatched", "schedule_id": schedule_id,
                "task_id": task_id, "job_id": job["job_id"], "at": _stamp(at),
            }
            self._watch[task_id] = (schedule_id, job["job_id"])
        except Exception as exc:
            event = {
                "type": "dispatch_failed", "schedule_id": schedule_id,
                "task_id": task_id, "at": _stamp(at),
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        if schedule_id.startswith("cron_"):
            try:
                self.store.record_result(
                    schedule_id, task_id=task_id,
                    error=event.get("error", ""),
                )
            except Exception as exc:
                event["record_error"] = f"{type(exc).__name__}: {exc}"[:300]
        self._events.append(event)
        return event

    def tick(self) -> list[dict[str, Any]]:
        """Synchronous, idempotent poll; safe to invoke from an operator CLI."""
        with self._lock:
            self._ensure_active()
            now = self._now()
            fired: list[dict[str, Any]] = []
            for record in self.store.due(now)[:5]:
                # The disk cursor moves BEFORE create/submit; a crash in that
                # interval may lose one occurrence but cannot replay it.
                self.store.consume(record.id, now=now)
                fired.append(
                    self._deliver(schedule_id=record.id, title=record.title, at=now)
                )
            for schedule_id, item in list(self._session.items()):
                if len(fired) >= 10:
                    break
                if item["status"] == "active" and datetime.fromisoformat(
                    item["next_run_at"]
                ) <= now:
                    self._session.pop(schedule_id)
                    fired.append(
                        self._deliver(schedule_id=schedule_id, title=item["title"], at=now)
                    )
            # Completion notifications do not affect durable S10 state;
            # they are one-process event hints. Full state remains queryable.
            for task_id, (schedule_id, job_id) in list(self._watch.items()):
                state = self.worker.get_status(task_id)
                if state["task_status"] in {"completed", "failed"}:
                    self._events.append({
                        "type": "task_finished", "schedule_id": schedule_id,
                        "task_id": task_id, "at": _stamp(now),
                        "task_status": state["task_status"],
                    })
                    self._watch.pop(task_id, None)
            return fired

    def events(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-max(1, min(100, int(limit))):]

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5 if wait else 0.1)
        with self._lock:
            self._session.clear()
            self._started = False
