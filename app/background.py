from __future__ import annotations
from contextvars import copy_context

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable
from uuid import uuid4


@dataclass
class BackgroundJob:
    id: str
    kind: str
    status: str = "queued"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)


class BackgroundEvolutionQueue:
    """Single-worker in-process queue for the slow evolution loop.

    V0.2.1 deliberately keeps this lightweight. A single worker preserves
    interaction order and avoids concurrent SQLite/Qdrant mutation. Later this
    boundary can be replaced by Redis/RQ/Celery without changing the fast path.
    """

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="evome-slow-loop",
        )
        self._jobs: dict[str, BackgroundJob] = {}
        self._futures: dict[str, Future] = {}
        self._lock = Lock()

    def submit(
        self,
        *,
        kind: str,
        fn: Callable[[], dict[str, Any]],
    ) -> str:
        job_id = uuid4().hex[:10]
        job = BackgroundJob(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job

        def runner() -> dict[str, Any]:
            with self._lock:
                job.status = "running"
                job.started_at = datetime.now()
            try:
                result = fn() or {}
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.completed_at = datetime.now()
                raise
            else:
                with self._lock:
                    job.status = "completed"
                    job.result = dict(result)
                    job.completed_at = datetime.now()
                return result

        context = copy_context()
        future = self._executor.submit(context.run, runner)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def get(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 10) -> list[BackgroundJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            raise KeyError(f"未知 background job: {job_id}")
        return future.result(timeout=timeout)

    def wait_all(self, timeout: float | None = None) -> None:
        with self._lock:
            futures = list(self._futures.values())
        for future in futures:
            future.result(timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
