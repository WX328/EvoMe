"""S10 durable Task System: validated file-backed dependency DAG.

This is NOT the S05 in-process execution plan. Each task is a stable
task_XXXXXXXX.json record under a protected workspace-relative .tasks directory.
S10 does not launch work, threads, timers, or tool executions; S11 adds that.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

TASK_ID = re.compile(r"task_[0-9a-f]{8}\Z")
_STATUSES = frozenset({"pending", "in_progress", "completed", "failed"})
MAX_TASKS = 250
MAX_DEPENDENCIES = 32
MAX_DOCUMENT_BYTES = 12000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value: Any, *, field: str, limit: int, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    value = value.strip()
    if (not value and required) or len(value) > limit or "\x00" in value:
        raise ValueError(f"{field} must have {'1' if required else '0'}-{limit} characters")
    return value


@dataclass(frozen=True)
class TaskRecord:
    id: str
    title: str
    description: str
    status: str
    blocked_by: tuple[str, ...]
    owner: str
    created_at: str
    updated_at: str
    result: str = ""
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title,
            "description": self.description, "status": self.status,
            "blockedBy": list(self.blocked_by), "owner": self.owner,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "result": self.result, "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Any, *, expected_id: str | None = None) -> "TaskRecord":
        if not isinstance(data, dict):
            raise ValueError("task record must be a JSON object")
        required = {
            "id", "title", "description", "status", "blockedBy", "owner",
            "created_at", "updated_at", "result", "version",
        }
        if set(data) != required:
            raise ValueError("task record schema mismatch")
        task_id = data["id"]
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise ValueError("invalid task ID")
        if expected_id is not None and task_id != expected_id:
            raise ValueError("task id / filename mismatch")
        title = _text(data["title"], field="title", limit=200)
        description = _text(data["description"], field="description", limit=2000, required=False)
        owner = _text(data["owner"], field="owner", limit=80, required=False)
        result = _text(data["result"], field="result", limit=1200, required=False)
        status = data["status"]
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("invalid task status")
        deps = data["blockedBy"]
        if (
            not isinstance(deps, list) or len(deps) > MAX_DEPENDENCIES
            or any(not isinstance(dep, str) or not TASK_ID.fullmatch(dep) for dep in deps)
            or len(deps) != len(set(deps)) or task_id in deps
        ):
            raise ValueError("invalid blockedBy dependencies")
        if (status == "pending" and (owner or result)) or (
            status == "in_progress" and (not owner or result)
        ) or (status == "completed" and not owner) or (
            status == "failed" and (not owner or not result)
        ):
            raise ValueError("inconsistent task status/owner/result")
        if any(not isinstance(data[field], str) or not data[field] for field in ("created_at", "updated_at")):
            raise ValueError("invalid task timestamps")
        if type(data["version"]) is not int or data["version"] < 1:
            raise ValueError("invalid task version")
        return cls(
            id=task_id, title=title, description=description, status=status,
            blocked_by=tuple(deps), owner=owner, created_at=data["created_at"],
            updated_at=data["updated_at"], result=result, version=data["version"],
        )


class TaskStore:
    """Single-process, atomic-write task store with validated dependency graph."""

    def __init__(self, *, workspace_root: str | Path = ".", directory: str = ".tasks") -> None:
        self.workspace = Path(workspace_root).resolve()
        folder = Path(directory)
        if not directory or folder.is_absolute() or ".." in folder.parts or folder == Path("."):
            raise ValueError("task directory must be workspace-relative")
        self.root = self.workspace / folder
        self._lock = RLock()
        self._guard()

    def _guard(self) -> None:
        target = self.root
        while target != self.workspace:
            if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
                raise PermissionError("S10 task store path may not use symlinks/junctions")
            target = target.parent
            if not target.is_relative_to(self.workspace):
                raise PermissionError("S10 task store escaped workspace")
        if not self.root.resolve().is_relative_to(self.workspace):
            raise PermissionError("S10 task store escaped workspace")

    def _prepare(self) -> None:
        self._guard()
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard()

    @staticmethod
    def _validate_id(task_id: str) -> None:
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise ValueError("task ID must match task_XXXXXXXX")

    def _path(self, task_id: str) -> Path:
        self._validate_id(task_id)
        self._guard()
        path = self.root / f"{task_id}.json"
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise PermissionError("task JSON symlink/path escape denied")
        return path

    def _read(self, path: Path) -> TaskRecord:
        if path.is_symlink():
            raise PermissionError("symlinked tasks are not allowed")
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError(f"oversized task record: {path.name}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"corrupt task JSON: {path.name}") from exc
        return TaskRecord.from_dict(data, expected_id=path.stem)

    def _all(self) -> dict[str, TaskRecord]:
        self._guard()
        if not self.root.is_dir():
            return {}
        result: dict[str, TaskRecord] = {}
        for path in sorted(self.root.iterdir()):
            if not path.name.startswith("task_") or path.suffix != ".json":
                continue
            if not TASK_ID.fullmatch(path.stem):
                raise ValueError(f"invalid task record filename: {path.name}")
            if len(result) >= MAX_TASKS:
                raise ValueError("S10 task limit reached")
            result[path.stem] = self._read(path)
        for task in result.values():
            if any(dep not in result for dep in task.blocked_by):
                raise ValueError(f"task {task.id} has a dangling dependency")
            if any(
                dep == task.id for dep in task.blocked_by
            ):
                raise ValueError(f"self dependency in {task.id}")
        self._assert_acyclic(result)
        # Fail closed: no task can remain claimed if its prerequisites
        # have never been completed. Concurrent changes are not supported.
        for task in result.values():
            if task.status in {"in_progress", "completed"} and not all(
                result[dep].status == "completed" for dep in task.blocked_by
            ):
                raise ValueError(f"inconsistent completed dependency in {task.id}")
        return result

    @staticmethod
    def _assert_acyclic(tasks: dict[str, TaskRecord]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task dependency cycle detected")
            if task_id in visited:
                return
            visiting.add(task_id)
            for parent in tasks[task_id].blocked_by:
                visit(parent)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in tasks:
            visit(task_id)

    @staticmethod
    def _ready(task: TaskRecord, tasks: dict[str, TaskRecord]) -> bool:
        return task.status == "pending" and all(
            tasks[dep].status == "completed" for dep in task.blocked_by
        )

    @staticmethod
    def _view(task: TaskRecord, tasks: dict[str, TaskRecord]) -> dict[str, Any]:
        output = task.to_dict()
        output["ready"] = TaskStore._ready(task, tasks)
        output["blockedByOpen"] = [
            dep for dep in task.blocked_by if tasks[dep].status != "completed"
        ]
        return output

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = self._all()
            return [
                self._view(task, tasks) for task in
                sorted(tasks.values(), key=lambda row: (row.created_at, row.id))
            ]

    def ready_tasks(self) -> list[dict[str, Any]]:
        return [row for row in self.list_tasks() if row["ready"]]

    def get_task(self, task_id: str) -> dict[str, Any]:
        self._validate_id(task_id)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks:
                raise KeyError(f"unknown task: {task_id}")
            return self._view(tasks[task_id], tasks)

    def _save(self, record: TaskRecord) -> None:
        self._prepare()
        path = self._path(record.id)
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n"
        if len(payload.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("S10 task record exceeds file size limit")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", dir=self.root,
                prefix=".s10-", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._guard()
            if path.is_symlink():
                raise PermissionError("task target became a symlink")
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def create_task(
        self, title: str, description: str = "",
        blockedBy: list[str] | None = None,
    ) -> dict[str, Any]:
        title = _text(title, field="title", limit=200)
        description = _text(description, field="description", limit=2000, required=False)
        deps = list(blockedBy or [])
        if not isinstance(blockedBy, (list, type(None))):
            raise ValueError("blockedBy must be a list")
        if len(deps) > MAX_DEPENDENCIES or len(deps) != len(set(deps)):
            raise ValueError("too many or duplicate dependencies")
        for dep in deps:
            self._validate_id(dep)
        with self._lock:
            tasks = self._all()
            if len(tasks) >= MAX_TASKS:
                raise ValueError("S10 task limit reached")
            if any(dep not in tasks for dep in deps):
                raise ValueError("dependency must already exist")
            task_id = f"task_{uuid4().hex[:8]}"
            while task_id in tasks:
                task_id = f"task_{uuid4().hex[:8]}"
            now = _utcnow()
            record = TaskRecord(
                id=task_id, title=title, description=description,
                status="pending", blocked_by=tuple(deps), owner="",
                result="", created_at=now, updated_at=now,
            )
            self._save(record)
            tasks[task_id] = record
            return self._view(record, tasks)

    def add_dependency(self, task_id: str, dependency_id: str) -> dict[str, Any]:
        self._validate_id(task_id)
        self._validate_id(dependency_id)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks or dependency_id not in tasks:
                raise KeyError("task/dependency does not exist")
            record = tasks[task_id]
            if record.status != "pending":
                raise ValueError("cannot alter dependencies after task started")
            if task_id == dependency_id:
                raise ValueError("self-dependency is forbidden")
            if dependency_id in record.blocked_by:
                raise ValueError("dependency already exists")
            if len(record.blocked_by) >= MAX_DEPENDENCIES:
                raise ValueError("too many dependencies")
            changed = replace(
                record, blocked_by=(*record.blocked_by, dependency_id),
                updated_at=_utcnow(), version=record.version + 1,
            )
            draft = dict(tasks)
            draft[task_id] = changed
            self._assert_acyclic(draft)
            self._save(changed)
            return self._view(changed, draft)

    def claim_task(self, task_id: str, owner: str) -> dict[str, Any]:
        self._validate_id(task_id)
        owner = _text(owner, field="owner", limit=80)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks:
                raise KeyError(f"unknown task: {task_id}")
            record = tasks[task_id]
            if record.status != "pending":
                raise ValueError("task is already claimed or completed")
            if not self._ready(record, tasks):
                raise ValueError("task blocked by unfinished dependencies")
            changed = replace(
                record, status="in_progress", owner=owner,
                updated_at=_utcnow(), version=record.version + 1,
            )
            self._save(changed)
            draft = dict(tasks)
            draft[task_id] = changed
            return self._view(changed, draft)

    def complete_task(
        self, task_id: str, owner: str, result: str = "",
    ) -> dict[str, Any]:
        self._validate_id(task_id)
        owner = _text(owner, field="owner", limit=80)
        result = _text(result, field="result", limit=1200, required=False)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks:
                raise KeyError(f"unknown task: {task_id}")
            record = tasks[task_id]
            if record.status != "in_progress":
                raise ValueError("task must be in_progress before completion")
            if record.owner != owner:
                raise PermissionError("only the claiming owner can complete this task")
            if not all(tasks[dep].status == "completed" for dep in record.blocked_by):
                raise ValueError("task dependency no longer complete")
            changed = replace(
                record, status="completed", result=result,
                updated_at=_utcnow(), version=record.version + 1,
            )
            self._save(changed)
            draft = dict(tasks)
            draft[task_id] = changed
            return self._view(changed, draft)

    def release_task(self, task_id: str, owner: str) -> dict[str, Any]:
        """Manual recovery for abandoned claims; no automatic retry or execution."""
        self._validate_id(task_id)
        owner = _text(owner, field="owner", limit=80)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks:
                raise KeyError(f"unknown task: {task_id}")
            record = tasks[task_id]
            if record.status != "in_progress" or record.owner != owner:
                raise PermissionError("only the active owner can release task")
            changed = replace(
                record, status="pending", owner="", result="",
                updated_at=_utcnow(), version=record.version + 1,
            )
            self._save(changed)
            draft = dict(tasks)
            draft[task_id] = changed
            return self._view(changed, draft)

    def fail_task(
        self, task_id: str, owner: str, error: str,
    ) -> dict[str, Any]:
        """Persist a worker failure; S10 dependencies remain blocked."""
        self._validate_id(task_id)
        owner = _text(owner, field="owner", limit=80)
        error = _text(error, field="error", limit=1200)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks:
                raise KeyError(f"unknown task: {task_id}")
            record = tasks[task_id]
            if record.status != "in_progress" or record.owner != owner:
                raise PermissionError("only the active owner can fail this task")
            changed = replace(
                record, status="failed", result=error,
                updated_at=_utcnow(), version=record.version + 1,
            )
            self._save(changed)
            tasks[task_id] = changed
            return self._view(changed, tasks)

    def retry_failed_task(self, task_id: str) -> dict[str, Any]:
        """Explicit operator recovery. No LLM or worker retries automatically."""
        self._validate_id(task_id)
        with self._lock:
            tasks = self._all()
            if task_id not in tasks:
                raise KeyError(f"unknown task: {task_id}")
            record = tasks[task_id]
            if record.status != "failed":
                raise ValueError("only a failed task may be manually retried")
            changed = replace(
                record, status="pending", owner="", result="",
                updated_at=_utcnow(), version=record.version + 1,
            )
            self._save(changed)
            tasks[task_id] = changed
            return self._view(changed, tasks)
