"""S16 durable workflow run snapshots, append-only agent journal and run locks.

Single workspace, multi-process-safe per-run O_EXCL lock. Crashed runs are not
automatically replayed. The operator may explicitly clear an orphan lock
only after verifying its previous process has exited.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

RUN_RE = re.compile(r"wf_[0-9a-f]{12}\Z")
MAX_DOC_BYTES = 48_000
MAX_JOURNAL_BYTES = 480_000
MAX_RUNS = 50
MAX_EVENTS = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(value: Any, limit: int = MAX_DOC_BYTES) -> bytes:
    result = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(result) > limit:
        raise ValueError("S16 JSON data exceeds configured size limit")
    return result


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_RE.fullmatch(run_id):
        raise ValueError("S16 run_id must match wf_<12 lowercase hex chars>")
    return run_id


class WorkflowStore:
    def __init__(self, workspace_root: str | Path, directory: str = ".workflows"):
        self.workspace = Path(workspace_root).resolve()
        rel = Path(directory)
        if (not directory or rel == Path(".") or rel.is_absolute()
                or ".." in rel.parts or "\x00" in directory):
            raise ValueError("S16 directory must be workspace-relative")
        self.root = self.workspace / rel
        self._lock = RLock()
        self._guard()

    def _guard(self) -> None:
        if not self.root.resolve().is_relative_to(self.workspace):
            raise PermissionError("S16 state directory escaped workspace")
        current = self.root
        while current != self.workspace:
            if current.is_symlink() or (
                hasattr(current, "is_junction") and current.is_junction()
            ):
                raise PermissionError("S16 rejects symlinks/junctions in its state path")
            current = current.parent
            if not current.is_relative_to(self.workspace):
                raise PermissionError("S16 state path escaped workspace")

    def _path(self, run_id: str, suffix: str) -> Path:
        validate_run_id(run_id)
        self._guard()
        result = self.root / f"{run_id}.{suffix}"
        if result.is_symlink():
            raise PermissionError("S16 run file may not be a symlink")
        return result

    def _write_atomic(self, path: Path, value: Any) -> None:
        self._guard()
        raw = json_bytes(value) + b"\n"
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard()
        if path.is_symlink():
            raise PermissionError("S16 cannot overwrite symlinked files")
        temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=self.root, prefix=".s16-", suffix=".tmp",
                delete=False,
            ) as out:
                temp = Path(out.name)
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            self._guard()
            if path.is_symlink():
                raise PermissionError("S16 output target changed to symlink")
            os.replace(temp, path)
        finally:
            if temp is not None and temp.exists():
                temp.unlink()

    def _read_json(self, path: Path) -> dict[str, Any]:
        self._guard()
        if path.is_symlink() or path.stat().st_size > MAX_DOC_BYTES:
            raise ValueError("S16 unsafe/oversized run document")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError("S16 corrupt run JSON") from exc
        if not isinstance(doc, dict):
            raise ValueError("S16 run JSON must be an object")
        return doc

    def _validate_snapshot(self, row: dict[str, Any], run_id: str) -> dict[str, Any]:
        required = {
            "id", "workflow", "args", "status", "phase", "created_at",
            "updated_at", "progress", "agent_calls", "version", "error",
        }
        if set(row) != required or row["id"] != run_id:
            raise ValueError("S16 invalid snapshot schema/id")
        if row["status"] not in {"queued", "running", "completed", "failed"}:
            raise ValueError("S16 invalid run status")
        if not isinstance(row["workflow"], str) or len(row["workflow"]) > 64:
            raise ValueError("S16 invalid workflow name")
        if not isinstance(row["args"], dict) or len(json_bytes(row["args"], 8000)) > 8000:
            raise ValueError("S16 invalid run args")
        if not isinstance(row["phase"], str) or len(row["phase"]) > 100:
            raise ValueError("S16 invalid phase")
        if not isinstance(row["progress"], list) or len(row["progress"]) > MAX_EVENTS:
            raise ValueError("S16 invalid progress events")
        for event in row["progress"]:
            if not isinstance(event, dict) or any(not isinstance(k, str) for k in event):
                raise ValueError("S16 invalid event")
        if type(row["agent_calls"]) is not int or not 0 <= row["agent_calls"] <= 30:
            raise ValueError("S16 invalid agent calls")
        if type(row["version"]) is not int or row["version"] < 1:
            raise ValueError("S16 invalid version")
        if not isinstance(row["error"], str) or len(row["error"]) > 200:
            raise ValueError("S16 invalid error field")
        for field in ("created_at", "updated_at"):
            if not isinstance(row[field], str) or len(row[field]) > 45:
                raise ValueError("S16 invalid timestamp")
        return row

    def load(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id, "json")
        with self._lock:
            return self._validate_snapshot(self._read_json(path), run_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            self._guard()
            if not self.root.is_dir():
                return []
            rows = []
            for path in self.root.glob("wf_*.json"):
                # WorkflowStore writes two distinct JSON files for a run:
                # wf_<id>.json (snapshot) and wf_<id>.output.json (result).
                # The glob matches both. Only snapshots count as runs.
                # Reject symlinks even for output files; do not silently
                # ignore unexpected or malformed paths.
                if path.is_symlink() or not path.is_file():
                    raise ValueError("S16 unsafe run snapshot/output path")
                if path.name.endswith(".output.json"):
                    output_id = path.name[:-len(".output.json")]
                    if not RUN_RE.fullmatch(output_id):
                        raise ValueError("S16 unexpected workflow output path")
                    continue
                if not RUN_RE.fullmatch(path.stem):
                    raise ValueError("S16 unexpected run snapshot path")
                rows.append(self.load(path.stem))
                if len(rows) > MAX_RUNS:
                    raise ValueError("S16 run count capacity exceeded")
            return sorted(rows, key=lambda item: (item["created_at"], item["id"]), reverse=True)

    def reserve(self, workflow: str, args: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if len(self.list()) >= MAX_RUNS:
                raise ValueError("S16 run capacity reached (archive older runs manually)")
            self.root.mkdir(parents=True, exist_ok=True)
            self._guard()
            for _ in range(5):
                run_id = f"wf_{uuid4().hex[:12]}"
                try:
                    self.lock(run_id)
                except FileExistsError:
                    continue
                snapshot = {
                    "id": run_id, "workflow": workflow, "args": args,
                    "status": "queued", "phase": "", "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "progress": [
                        {"kind": "async_launched", "message": workflow, "at": utc_now()},
                    ],
                    "agent_calls": 0, "version": 1, "error": "",
                }
                try:
                    self.save(snapshot)
                except Exception:
                    self.unlock(run_id)
                    raise
                return snapshot
            raise RuntimeError("S16 could not reserve unique run id")

    def save(self, snapshot: dict[str, Any]) -> None:
        run_id = validate_run_id(snapshot["id"])
        with self._lock:
            self._validate_snapshot(snapshot, run_id)
            self._write_atomic(self._path(run_id, "json"), snapshot)

    def lock(self, run_id: str) -> None:
        with self._lock:
            path = self._path(run_id, "lock")
            self.root.mkdir(parents=True, exist_ok=True)
            self._guard()
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}))
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                path.unlink(missing_ok=True)
                raise

    def unlock(self, run_id: str) -> None:
        with self._lock:
            path = self._path(run_id, "lock")
            path.unlink(missing_ok=True)

    def locked(self, run_id: str) -> bool:
        return self._path(run_id, "lock").exists()

    def clear_orphan_lock(self, run_id: str, *, confirmed: bool, active: bool = False) -> None:
        if not confirmed or active:
            raise PermissionError("operator confirmation and no active worker required")
        with self._lock:
            self.load(run_id)
            path = self._path(run_id, "lock")
            if not path.exists():
                raise ValueError("S16 run has no lock to clear")
            # A live process may legitimately own this file. Do not trust
            # elapsed time alone. Avoid kill(0) semantics on Windows.
            row = self._read_json(path)
            owner = row.get("pid")
            if not isinstance(owner, int) or owner < 1:
                raise ValueError("invalid S16 lock owner")
            if owner == os.getpid():
                raise PermissionError("S16 lock is owned by the current process")
            try:
                import psutil  # optional, installed only if available
            except ImportError:
                raise RuntimeError("cannot safely verify stale PID; install psutil to unlock")
            if psutil.pid_exists(owner):
                raise PermissionError("S16 lock holder PID still exists; do not force unlock")
            path.unlink()

    def read_output(self, run_id: str) -> dict[str, Any] | None:
        path = self._path(run_id, "output.json")
        if not path.exists():
            return None
        return self._read_json(path)

    def write_output(self, run_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._write_atomic(self._path(run_id, "output.json"), value)

    def journal(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(run_id, "journal.jsonl")
            if not path.exists():
                return {}
            if path.stat().st_size > MAX_JOURNAL_BYTES:
                raise ValueError("S16 journal exceeds size limit")
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > 40:
                    raise ValueError("S16 journal has excessive entries")
                cache = {}
                for line in lines:
                    row = json.loads(line)
                    if (not isinstance(row, dict) or set(row) != {"key", "value"}
                            or not isinstance(row["key"], str)
                            or not re.fullmatch(r"agent-[0-9a-f]{32}", row["key"])):
                        raise ValueError("S16 invalid journal entry")
                    if row["key"] in cache:
                        raise ValueError("S16 duplicate journal key")
                    json_bytes(row["value"], 12000)
                    cache[row["key"]] = row["value"]
                return cache
            except (UnicodeError, ValueError) as exc:
                raise ValueError("S16 corrupt journal; do not resume until repaired") from exc

    def append(self, run_id: str, key: str, value: Any) -> None:
        with self._lock:
            if not self.locked(run_id):
                raise PermissionError("S16 append requires exclusive run lock")
            existing = self.journal(run_id)
            if key in existing:
                raise ValueError("S16 cannot overwrite an existing journal call")
            if not re.fullmatch(r"agent-[0-9a-f]{32}", key):
                raise ValueError("S16 invalid semantic key")
            raw = json_bytes({"key": key, "value": value}, 12000) + b"\n"
            path = self._path(run_id, "journal.jsonl")
            if path.exists() and path.stat().st_size + len(raw) > MAX_JOURNAL_BYTES:
                raise ValueError("S16 journal capacity exceeded")
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
