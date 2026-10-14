"""S13 persisted roster and explicit task bindings (never auto-spawn on restart)."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from app.agent.team_bus import MessageBus, _agent

_TASK = re.compile(r"task_[0-9a-f]{8}\Z")
_TEAM = re.compile(r"team_[0-9a-f]{8}\Z")
_MAX_SIZE = 16 * 1024


class TeamStateStore:
    def __init__(self, *, workspace_root: str | Path, directory: str = ".team") -> None:
        self.bus = MessageBus(workspace_root=workspace_root, directory=directory)
        self.root = self.bus.root
        self.path = self.root / "roster.json"
        self._lock = RLock()

    @staticmethod
    def validate(record: Any) -> dict[str, Any]:
        if not isinstance(record, dict) or set(record) != {
            "team_id", "phase", "members", "bindings", "plans", "version",
        }:
            raise ValueError("invalid S13 roster schema")
        if not isinstance(record["team_id"], str) or not _TEAM.fullmatch(record["team_id"]):
            raise ValueError("invalid S13 team id")
        if record["phase"] not in {"active", "stopped"}:
            raise ValueError("invalid team phase")
        members = record["members"]
        if (not isinstance(members, list) or not 1 <= len(members) <= 3
                or any(not isinstance(m, dict) or set(m) != {"name", "role"}
                       for m in members)):
            raise ValueError("invalid team member roster")
        names = []
        for member in members:
            name = _agent(member["name"])
            role = member["role"]
            if (name in {"lead", "agent"} or not isinstance(role, str)
                    or not 1 <= len(role) <= 100):
                raise ValueError("invalid teammate name/role")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("duplicate teammate")
        bindings = record["bindings"]
        if not isinstance(bindings, dict) or len(bindings) > 90:
            raise ValueError("invalid team task bindings")
        for task_id, val in bindings.items():
            if (not isinstance(task_id, str) or not _TASK.fullmatch(task_id)
                    or not isinstance(val, dict) or set(val) != {"member", "worktree"}):
                raise ValueError("invalid task binding")
            if val["member"] not in set(names) | {"any"}:
                raise ValueError("task bound to unknown teammate")
            if (not isinstance(val["worktree"], str)
                    or len(val["worktree"]) > 72
                    or (val["worktree"] and
                        not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", val["worktree"]))):
                raise ValueError("invalid worktree binding")
        plans = record["plans"]
        if not isinstance(plans, dict) or set(plans) != set(names):
            raise ValueError("invalid teammate plan states")
        for state in plans.values():
            if not isinstance(state, dict) or set(state) != {"status", "content"}:
                raise ValueError("invalid plan state")
            if state["status"] not in {"none", "pending", "approved", "rejected"}:
                raise ValueError("invalid plan approval state")
            if not isinstance(state["content"], str) or len(state["content"]) > 1000:
                raise ValueError("invalid plan content")
        if type(record["version"]) is not int or record["version"] < 1:
            raise ValueError("invalid roster version")
        return record

    def load(self) -> dict[str, Any] | None:
        with self._lock:
            self.bus._guard()
            if self.path.is_symlink():
                raise PermissionError("symlinked team roster forbidden")
            if not self.path.exists():
                return None
            if self.path.stat().st_size > _MAX_SIZE:
                raise ValueError("oversized team roster")
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise ValueError("corrupt S13 roster JSON") from exc
            return self.validate(raw)

    def save(self, row: dict[str, Any]) -> None:
        self.validate(row)
        with self._lock:
            self.bus._guard()
            self.root.mkdir(parents=True, exist_ok=True)
            self.bus._guard()
            if self.path.is_symlink():
                raise PermissionError("symlinked team roster forbidden")
            body = json.dumps(row, ensure_ascii=False, indent=2) + "\n"
            if len(body.encode("utf-8")) > _MAX_SIZE:
                raise ValueError("roster exceeds maximum size")
            tmp: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", newline="\n",
                    prefix=".s13-", suffix=".tmp", dir=self.root, delete=False,
                ) as stream:
                    tmp = Path(stream.name)
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.bus._guard()
                if self.path.is_symlink():
                    raise PermissionError("roster target became symlinked")
                os.replace(tmp, self.path)
            finally:
                if tmp is not None and tmp.exists():
                    tmp.unlink()
