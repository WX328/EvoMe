"""S13 team inboxes: bounded append-only JSONL, drain-on-read, validated peers.

One active EvoMe process owns this bus; file operations are serialized by its
Condition. Mailboxes persist across restarts but are not a transactional broker.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, RLock
from typing import Any
from uuid import uuid4


_AGENT = re.compile(r"[a-z][a-z0-9_-]{0,23}\Z")
_TYPES = frozenset({
    "message", "task_assignment", "result", "idle_notification",
    "shutdown_request", "shutdown_response",
    "plan_approval_request", "plan_approval_response",
})
_MAX_FILE_BYTES = 96 * 1024
_MAX_MESSAGES = 80


def _agent(name: str) -> str:
    if not isinstance(name, str) or not _AGENT.fullmatch(name):
        raise ValueError("S13 agent name must be lowercase ASCII slug (<=24 chars)")
    return name


class MessageBus:
    def __init__(self, *, workspace_root: str | Path, directory: str = ".team") -> None:
        self.workspace = Path(workspace_root).resolve()
        relative = Path(directory)
        if (not directory or relative == Path(".") or relative.is_absolute()
                or ".." in relative.parts):
            raise ValueError("S13 team directory must be workspace relative")
        self.root = self.workspace / relative
        self.mailboxes = self.root / "mailboxes"
        self._condition = Condition(RLock())
        self._members: frozenset[str] = frozenset({"lead"})
        self._guard()

    def register(self, members: list[str]) -> None:
        ids = [_agent(name) for name in members]
        if "lead" in ids or len(ids) != len(set(ids)) or len(ids) > 3:
            raise ValueError("duplicate, reserved or excessive S13 teammate names")
        with self._condition:
            self._members = frozenset(["lead", *ids])

    def _guard(self) -> None:
        target = self.mailboxes
        while target != self.workspace:
            if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
                raise PermissionError("S13 mailbox path must not contain symlinks/junctions")
            target = target.parent
            if not target.is_relative_to(self.workspace):
                raise PermissionError("S13 mailbox escaped workspace")
        if not self.mailboxes.resolve().is_relative_to(self.workspace):
            raise PermissionError("S13 mailbox escaped workspace")

    def _path(self, name: str) -> Path:
        _agent(name)
        self._guard()
        if name not in self._members:
            raise PermissionError("unknown team member")
        path = self.mailboxes / f"{name}.jsonl"
        if path.is_symlink():
            raise PermissionError("symlink mailbox forbidden")
        return path

    @staticmethod
    def _validate(obj: Any) -> dict[str, Any]:
        if not isinstance(obj, dict) or set(obj) != {
            "id", "from", "to", "type", "content", "metadata", "created_at",
        }:
            raise ValueError("invalid team message schema")
        _agent(obj["from"])
        _agent(obj["to"])
        if obj["type"] not in _TYPES:
            raise ValueError("unknown team message type")
        if (not isinstance(obj["content"], str)
                or len(obj["content"]) > 2000):
            raise ValueError("team message exceeds 2000 chars")
        if (not isinstance(obj["metadata"], dict)
                or len(obj["metadata"]) > 8
                or any(not isinstance(k, str) or len(k) > 32
                       or not isinstance(v, (str, int, bool)) or len(str(v)) > 160
                       for k, v in obj["metadata"].items())):
            raise ValueError("invalid team metadata")
        if (not isinstance(obj["id"], str) or len(obj["id"]) != 16
                or not isinstance(obj["created_at"], str)
                or len(obj["created_at"]) > 40):
            raise ValueError("invalid team message id/time")
        return obj

    def send(
        self, from_agent: str, to_agent: str, content: str,
        *, msg_type: str = "message", metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = self._validate({
            "id": uuid4().hex[:16], "from": from_agent, "to": to_agent,
            "type": msg_type, "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        with self._condition:
            path = self._path(to_agent)
            if from_agent not in self._members:
                raise PermissionError("sender not in current team")
            self._guard()
            self.mailboxes.mkdir(parents=True, exist_ok=True)
            self._guard()
            existing = path.stat().st_size if path.exists() else 0
            if existing > _MAX_FILE_BYTES:
                raise ValueError("mailbox already oversized")
            encoded = (json.dumps(item, ensure_ascii=True) + "\n").encode("utf-8")
            if existing + len(encoded) > _MAX_FILE_BYTES:
                raise ValueError("mailbox size limit reached")
            if path.exists() and path.read_bytes().count(b"\n") >= _MAX_MESSAGES:
                raise ValueError("mailbox message capacity reached")
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._condition.notify_all()
            return item

    def _read(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        if not path.exists():
            return []
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError("mailbox exceeds size limit")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) > _MAX_MESSAGES:
                raise ValueError("mailbox message count exceeded")
            records = [self._validate(json.loads(line)) for line in lines]
            # Old team messages may still exist after explicit replacement of
            # the roster. Validate recipients here; TeamRuntime drops messages
            # whose team_id is not the currently approved team.
            if any(item["to"] != name for item in records):
                raise ValueError("mailbox recipient mismatch")
            return records
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"corrupt team mailbox for {name}") from exc

    def peek(self, name: str) -> list[dict[str, Any]]:
        with self._condition:
            return list(self._read(name))

    def read_inbox(self, name: str) -> list[dict[str, Any]]:
        """Drain only after validating all messages; no duplicate drain in-process."""
        with self._condition:
            records = self._read(name)
            path = self._path(name)
            if path.exists():
                path.unlink()
            return records

    def wait_for_messages(self, name: str, timeout: float = 0.5) -> list[dict[str, Any]]:
        if timeout < 0 or timeout > 60:
            raise ValueError("invalid inbox wait timeout")
        with self._condition:
            self._condition.wait_for(
                lambda: bool(self._read(name)),
                timeout=timeout,
            )
            return self.read_inbox(name)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()
