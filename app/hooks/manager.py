"""S04 synchronous lifecycle hooks; before hooks veto, after hooks observe."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping

EVENTS = frozenset({"before_agent", "after_agent", "before_tool", "after_tool"})


@dataclass(frozen=True)
class HookContext:
    event: str
    tool_name: str = ""
    risk: str = ""
    status: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    input_length: int = 0
    step_count: int = 0


@dataclass(frozen=True)
class HookDecision:
    allow: bool = True
    reason: str = ""


@dataclass(frozen=True)
class HookReport:
    allowed: bool = True
    reason: str = ""


@dataclass(frozen=True)
class HookRecord:
    timestamp: str
    event: str
    hook_name: str
    outcome: str
    tool_name: str
    risk: str
    status: str
    reason: str


class HookManager:
    def __init__(self, *, history_limit: int = 100) -> None:
        self._handlers: dict[str, list[tuple[int, int, str, Callable]]] = {
            event: [] for event in EVENTS
        }
        self._registration_order = 0
        self._history: deque[HookRecord] = deque(maxlen=max(1, int(history_limit)))
        self._lock = Lock()

    def register(
        self,
        event: str,
        handler: Callable[[HookContext], HookDecision | None],
        *,
        name: str | None = None,
        priority: int = 0,
    ) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown hook event: {event}")
        if not callable(handler):
            raise TypeError("hook handler must be callable")
        with self._lock:
            self._registration_order += 1
            index = self._registration_order
            self._handlers[event].append((int(priority), index, name or handler.__name__, handler))
            self._handlers[event].sort(key=lambda item: (-item[0], item[1]))

    def _record(self, context: HookContext, name: str, outcome: str, reason: str = "") -> None:
        # Never persist arguments, commands, prompts, responses or file contents.
        record = HookRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            event=context.event,
            hook_name=name,
            outcome=outcome,
            tool_name=context.tool_name,
            risk=context.risk,
            status=context.status,
            reason=reason[:240],
        )
        with self._lock:
            self._history.append(record)

    def dispatch(self, context: HookContext) -> HookReport:
        if context.event not in EVENTS:
            raise ValueError(f"unknown hook event: {context.event}")
        before = context.event.startswith("before_")
        with self._lock:
            handlers = tuple(self._handlers[context.event])
        for _, _, name, handler in handlers:
            try:
                decision = handler(context)
                if decision is not None and not isinstance(decision, HookDecision):
                    raise TypeError("hook must return HookDecision or None")
                if decision is not None and not decision.allow and before:
                    reason = decision.reason or f"{name} denied action"
                    self._record(context, name, "blocked", reason)
                    return HookReport(False, reason)
                self._record(context, name, "passed")
            except Exception as exc:
                reason = f"{name}: {type(exc).__name__}: {exc}"
                self._record(context, name, "error", reason)
                if before:
                    return HookReport(False, reason)
        self._record(context, "lifecycle", "allowed" if before else "observed")
        return HookReport()

    def recent(self, limit: int = 20) -> list[HookRecord]:
        with self._lock:
            return list(self._history)[-max(1, int(limit)):]
