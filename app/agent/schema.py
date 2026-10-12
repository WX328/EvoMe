from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentStep:
    index: int
    action: str
    tool_call: ToolCall | None = None
    tool_output: Any = None
    error: str = ""


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    steps: tuple[AgentStep, ...]
    termination_reason: str

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(
            step.tool_call
            for step in self.steps
            if step.tool_call is not None
        )

    @property
    def tool_error_count(self) -> int:
        return sum(1 for step in self.steps if step.error)
