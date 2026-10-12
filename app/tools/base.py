from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolImplementation:
    name: str
    tool_name: str
    handler: Callable[..., Any]
    version: int = 1
    enabled: bool = True

    def execute(self, **kwargs) -> Any:
        return self.handler(**kwargs)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    status: str = "executed"
    risk: str = "low"
    approval_id: str = ""
