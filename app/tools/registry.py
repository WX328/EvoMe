from __future__ import annotations

from app.tools.base import ToolDefinition, ToolImplementation


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._implementations: dict[str, list[ToolImplementation]] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        self._implementations.setdefault(tool.name, [])

    def register_implementation(self, implementation: ToolImplementation) -> None:
        if implementation.tool_name not in self._tools:
            raise KeyError(f"Tool 未注册: {implementation.tool_name}")
        self._implementations[implementation.tool_name].append(implementation)

    def unregister_tool(self, name: str) -> None:
        """Remove a dynamic plugin tool without touching other tool entries."""
        if name not in self._tools:
            raise KeyError(f"unknown registered tool: {name}")
        self._tools.pop(name)
        self._implementations.pop(name, None)

    def get_tool(self, name: str) -> ToolDefinition:
        return self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def resolve(self, tool_name: str) -> ToolImplementation:
        candidates = [
            item
            for item in self._implementations.get(tool_name, [])
            if item.enabled
        ]
        if not candidates:
            raise LookupError(f"Tool '{tool_name}' 没有可用实现")
        return max(candidates, key=lambda item: item.version)
