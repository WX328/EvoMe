from __future__ import annotations

from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.registry import ToolRegistry


def build_tool_registry_from_skills(skill_registry) -> ToolRegistry:
    """Expose existing EvoMe Skills through the S1 Tool boundary.

    Skill remains the capability abstraction. Tool is the executable surface
    consumed by the Agent Loop. The adapter keeps those concepts separate while
    allowing the current Skill Registry to participate immediately.
    """
    registry = ToolRegistry()
    for skill in skill_registry.list_skills():
        registry.register_tool(
            ToolDefinition(
                name=skill.name,
                description=skill.description,
                input_schema=dict(skill.input_schema or {}),
            )
        )
        for implementation in skill_registry.list_implementations(skill.name):
            registry.register_implementation(
                ToolImplementation(
                    name=implementation.name,
                    tool_name=skill.name,
                    handler=implementation.handler,
                    version=implementation.version,
                    enabled=implementation.enabled,
                )
            )
    return registry
