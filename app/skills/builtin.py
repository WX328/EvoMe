from app.skills.base import SkillDefinition, SkillImplementation
from app.skills.registry import SkillRegistry


def build_default_skill_registry(personal_memory_service) -> SkillRegistry:
    registry = SkillRegistry()

    registry.register_skill(
        SkillDefinition(
            name="recall_relevant_memory",
            description="根据当前任务回忆相关 Personal Memory",
            input_schema={
                "query": {"type": "string", "description": "当前问题或检索关键词"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            },
            output_schema={
                "type": "array",
                "description": "与当前任务相关的 Personal Memory 列表",
            },
        )
    )

    registry.register_implementation(
        SkillImplementation(
            name="sqlite_personal_memory_recall_v1",
            skill_name="recall_relevant_memory",
            kind="python_function",
            version=1,
            handler=lambda query, top_k=5: personal_memory_service.search(
                query=query, top_k=top_k
            ),
        )
    )
    return registry
