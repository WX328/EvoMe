from app.skills.base import SkillDefinition, SkillImplementation
from app.skills.registry import SkillRegistry


def build_default_skill_registry(
    memory_service,
) -> SkillRegistry:
    registry = SkillRegistry()

    registry.register_skill(
        SkillDefinition(
            name="recall_relevant_memory",
            description="根据当前任务回忆相关长期经验",
        )
    )

    registry.register_implementation(
        SkillImplementation(
            name="sqlite_memory_recall_v1",
            skill_name="recall_relevant_memory",
            kind="python_function",
            version=1,
            handler=lambda query, top_k=5: memory_service.search(
                query=query,
                top_k=top_k,
            ),
        )
    )

    return registry
