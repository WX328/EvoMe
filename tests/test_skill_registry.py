from app.skills.base import SkillDefinition, SkillImplementation
from app.skills.registry import SkillRegistry


def test_skill_is_abstraction_and_impl_is_replaceable():
    registry = SkillRegistry()

    registry.register_skill(
        SkillDefinition(
            name="retrieve_memory",
            description="检索相关记忆",
        )
    )

    registry.register_implementation(
        SkillImplementation(
            name="keyword_v1",
            skill_name="retrieve_memory",
            kind="python_function",
            version=1,
            handler=lambda query: "keyword",
        )
    )

    registry.register_implementation(
        SkillImplementation(
            name="vector_v2",
            skill_name="retrieve_memory",
            kind="qdrant",
            version=2,
            handler=lambda query: "vector",
        )
    )

    selected = registry.resolve("retrieve_memory")

    assert selected.name == "vector_v2"
    assert selected.execute(query="test") == "vector"
