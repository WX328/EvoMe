from app.context.schema import sanitize_applicable_context, split_context
from app.reflection.service import ReflectionService


class ReflectionLLM:
    def json_chat(self, messages, temperature=0.1):
        return {
            "reflection_type": "success",
            "observation": "test",
            "insight": "test insight",
            "applicable_context": {
                "task_type": "architecture_design",
                "complexity": "medium",
                "scope_bounded": True,
                "available_skills": ["recall_relevant_memory"],
                "model": "deepseek-flash",
            },
            "strategy_signal": "test signal",
            "skill_signal": "",
            "expected_effect": "improve clarity",
            "confidence": 0.9,
        }


def test_split_context_separates_applicability_from_execution():
    context = {
        "task_type": "architecture_design",
        "complexity": "medium",
        "interaction_mode": "collaborative",
        "project_stage": "design",
        "scope_bounded": True,
        "available_skills": ["recall_relevant_memory"],
        "available_tools": [],
        "model": "deepseek-flash",
        "memory_quality": 0.95,
        "relevant_memory_count": 3,
        "complexity_indicators": {
            "cross_module": True,
            "multi_step": True,
            "tradeoff": True,
            "external_tool": False,
        },
    }

    applicability, execution = split_context(context)

    assert applicability["task_type"] == "architecture_design"
    assert applicability["cross_module"] is True
    assert applicability["scope_bounded"] is True
    assert "available_skills" not in applicability
    assert execution["available_skills"] == ["recall_relevant_memory"]
    assert execution["model"] == "deepseek-flash"


def test_reflection_filters_execution_fields_from_applicable_context():
    service = ReflectionService(ReflectionLLM())
    result = service.reflect(
        task_type="architecture_design",
        result_summary="test",
        evaluation={"score": 1.0},
        strategy_name="default",
        skills_used=["recall_relevant_memory"],
        context_data={
            "task_type": "architecture_design",
            "complexity": "medium",
            "scope_bounded": True,
            "available_skills": ["recall_relevant_memory"],
            "model": "deepseek-flash",
        },
    )

    assert result["applicable_context"] == {
        "task_type": "architecture_design",
        "complexity": "medium",
        "scope_bounded": True,
    }


def test_sanitize_applicable_context_drops_execution_fields():
    result = sanitize_applicable_context(
        {
            "task_type": "debugging",
            "available_tools": ["terminal"],
            "memory_quality": 0.9,
        }
    )
    assert result == {"task_type": "debugging"}
