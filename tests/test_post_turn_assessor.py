from __future__ import annotations

from app.context.builder import ContextBuilder
from app.evaluation.service import EvaluationService
from app.learning.post_turn_assessor import PostTurnAssessor
from app.reflection.service import ReflectionService


class CountingLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.messages = None

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        self.messages = messages
        return self.payload


def test_post_turn_assessor_reads_turn_once():
    llm = CountingLLM({
        "evaluation": {"score": 0.95, "success": True, "reason": "direct"},
        "context": {"task_type": "architecture_design"},
        "reflection": {"reflection_type": "success", "insight": "reuse pattern"},
    })
    assessor = PostTurnAssessor(llm)
    result = assessor.assess(
        user_input="先讲数据流，再讲模块接口",
        assistant_response="收到。",
        relevant_memories=["用户偏好结构化架构讲解"],
        strategy_name="default_v2",
        skills_used=["recall_relevant_memory"],
    )
    assert llm.calls == 1
    assert result["evaluation"]["score"] == 0.95
    assert result["context"]["task_type"] == "architecture_design"
    assert result["reflection"]["insight"] == "reuse pattern"


def test_domain_normalizers_do_not_call_llm():
    llm = CountingLLM({})
    evaluation = EvaluationService(llm).normalize_result(
        {"score": 0.9, "success": True, "reason": "ok"}
    )
    context = ContextBuilder(llm, model_name="test").normalize_result(
        user_input="一起设计架构",
        result={
            "task_type": "architecture_design",
            "task_type_confidence": 0.9,
            "interaction_mode": "collaborative",
            "interaction_mode_confidence": 0.9,
            "complexity_indicators": {"multi_step": True},
            "risk_indicators": {},
        },
        relevant_memories=[],
        available_skills=["recall_relevant_memory"],
    )
    reflection = ReflectionService(llm).normalize_result(
        {
            "reflection_type": "success",
            "observation": "worked",
            "insight": "data flow first helps",
            "applicable_context": {
                "task_type": "architecture_design",
                "model": "must-be-removed",
            },
            "strategy_signal": "show data flow first",
            "expected_effect": "clarity",
            "confidence": 0.9,
        },
        context_data=context,
    )
    assert llm.calls == 0
    assert evaluation["score"] == 0.9
    assert context["task_type"] == "architecture_design"
    assert reflection["applicable_context"] == {"task_type": "architecture_design"}


def test_deterministic_experience_task_type_overrides_assessor_guess():
    llm = CountingLLM({})
    context = ContextBuilder(llm, model_name="test").normalize_result(
        user_input="检查架构边界",
        result={"task_type": "qa", "task_type_confidence": 0.9},
        relevant_memories=[],
        available_skills=[],
        task_type_hint="architecture_design",
        prefer_task_type_hint=True,
    )

    assert context["task_type"] == "architecture_design"
    assert context["field_meta"]["task_type"]["source"] == "deterministic"
    assert llm.calls == 0
