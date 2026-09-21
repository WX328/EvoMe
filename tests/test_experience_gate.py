from app.gates.experience_gate import ExperienceGate


class CountingLLM:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result or {
            "should_save": False,
            "task_type": "general",
            "result_summary": "",
            "reason": "ambiguous ordinary interaction",
        }

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        return dict(self.result)


def evaluate(gate: ExperienceGate, user_input: str, *, score=0.95, success=True):
    return gate.evaluate(
        user_input=user_input,
        assistant_response="给出了一份结构清晰、可执行的回答。",
        evaluation={"score": score, "success": success, "reason": "test"},
        strategy_name="default_conversation_strategy_v2",
        skills_used=["recall_relevant_memory"],
    )


def test_structural_architecture_task_hard_passes_without_llm():
    llm = CountingLLM()
    gate = ExperienceGate(llm)

    result = evaluate(
        gate,
        "请和我一起检查 PersonalMemory 和 Experience 两个模块的数据边界、职责边界和数据流。",
    )

    assert result["should_save"] is True
    assert result["task_type"] == "architecture_design"
    assert result["decision_source"] == "deterministic_hard_pass"
    assert llm.calls == 0


def test_short_definition_hard_skips_without_llm():
    llm = CountingLLM()
    gate = ExperienceGate(llm)

    result = evaluate(gate, "Python 的 list 是什么？")

    assert result["should_save"] is False
    assert result["decision_source"] == "deterministic_hard_skip"
    assert llm.calls == 0


def test_failure_hard_passes_even_for_ordinary_task():
    llm = CountingLLM()
    gate = ExperienceGate(llm)

    result = evaluate(gate, "请解释一下这个问题。", score=0.4, success=False)

    assert result["should_save"] is True
    assert result["decision_source"] == "deterministic_hard_pass"
    assert "failure" in result["reason"]
    assert llm.calls == 0


def test_ambiguous_case_falls_back_to_llm():
    llm = CountingLLM(
        {
            "should_save": True,
            "task_type": "general",
            "result_summary": "有一定学习价值",
            "reason": "LLM judged ambiguous case valuable",
        }
    )
    gate = ExperienceGate(llm)

    result = evaluate(gate, "帮我看看这个方案是否合理。")

    assert result["should_save"] is True
    assert result["decision_source"] == "llm"
    assert llm.calls == 1
