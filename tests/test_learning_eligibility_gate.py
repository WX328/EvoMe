from app.gates.learning_eligibility_gate import LearningEligibilityGate


class CountingLLM:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result if result is not None else {
            "category": "natural",
            "allow_personal_memory": False,
            "allow_evolution": False,
            "reason": "no durable learning signal",
        }

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        return self.result


def test_operational_input_is_audit_only_without_llm():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("revision-signals")

    assert result["category"] == "operational"
    assert result["learning_eligible"] is False
    assert result["allow_personal_memory"] is False
    assert result["allow_evolution"] is False
    assert llm.calls == 0


def test_noise_is_audit_only_without_llm():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("......")

    assert result["category"] == "noise"
    assert result["learning_eligible"] is False
    assert llm.calls == 0


def test_testing_input_does_not_write_simulated_user_fact():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("假设我喜欢 Java，这只是测试，不是真的。")

    assert result["category"] == "testing"
    assert result["allow_personal_memory"] is False
    assert result["allow_evolution"] is False
    assert llm.calls == 0


def test_testing_can_opt_into_evolution_but_never_personal_memory():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("这只是测试，把这次测试结果作为经验记录下来。")

    assert result["category"] == "testing"
    assert result["allow_personal_memory"] is False
    assert result["allow_evolution"] is True
    assert llm.calls == 0


def test_pure_user_fact_uses_memory_lane_only():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("我正在开发 EvoMe。")

    assert result["category"] == "natural"
    assert result["allow_personal_memory"] is True
    assert result["allow_evolution"] is False
    assert llm.calls == 0


def test_meta_observation_is_audit_only():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("你现在有哪些 hypothesis？")

    assert result["category"] == "meta"
    assert result["allow_personal_memory"] is False
    assert result["allow_evolution"] is False
    assert llm.calls == 0


def test_real_structural_task_can_enter_evolution():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("请检查 EvoMe 的模块职责、数据流和边界，然后给出修复方案。")

    assert result["category"] == "natural"
    assert result["allow_evolution"] is True
    assert llm.calls == 0


def test_ordinary_knowledge_question_is_audit_only():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("Python 的 list 是什么？")

    assert result["category"] == "natural"
    assert result["learning_eligible"] is False
    assert llm.calls == 0


def test_explicit_feedback_enters_evolution():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("你刚才讲错了，应该先解释数据流再讲接口。")

    assert result["category"] == "natural"
    assert result["allow_evolution"] is True
    assert llm.calls == 0


def test_ambiguous_input_uses_llm():
    llm = CountingLLM({
        "category": "natural",
        "allow_personal_memory": False,
        "allow_evolution": True,
        "reason": "real task execution",
    })
    gate = LearningEligibilityGate(llm)

    result = gate.classify("帮我看看这个方案是否合理。")

    assert result["allow_evolution"] is True
    assert result["decision_source"] == "llm"
    assert llm.calls == 1


def test_invalid_llm_output_fails_closed():
    llm = CountingLLM({})
    gate = LearningEligibilityGate(llm)

    result = gate.classify("这是一段无法由确定性规则判断的内容")

    assert result["learning_eligible"] is False
    assert result["decision_source"] == "llm_fail_closed"
    assert llm.calls == 1


def test_real_debugging_question_is_not_mistaken_for_ordinary_qa():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("为什么我的 FastAPI 接口一直报错 502？")

    assert result["category"] == "natural"
    assert result["allow_evolution"] is True
    assert llm.calls == 0


def test_explicit_future_explanation_preference_routes_to_both_lanes():
    llm = CountingLLM()
    gate = LearningEligibilityGate(llm)

    result = gate.classify("你以后给我设计架构时，先讲数据流，再讲模块接口，这样我更容易理解。")

    assert result["category"] == "natural"
    assert result["allow_personal_memory"] is True
    assert result["allow_evolution"] is True
    assert result["decision_source"] == "deterministic_hard_pass"
    assert llm.calls == 0
