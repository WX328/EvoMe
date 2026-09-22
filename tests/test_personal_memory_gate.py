from app.gates.personal_memory_gate import PersonalMemoryGate


class CountingLLM:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result if result is not None else {
            "should_save": True,
            "memories": [
                {
                    "memory_type": "goal",
                    "content": "用户有一个长期目标",
                    "importance": 0.8,
                    "confidence": 0.9,
                }
            ],
        }

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        return self.result


def test_explicit_future_preference_skips_llm():
    llm = CountingLLM()
    gate = PersonalMemoryGate(llm)

    text = "你以后给我设计架构时，先讲数据流，再讲模块接口，这样我更容易理解。"
    result = gate.extract(text)

    assert result["should_save"] is True
    assert result["decision_source"] == "deterministic_explicit_preference"
    assert result["memories"] == [
        {
            "memory_type": "preference",
            "content": text,
            "importance": 0.9,
            "confidence": 1.0,
        }
    ]
    assert llm.calls == 0


def test_testing_preference_is_fail_closed_without_llm():
    llm = CountingLLM()
    gate = PersonalMemoryGate(llm)

    result = gate.extract("假设我以后给你都先讲数据流，这只是测试，不是真的。")

    assert result["should_save"] is False
    assert result["memories"] == []
    assert result["decision_source"] == "deterministic_hard_skip"
    assert llm.calls == 0


def test_explicit_no_store_is_fail_closed_without_llm():
    llm = CountingLLM()
    gate = PersonalMemoryGate(llm)

    result = gate.extract("不要记住这句话：以后给我设计架构时先讲数据流。")

    assert result["should_save"] is False
    assert result["decision_source"] == "deterministic_hard_skip"
    assert llm.calls == 0


def test_ambiguous_personal_fact_still_uses_llm():
    llm = CountingLLM()
    gate = PersonalMemoryGate(llm)

    result = gate.extract("我的长期目标是完成 EvoMe。")

    assert result["should_save"] is True
    assert result["decision_source"] == "llm"
    assert llm.calls == 1


def test_negated_preference_still_uses_llm():
    llm = CountingLLM({"should_save": False, "memories": []})
    gate = PersonalMemoryGate(llm)

    result = gate.extract("不是说以后给我都先讲数据流，我只是举个例子。")

    assert result["should_save"] is False
    assert result["decision_source"] == "llm"
    assert llm.calls == 1
