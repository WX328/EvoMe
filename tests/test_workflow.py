from app.runtime import EvoMeRuntime


class FakeLLM:
    def chat(self, messages, temperature=0.4):
        return "这是一个测试回复。"

    def json_chat(self, messages, temperature=0.1):
        system = messages[0]["content"]

        if "Evaluator" in system:
            return {
                "score": 0.9,
                "success": True,
                "reason": "测试通过",
            }

        return {
            "should_save": True,
            "memory_type": "experience",
            "importance": 0.8,
            "confidence": 0.9,
            "summary": "一次测试交互成功完成。",
            "lesson": "当前默认 Strategy 能完整跑通闭环。",
            "strategy_signal": "",
        }


def test_full_evolution_loop_skeleton(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM())

    result = runtime.run("测试 EvoMe 闭环")

    assert result["response"] == "这是一个测试回复。"
    assert result["evaluation_score"] == 0.9
    assert result["memory_saved"] is True

    memories = runtime.list_memories()
    assert len(memories) == 1
    assert "默认 Strategy" in memories[0].content
