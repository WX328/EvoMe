from app.runtime import EvoMeRuntime


class FakeLLM:
    def chat(self, messages, temperature=0.4):
        return "这是一个测试回复。"

    def json_chat(self, messages, temperature=0.1):
        system = messages[0]["content"]
        user_text = messages[-1]["content"]

        if "Evaluator" in system:
            return {"score": 0.9, "success": True, "reason": "测试通过"}

        if "Personal Memory Gate" in system:
            if "正在开发 EvoMe" in user_text:
                return {
                    "should_save": True,
                    "memories": [{
                        "memory_type": "project",
                        "content": "用户正在开发 EvoMe，目标是构建长期自进化 Personal AI。",
                        "importance": 0.95,
                        "confidence": 1.0,
                    }],
                }
            return {"should_save": False, "memories": []}

        if system.strip().startswith("你是 EvoMe 的 Experience Gate"):
            if "复杂任务" in user_text:
                return {
                    "should_save": True,
                    "task_type": "complex_task",
                    "result_summary": "EvoMe 使用默认 Strategy 完成了一次有学习价值的复杂任务。",
                    "reason": "值得形成执行经验",
                }
            return {
                "should_save": False,
                "task_type": "general",
                "result_summary": "",
                "reason": "普通交互，无需形成进化经验",
            }

        if "Reflection 模块" in system:
            return {
                "reflection_type": "success",
                "observation": "当前 Strategy 能完成复杂任务闭环。",
                "lesson": "应继续收集真实任务结果，再决定是否调整 Strategy。",
                "strategy_signal": "",
                "skill_signal": "",
                "confidence": 0.85,
            }
        return {}


def test_personal_memory_and_audit_are_separated(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM(), session_id="test-session")

    first = runtime.run("我正在开发 EvoMe。")
    assert first["personal_memories_saved"] == 1
    assert len(runtime.list_personal_memories()) == 1
    assert len(runtime.list_interactions()) == 1

    second = runtime.run("EvoMe 是什么项目？")
    assert second["personal_memories_saved"] == 0
    assert len(runtime.list_personal_memories()) == 1
    assert len(runtime.list_interactions()) == 2
    assert len(runtime.list_experiences()) == 0


def test_experience_triggers_reflection(db):
    runtime = EvoMeRuntime(db, llm=FakeLLM(), session_id="test-session")
    result = runtime.run("这是一个复杂任务，请测试完整进化经验链。")

    assert result["experience_saved"] is True
    assert result["reflection_saved"] is True

    experiences = runtime.list_experiences()
    reflections = runtime.list_reflections()
    assert len(experiences) == 1
    assert len(reflections) == 1
    assert reflections[0].experience_id == experiences[0].id
