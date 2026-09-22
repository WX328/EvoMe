from app.runtime import EvoMeRuntime
from tests.test_workflow import FakeLLM


class EligibilityAwareFakeLLM(FakeLLM):
    def json_chat(self, messages, temperature=0.1):
        system = messages[0]["content"]
        if "MODULE: LEARNING_ELIGIBILITY_GATE" in system:
            # Tests below intentionally use deterministic eligibility paths.
            raise AssertionError("deterministic eligibility rule should have handled this input")
        return super().json_chat(messages, temperature=temperature)


def test_simulated_preference_is_audit_only(db):
    runtime = EvoMeRuntime(
        db,
        llm=EligibilityAwareFakeLLM(),
        session_id="eligibility-test",
        enable_background=False,
    )

    result = runtime.run("假设我喜欢 Java，这只是测试，不是真的。")

    assert result["learning_category"] == "testing"
    assert result["allow_personal_memory"] is False
    assert result["allow_evolution"] is False
    assert len(runtime.list_interactions()) == 1
    assert len(runtime.list_personal_memories()) == 0
    assert len(runtime.list_experiences()) == 0
    assert len(runtime.list_reflections()) == 0

    metadata = runtime.list_interactions()[0].record_metadata
    assert metadata["learning_eligibility"]["category"] == "testing"
    assert "evaluation" not in metadata


def test_user_fact_can_write_memory_without_creating_experience(db):
    runtime = EvoMeRuntime(
        db,
        llm=EligibilityAwareFakeLLM(),
        session_id="memory-only-test",
        enable_background=False,
    )

    result = runtime.run("我正在开发 EvoMe。")

    assert result["allow_personal_memory"] is True
    assert result["allow_evolution"] is False
    assert result["personal_memories_saved"] == 1
    assert len(runtime.list_personal_memories()) == 1
    assert len(runtime.list_experiences()) == 0


def test_meta_observation_does_not_create_evolution_data(db):
    runtime = EvoMeRuntime(
        db,
        llm=EligibilityAwareFakeLLM(),
        session_id="meta-test",
        enable_background=False,
    )

    result = runtime.run("你现在有哪些 hypothesis？")

    assert result["learning_category"] == "meta"
    assert result["learning_eligible"] is False
    assert len(runtime.list_interactions()) == 1
    assert len(runtime.list_experiences()) == 0
    assert len(runtime.list_hypotheses()) == 0


def test_real_architecture_task_still_enters_evolution(db):
    runtime = EvoMeRuntime(
        db,
        llm=EligibilityAwareFakeLLM(),
        session_id="real-task-test",
        enable_background=False,
    )

    result = runtime.run(
        "请和我一起检查 PersonalMemory 和 Experience 两个模块的数据边界、职责边界和数据流。"
    )

    assert result["allow_evolution"] is True
    assert result["experience_saved"] is True
    assert result["reflection_saved"] is True
