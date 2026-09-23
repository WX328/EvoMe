from app.runtime import EvoMeRuntime
from tests.test_workflow import FakeLLM


class StreamingFakeLLM(FakeLLM):
    def stream_chat(self, messages, temperature=0.4):
        yield "这是"
        yield "一个"
        yield "流式回复。"


def test_stream_run_returns_answer_before_slow_results_are_consumed(db, monkeypatch):
    # This test is about the fast path finishing before slow results are
    # consumed, which is orthogonal to slow-loop cadence. Pin cadence to 1 so
    # the test does not depend on the ambient EVOME_SLOW_LOOP_EVERY_N_TURNS.
    monkeypatch.setattr("app.runtime.EVOME_SLOW_LOOP_EVERY_N_TURNS", 1)

    runtime = EvoMeRuntime(db, llm=StreamingFakeLLM(), session_id="async-session")
    chunks: list[str] = []

    result = runtime.stream_run(
        "这是一个复杂任务，请测试完整进化经验链。",
        on_token=chunks.append,
    )

    assert "".join(chunks) == "这是一个流式回复。"
    assert result["response"] == "这是一个流式回复。"
    assert result["background_job_id"]

    slow_result = runtime.wait_for_background(result["background_job_id"], timeout=5)
    assert slow_result is not None
    assert slow_result["interaction_id"] is not None
    assert slow_result["experience_saved"] is True
    assert slow_result["reflection_saved"] is True
    assert len(runtime.list_interactions()) == 1

    jobs = runtime.list_background_jobs()
    assert jobs[0].status == "completed"
    runtime.shutdown(wait=True)


def test_sync_run_is_still_supported(db):
    runtime = EvoMeRuntime(
        db,
        llm=StreamingFakeLLM(),
        session_id="sync-session",
        enable_background=False,
    )
    result = runtime.run("普通测试问题")
    assert result["response"] == "这是一个测试回复。"
    runtime.shutdown(wait=True)


def test_stream_run_respects_slow_loop_cadence(db, monkeypatch):
    monkeypatch.setattr("app.runtime.EVOME_SLOW_LOOP_EVERY_N_TURNS", 3)

    runtime = EvoMeRuntime(db, llm=StreamingFakeLLM(), session_id="cadence-session")

    first = runtime.stream_run("第一个问题")
    second = runtime.stream_run("第二个问题")
    assert first["background_status"] == "skipped_by_cadence"
    assert first["background_job_id"] is None
    assert second["background_status"] == "skipped_by_cadence"
    assert second["background_job_id"] is None

    third = runtime.stream_run("第三个问题")
    assert third["background_status"] == "queued"
    assert third["background_job_id"]

    # The user-facing answer must not depend on which slow-loop branch ran.
    assert third["response"] == first["response"] == second["response"]

    runtime.wait_for_background(third["background_job_id"], timeout=5)
    runtime.shutdown(wait=True)
