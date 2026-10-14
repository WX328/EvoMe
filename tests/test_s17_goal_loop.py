"""S17 Goal Stop hook regression suite (offline; no API keys / network).

The project owner runs pytest after applying the staged S17 patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.agent.goal_loop import (
    GoalController, GoalError, GoalEvaluation, PromptGoalEvaluator,
    _bounded_transcript,
)
from app.agent.loop import AgentLoop
from app.agent.schema import AgentCheckpoint
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookManager
from app.runtime import EvoMeRuntime
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.permissions import ApprovalStore, PermissionPolicy
from app.tools.registry import ToolRegistry


class FakeWorkerLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen = []
    def json_chat(self, messages, **kwargs):
        self.seen.append(list(messages))
        assert self.decisions, "Unexpected worker decision call"
        return self.decisions.pop(0)
    def chat(self, messages, **kwargs):
        return "Worker answer"


class FakeEvaluator:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen = []
    def evaluate(self, condition, messages):
        self.seen.append((condition, list(messages)))
        assert self.decisions
        result = self.decisions.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def final(answer="done", complete=True):
    return {"action": "final", "task_complete": complete, "answer": answer}


def make_goal(
    decisions, verdicts, *, max_steps=8, block_cap=3,
    max_turns=12, background_running=None, registry=None, hooks=None,
):
    worker = FakeWorkerLLM(decisions)
    evaluator = FakeEvaluator(verdicts)
    control = GoalController(
        evaluator, block_cap=block_cap, max_turns=max_turns,
        background_running=background_running,
    )
    agent = AgentLoop(
        llm=worker, tool_registry=registry or ToolRegistry(),
        max_steps=max_steps, hook_manager=hooks,
        permission_policy=PermissionPolicy(),
    )
    agent.goal_controller = control
    return agent, control, worker, evaluator


def test_no_active_goal_keeps_s01_normal_final_answer():
    agent, ctrl, llm, eval_ = make_goal([final("ordinary")], [])
    result = agent.run(user_input="普通对话")
    assert result.termination_reason == "final_answer"
    assert result.answer == "ordinary"
    assert eval_.seen == []
    assert ctrl.status()["status"] == "idle"


def test_explicit_goal_starts_immediately_and_needs_independent_verdict():
    agent, ctrl, llm, evaluator = make_goal(
        [final("premature"), final("verified")],
        [GoalEvaluation(False, "missing file evidence"),
         GoalEvaluation(True, "actual evidence exists")],
    )
    ctrl.set("完成审查并提供读取证据")
    outcome = agent.run(user_input=ctrl.condition)
    assert outcome.termination_reason == "goal_achieved"
    assert outcome.answer == "verified"
    assert ctrl.condition is None
    assert ctrl.status()["status"] == "achieved"
    assert len(evaluator.seen) == 2
    assert len(llm.seen) == 2
    assert any(
        "[GOAL_STOP_HOOK]" in item["content"]
        for item in llm.seen[1] if item["role"] == "user"
    )


def test_evaluator_receives_real_tool_evidence_not_only_worker_self_report():
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="read_file", description="safe read"))
    reads = []
    registry.register_implementation(ToolImplementation(
        name="fixture_read", tool_name="read_file",
        handler=lambda path: reads.append(path) or "read_file: 4 classes",
    ))
    agent, ctrl, _llm, evaluator = make_goal(
        [
            {"action": "tool", "tool_name": "read_file",
             "arguments": {"path": "app/agent/schema.py"}},
            final("4 classes verified"),
        ],
        [GoalEvaluation(True, "observed tool result")],
        registry=registry,
    )
    ctrl.set("确认 schema.py 的数据结构")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_achieved"
    assert reads == ["app/agent/schema.py"]
    transcript = evaluator.seen[0][1]
    assert any("4 classes" in item["content"]
               for item in transcript if item["role"] == "user")
    assert any(step.tool_call and step.tool_call.name == "read_file"
               for step in result.steps)


def test_goal_does_not_allow_model_only_success_without_gate():
    agent, ctrl, _worker, evaluator = make_goal(
        [final("done"), final("done"), final("done")],
        [GoalEvaluation(False, "no tool evidence"),
         GoalEvaluation(False, "still no evidence")],
        block_cap=1, max_steps=5,
    )
    ctrl.set("必须展示真实工具结果")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_limit"
    assert ctrl.condition is not None
    assert ctrl.status()["evaluations"] == 2
    assert ctrl.status()["status"] == "goal_limit"
    assert result.checkpoint is not None


def test_goal_evaluator_invalid_or_unavailable_stops_and_preserves_goal():
    agent, ctrl, worker, evaluator = make_goal(
        [final()],
        [RuntimeError("private source text")],
    )
    ctrl.set("读取并确认文件")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_error"
    assert ctrl.condition is not None
    assert ctrl.status()["evaluations"] == 0
    assert "private source text" not in ctrl.status()["last_reason"]


def test_goal_impossible_fails_without_more_agent_calls():
    agent, ctrl, worker, evaluator = make_goal(
        [final()],
        [GoalEvaluation(False, "evidence is impossible", impossible=True)],
    )
    ctrl.set("不可实现的要求")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_failed"
    assert ctrl.condition is None
    assert ctrl.status()["status"] == "failed"
    assert len(worker.seen) == 1


def test_goal_incomplete_model_cannot_override_evaluator_as_achieved():
    agent, ctrl, worker, evaluator = make_goal(
        [final(complete=False), final()],
        [GoalEvaluation(True, "evidence exists"),
         GoalEvaluation(True, "now worker also finished")],
    )
    ctrl.set("完成真实目标")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_achieved"
    assert len(worker.seen) == 2


def test_background_running_defers_gate_without_invoking_evaluator():
    busy = [True]
    agent, ctrl, worker, evaluator = make_goal(
        [final(), final()],
        [GoalEvaluation(True, "evidence appears after completion")],
        background_running=lambda: busy[0],
    )
    ctrl.set("等待 Workflow 完成再判定")
    first = agent.run(user_input=ctrl.condition)
    assert first.termination_reason == "goal_deferred"
    assert ctrl.status()["deferred_background"] is True
    assert ctrl.status()["has_checkpoint"] is True
    assert evaluator.seen == []
    busy[0] = False
    ctrl.begin_query()
    second = agent.run(user_input=ctrl.condition)
    assert second.termination_reason == "goal_achieved"
    assert len(evaluator.seen) == 1


def test_background_status_callback_error_defers_fail_closed():
    def broken():
        raise PermissionError("untrusted failure detail")
    agent, ctrl, worker, evaluator = make_goal(
        [final()], [], background_running=broken,
    )
    ctrl.set("后台状态确认")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_deferred"
    assert evaluator.seen == []


def test_goal_per_query_global_turn_budget_stops_without_evaluator_for_no_final():
    agent, ctrl, worker, evaluator = make_goal(
        [
            {"action": "nonsense"},
            {"action": "nonsense"},
            {"action": "nonsense"},
        ], [], max_steps=6, max_turns=2,
    )
    ctrl.set("做完才停止")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_limit"
    assert ctrl.condition is not None
    assert ctrl.status()["worker_turns"] == 2
    assert len(worker.seen) == 2
    assert evaluator.seen == []


def test_s01_step_budget_is_a_hard_exit_even_without_goal_evaluator_decision():
    agent, ctrl, worker, evaluator = make_goal(
        [{"action": "nonsense"}, {"action": "nonsense"}],
        [], max_steps=2,
    )
    ctrl.set("不能空口宣称成功")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_limit"
    assert result.checkpoint is not None
    assert ctrl.condition is not None


def test_interrupted_goal_can_resume_from_same_in_process_checkpoint():
    agent, ctrl, worker, evaluator = make_goal(
        [{"action": "nonsense"}, final("after user continue")],
        [GoalEvaluation(True, "goal checked")],
        max_steps=1,
    )
    ctrl.set("继续同一个目标")
    first = agent.run(user_input=ctrl.condition)
    assert first.termination_reason == "goal_limit"
    assert ctrl.status()["has_checkpoint"]
    ctrl.begin_query()
    second = agent.run(user_input=ctrl.condition)
    assert second.termination_reason == "goal_achieved"
    assert second.answer == "after user continue"
    assert ctrl.condition is None


def test_s03_approval_pauses_goal_and_approved_result_is_carried_on_resume():
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="Workflow", description="trusted script"))
    calls = []
    registry.register_implementation(ToolImplementation(
        name="workflow_stub", tool_name="Workflow",
        handler=lambda name: calls.append(name) or {"launched": True},
    ))
    agent, ctrl, worker, evaluator = make_goal(
        [
            {"action": "tool", "tool_name": "Workflow",
             "arguments": {"name": "review-files"}},
            final("workflow launched"),
            final("workflow queued, awaiting verification"),
        ],
        [
            GoalEvaluation(False, "workflow queued but not verified"),
            GoalEvaluation(False, "task output still missing"),
        ],
        registry=registry, max_steps=4, block_cap=1,
    )
    ctrl.set("运行并核实审查结果")
    first = agent.run(user_input=ctrl.condition)
    assert first.termination_reason == "approval_required"
    assert calls == []
    approval_id = first.approval_id
    assert ctrl.waiting_approval_id == approval_id
    result = agent.executor.approve(approval_id)
    assert result.success
    assert calls == ["review-files"]
    ctrl.note_approval(approval_id, success=True, status=result.status)
    assert ctrl.waiting_approval_id == ""
    ctrl.begin_query()
    second = agent.run(user_input=ctrl.condition)
    assert second.termination_reason == "goal_limit"
    assert any(
        "[GOAL_APPROVAL_OUTCOME]" in message["content"]
        for message in worker.seen[1] if message["role"] == "user"
    )
    assert calls == ["review-files"]


def test_goal_clear_releases_checkpoint_and_status():
    agent, ctrl, worker, evaluator = make_goal([final()], [GoalEvaluation(False, "missing")], max_steps=1)
    ctrl.set("保留目标")
    agent.run(user_input=ctrl.condition)
    assert ctrl.status()["has_checkpoint"]
    cleared = ctrl.clear()
    assert cleared["active"] is False
    assert cleared["has_checkpoint"] is False
    assert cleared["status"] == "cleared"


def test_new_goal_replaces_old_condition_without_reusing_checkpoint():
    ctrl = GoalController(FakeEvaluator([]))
    ctrl.set("old")
    ctrl._checkpoint = AgentCheckpoint(
        user_input="old", messages=(), steps=(), seen_calls=frozenset(),
    )
    ctrl.set("new")
    assert ctrl.resume_checkpoint("new") is None
    assert ctrl.status()["worker_turns"] == 0


@pytest.mark.parametrize("condition", ["", "x" * 1801, None, 123])
def test_goal_rejects_missing_or_oversized_input(condition):
    ctrl = GoalController(FakeEvaluator([]))
    with pytest.raises(GoalError):
        ctrl.set(condition)


def test_goal_only_gates_explicit_condition_not_unrelated_chat():
    agent, ctrl, worker, evaluator = make_goal(
        [final("ordinary")], [],
    )
    ctrl.set("my active goal")
    result = agent.run(user_input="unrelated conversation")
    assert result.termination_reason == "final_answer"
    assert ctrl.condition == "my active goal"
    assert evaluator.seen == []


def test_goal_logs_no_sensitive_input_from_evaluator_errors():
    ctrl = GoalController(FakeEvaluator([RuntimeError("API_KEY_SECRET")]))
    ctrl.set("verify without secrets")
    decision = ctrl.after_final([{"role": "user", "content": "test"}], model_claimed_complete=True)
    assert decision.action == "error"
    assert "API_KEY_SECRET" not in str(decision)
    assert ctrl.condition is not None


def test_prompt_evaluator_has_no_tools_and_parses_strict_json():
    seen = []
    class EvaluatorLLM:
        def json_chat(self, messages, **kwargs):
            seen.append((messages, kwargs))
            return {"ok": False, "reason": "pytest exit code not seen", "impossible": False}
    gate = PromptGoalEvaluator(EvaluatorLLM, max_tokens=256)
    result = gate.evaluate(
        "pytest exits with code 0",
        [{"role": "user", "content": "I claim success without result"}],
    )
    assert result.ok is False
    assert "exit code" in result.reason
    assert len(seen) == 1
    messages, kwargs = seen[0]
    assert all("tools" not in item for item in messages)
    assert messages[0]["role"] == "system"
    assert kwargs["purpose"] == "goal_evaluation"
    assert kwargs["max_tokens"] == 256


@pytest.mark.parametrize("reply", [
    {"ok": "true", "reason": "yes"},
    {"ok": True, "reason": ""},
    {"ok": True, "reason": "yes", "impossible": True},
    {"ok": False, "reason": "no", "impossible": "yes"},
    {"reason": "missing ok"},
    [],
])
def test_evaluator_rejects_invalid_or_contradictory_outputs(reply):
    with pytest.raises(GoalError):
        PromptGoalEvaluator._parse(reply)


def test_evaluator_transcript_budget_preserves_recent_message_ends():
    result = _bounded_transcript([
        {"role": "user", "content": "early"},
        {"role": "user", "content": "prefix" + "x" * 15000 + "tail-marker"},
    ], limit=150)
    assert "tail-marker" in result
    assert "early" not in result
    assert "middle omitted" in result
    assert len(result) <= 150


def test_goal_background_notifications_are_untrusted_and_deduplicated():
    ctrl = GoalController(FakeEvaluator([]))
    ctrl.set("wait for background result")
    ctrl._checkpoint = AgentCheckpoint(
        user_input=ctrl.condition, messages=({"role": "user", "content": "start"},),
        steps=(), seen_calls=frozenset(),
    )
    item = {"source": "s16_workflow", "ref": "wf_123456789abc",
            "status": "completed", "kind": "task_notification",
            "content": "SENSITIVE PRIVATE OUTPUT SHOULD NOT BE FORWARDED"}
    ctrl.note_runtime_events([item, item])
    snapshot = ctrl.resume_checkpoint(ctrl.condition)
    assert snapshot is not None
    hints = [x["content"] for x in snapshot.messages if "GOAL_BACKGROUND_OBSERVATION" in x["content"]]
    assert len(hints) == 1
    assert "SENSITIVE PRIVATE" not in str(hints)
    assert "untrusted" in hints[0]


def test_goal_event_bridge_ignores_non_async_sources():
    ctrl = GoalController(FakeEvaluator([]))
    ctrl.set("notify me")
    ctrl._checkpoint = AgentCheckpoint(
        user_input=ctrl.condition, messages=(), steps=(), seen_calls=frozenset(),
    )
    ctrl.note_runtime_events([{"source": "s03_permission", "ref": "secret",
                               "status": "approved", "kind": "approval"}])
    assert not ctrl._approval_events


def test_goal_cli_commands_not_learned_as_preferences():
    for item in ("goal", "goal continue", "goal clear",
                 "goal 完成代码结构审查"):
        assert normalize_cli_input(item) == "/" + item
        assert looks_like_operational_input(item)


def test_session_goal_logic_keeps_preexisting_s04_hooks():
    hooks = HookManager()
    agent, ctrl, worker, evaluator = make_goal(
        [final("completed")], [GoalEvaluation(True, "proof exists")], hooks=hooks,
    )
    ctrl.set("finish")
    result = agent.run(user_input=ctrl.condition)
    assert result.termination_reason == "goal_achieved"
    events = [item.event for item in hooks.recent()]
    assert "before_agent" in events
    assert "after_agent" in events


def test_runtime_goal_internal_turn_never_enqueues_learning_or_stores_memory():
    runtime = object.__new__(EvoMeRuntime)
    # __new__ intentionally skips __init__, so provide the session state
    # that stream_run normally receives from EvoMeRuntime initialization.
    runtime.session_id = "s17-fixture-session"
    class FakeStrategy:
        name = "fixture"
        version = 1
        def prepare_fast_state(self, **kwargs):
            assert kwargs["session_id"] == "s17-fixture-session"
            return {"user_input": kwargs["user_input"],
                    "relevant_memories": [], "agent_steps": []}
        def stream_response(self, state):
            yield "S17 resumed"
    runtime.strategy = FakeStrategy()
    ctrl = GoalController(FakeEvaluator([]))
    ctrl.set("run goal")
    runtime.goal_controller = ctrl
    runtime.background_queue = object()
    runtime._maybe_stage_agent_memory = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("must never stage memory during Goal control")
    )
    output = runtime.stream_run("run goal", goal_internal=True)
    assert output["response"] == "S17 resumed"
    assert output["background_job_id"] is None
    assert output["background_status"] == "goal_only"
    with pytest.raises(GoalError):
        runtime.stream_run("different text", goal_internal=True)


def test_runtime_background_probe_tracks_s16_active_runs_only():
    runtime = object.__new__(EvoMeRuntime)
    runtime.workflow_runtime = SimpleNamespace(
        current_runs=lambda: [{"status": "running", "id": "wf_001"}]
    )
    runtime.task_background_service = None
    runtime.team_runtime = None
    assert runtime._goal_background_running() is True
    runtime.workflow_runtime = SimpleNamespace(
        current_runs=lambda: [{"status": "completed", "id": "wf_001"}]
    )
    assert runtime._goal_background_running() is False


def test_runtime_goal_set_and_clear_do_not_spawn_any_worker():
    runtime = object.__new__(EvoMeRuntime)
    ctrl = GoalController(FakeEvaluator([]))
    runtime.goal_controller = ctrl
    runtime.deny_tool_call = lambda _id: False
    assert runtime.goal_set("检查 4 个类")["active"]
    assert runtime.goal_status()["condition"] == "检查 4 个类"
    assert runtime.goal_clear()["active"] is False


def test_goal_remains_active_if_answer_generation_fails_after_positive_verdict():
    class BadAnswerLLM:
        def json_chat(self, messages, **kwargs):
            return {"action": "final", "task_complete": True}
        def chat(self, messages, **kwargs):
            raise RuntimeError("simulated final answer failure")
    ctrl = GoalController(FakeEvaluator([GoalEvaluation(True, "evidence sufficient")]))
    ctrl.set("核验完成再输出")
    agent = AgentLoop(
        llm=BadAnswerLLM(), tool_registry=ToolRegistry(), max_steps=3,
    )
    agent.goal_controller = ctrl
    with pytest.raises(RuntimeError, match="simulated final answer"):
        agent.run(user_input=ctrl.condition)
    # A verdict is not an observable delivered answer.
    assert ctrl.condition == "核验完成再输出"
    assert ctrl.status()["status"] == "achieved_pending_answer"


def test_goal_checkpoint_keeps_operator_approval_note_when_next_model_fails():
    controller = GoalController(FakeEvaluator([]))
    controller.set("confirmed work")
    controller._checkpoint = AgentCheckpoint(
        user_input="confirmed work",
        messages=({"role": "user", "content": "start"},),
        steps=(), seen_calls=frozenset(),
    )
    controller.waiting_approval_id = "approval_123"
    controller.note_approval("approval_123", success=True, status="executed")
    first = controller.resume_checkpoint("confirmed work")
    second = controller.resume_checkpoint("confirmed work")
    assert first is not None and second is not None
    assert any("GOAL_APPROVAL_OUTCOME" in item["content"]
               for item in second.messages)
