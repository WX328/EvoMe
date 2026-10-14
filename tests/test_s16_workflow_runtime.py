"""S16 journaled Workflow tests. All runners are fakes; zero external API use.

As agreed, the project owner executes pytest after applying the patch.
"""
from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest

from app.agent.workflow_store import WorkflowStore
from app.agent.workflow_runtime import (
    ExecutionState, WorkflowRuntime, WorkflowInputError, ScriptRegistry,
    OUTPUT_SCHEMA, _validate_schema, ReadOnlyAgentRunner,
)
from app.agent.integrated_harness import IntegratedHarness
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookContext
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.executor import ToolExecutor
from app.tools.permissions import ApprovalStore, PermissionPolicy
from app.tools.registry import ToolRegistry


P1 = "app/agent/schema.py"
P2 = "app/background.py"


class FakeRunner:
    def __init__(self, *, fail_label=None, invalid_once=False,
                 started: Event | None = None, release: Event | None = None):
        self.calls = []
        self.lock = Lock()
        self.fail_label = fail_label
        self.invalid_once = invalid_once
        self.started = started
        self.release = release

    def run(self, prompt, schema, label):
        with self.lock:
            self.calls.append(label)
            index = len(self.calls)
            should_fail = label == self.fail_label
            if should_fail:
                self.fail_label = None
            should_invalid = self.invalid_once
            if should_invalid:
                self.invalid_once = False
        if self.started:
            self.started.set()
        if self.release and not self.release.wait(5):
            raise RuntimeError("simulated blocked research")
        if should_fail:
            raise RuntimeError("simulated model failure")
        if should_invalid:
            return {"not_the_schema": "broken"}
        return {
            "summary": f"Grounded {label} using source evidence",
            "paths": [P1], "verified": True,
        }


def make(tmp_path, runner=None, *, max_calls=8, max_parallel=2):
    store = WorkflowStore(tmp_path)
    stub = runner or FakeRunner()
    runtime = WorkflowRuntime(
        store=store, runner_factory=lambda: stub,
        workspace_root=tmp_path,
        max_calls=max_calls, max_parallel=max_parallel,
    )
    return store, runtime, stub


def finish(runtime, run_id):
    return runtime.wait(run_id, timeout=5)


def test_no_files_or_workers_are_started_on_read_only_construction(tmp_path):
    store, runtime, runner = make(tmp_path)
    try:
        assert runtime.registry.list()[0]["name"] == "review-files"
        assert runtime.list() == []
        assert not (tmp_path / ".workflows").exists()
        assert not runner.calls
        assert runtime.current_runs() == []
    finally:
        runtime.shutdown()


def test_one_file_pipeline_runs_both_stages_and_persists_result(tmp_path):
    store, runtime, runner = make(tmp_path)
    try:
        accepted = runtime.launch("review-files", {"paths": [P1]})
        assert accepted["launched"]
        run_id = accepted["run_id"]
        result = finish(runtime, run_id)
        assert result["status"] == "completed", result
        assert result["agent_calls"] == 2
        assert result["phase"] == "Report"
        assert len(store.journal(run_id)) == 2
        assert result["result"]["status"] == "reviewed"
        assert result["result"]["files"][0]["path"] == P1
        assert {label.split(":")[0] for label in runner.calls} == {"audit", "verify"}
        kinds = [e["kind"] for e in runtime.events(run_id)]
        assert "async_launched" in kinds
        assert "task_started" in kinds
        assert "phase" in kinds
        assert "task_notification" in kinds
        assert all(store._path(run_id, suffix).exists() for suffix in (
            "json", "journal.jsonl", "output.json",
        ))
        assert not store.locked(run_id)
    finally:
        runtime.shutdown()


def test_multiple_files_pipeline_progresses_independently(tmp_path):
    store, runtime, runner = make(tmp_path)
    try:
        job = runtime.launch("review-files", {"paths": [P1, P2]})
        state = finish(runtime, job["run_id"])
        assert state["status"] == "completed"
        assert len(runner.calls) == 4
        assert len(store.journal(job["run_id"])) == 4
        assert [row["path"] for row in state["result"]["files"]] == [P1, P2]
    finally:
        runtime.shutdown()


def test_exactly_two_agent_calls_run_in_parallel(tmp_path):
    began = Event()
    release = Event()
    runner = FakeRunner(started=began, release=release)
    store, runtime, _ = make(tmp_path, runner=runner)
    try:
        accepted = runtime.launch("review-files", {"paths": [P1, P2]})
        assert began.wait(4)
        # The one worker thread running asyncio can dispatch up to two
        # parallel agent threads, not one serial task at a time.
        import time
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if len(runner.calls) >= 2:
                break
            time.sleep(0.025)
        assert len(runner.calls) == 2
        assert store.load(accepted["run_id"])["status"] == "running"
        release.set()
        assert finish(runtime, accepted["run_id"])["status"] == "completed"
    finally:
        release.set()
        runtime.shutdown()


def test_failure_records_progress_and_preserves_successful_journal(tmp_path):
    runner = FakeRunner(fail_label=f"verify:1:{P2}")
    store, runtime, _ = make(tmp_path, runner=runner)
    try:
        job = runtime.launch("review-files", {"paths": [P1, P2]})
        state = finish(runtime, job["run_id"])
        assert state["status"] == "failed", state
        assert state["error"] == "RuntimeError"
        assert len(store.journal(job["run_id"])) >= 2
        assert not store.locked(job["run_id"])
        assert store.read_output(job["run_id"]) is None
    finally:
        runtime.shutdown()


def test_resuming_failed_run_reuses_stable_semantic_agent_keys(tmp_path):
    runner = FakeRunner(fail_label=f"verify:1:{P2}")
    store, runtime, _ = make(tmp_path, runner=runner)
    try:
        launched = runtime.launch("review-files", {"paths": [P1, P2]})
        first = finish(runtime, launched["run_id"])
        assert first["status"] == "failed"
        old = len(runner.calls)
        cached = len(store.journal(launched["run_id"]))
        again = runtime.launch("review-files", resume_from_run_id=launched["run_id"])
        assert again["launched"]
        state = finish(runtime, launched["run_id"])
        assert state["status"] == "completed", state
        assert len(store.journal(launched["run_id"])) == 4
        assert len(runner.calls) - old == 4 - cached
        assert any("cached" in e["message"] for e in state["progress"])
        assert state["result"]["count"] == 2
    finally:
        runtime.shutdown()


def test_resuming_successful_run_returns_cached_output_without_execution(tmp_path):
    store, runtime, runner = make(tmp_path)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        first = finish(runtime, run_id)
        call_count = len(runner.calls)
        result = runtime.launch("review-files", resume_from_run_id=run_id)
        assert result["launched"] is False
        assert result["status"] == "completed"
        assert result["result"] == first["result"]
        assert len(runner.calls) == call_count
    finally:
        runtime.shutdown()


def test_resume_rejects_mutated_args_and_wrong_name_without_touching_snapshot(tmp_path):
    store, runtime, _ = make(tmp_path)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        finish(runtime, run_id)
        before = store.load(run_id)
        with pytest.raises(WorkflowInputError, match="mismatch"):
            runtime.launch("review-files", {"paths": [P2]}, resume_from_run_id=run_id)
        with pytest.raises(WorkflowInputError):
            runtime.launch("not-registered", resume_from_run_id=run_id)
        assert store.load(run_id) == before
    finally:
        runtime.shutdown()


def test_active_run_refuses_second_launch_and_active_resume(tmp_path):
    started, release = Event(), Event()
    store, runtime, _ = make(
        tmp_path, runner=FakeRunner(started=started, release=release),
    )
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        assert started.wait(4)
        with pytest.raises(RuntimeError, match="one active"):
            runtime.launch("review-files", {"paths": [P2]})
        with pytest.raises(RuntimeError, match="one active"):
            runtime.launch("review-files", resume_from_run_id=run_id)
        release.set()
        assert finish(runtime, run_id)["status"] == "completed"
    finally:
        release.set()
        runtime.shutdown()


def test_os_lock_rejects_multi_process_resumption(tmp_path):
    store, runtime, _ = make(tmp_path)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        finish(runtime, run_id)
        store.lock(run_id)
        try:
            with pytest.raises(FileExistsError):
                store.lock(run_id)
            with pytest.raises(FileExistsError):
                runtime.launch("review-files", resume_from_run_id=run_id)
        finally:
            store.unlock(run_id)
    finally:
        runtime.shutdown()


def test_incomplete_journal_line_fails_closed_before_locking(tmp_path):
    store, runtime, _ = make(tmp_path)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        finish(runtime, run_id)
        path = store._path(run_id, "journal.jsonl")
        path.write_text('{"truncated":', encoding="utf-8")
        with pytest.raises(ValueError, match="corrupt"):
            runtime.launch("review-files", resume_from_run_id=run_id)
        assert not store.locked(run_id)
        assert store._path(run_id, "json").exists()
    finally:
        runtime.shutdown()


def test_corrupted_snapshot_refuses_to_resume(tmp_path):
    store, runtime, _ = make(tmp_path)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        finish(runtime, run_id)
        store._path(run_id, "json").write_text('{"nonsense":true}', encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            runtime.launch("review-files", resume_from_run_id=run_id)
    finally:
        runtime.shutdown()


def test_unknown_scripts_and_code_like_args_are_never_executed(tmp_path):
    store, runtime, _ = make(tmp_path)
    try:
        with pytest.raises(WorkflowInputError):
            runtime.launch("unknown", {"paths": [P1]})
        for value in (
            {"code": "import os"}, {"paths": ["../secrets.txt"]},
            {"paths": [".env"]}, {"paths": ["C:\\Windows\\system.ini"]},
            {"paths": [P1, P1]}, {"paths": []},
            {"paths": [P1, P2, P1, "other.py"]},
            {"paths": P1}, {"paths": [123]},
        ):
            with pytest.raises((WorkflowInputError, PermissionError, ValueError)):
                runtime.launch("review-files", value)
        assert not (tmp_path / ".workflows").exists()
    finally:
        runtime.shutdown()


def test_schema_invalid_result_retries_once_then_succeeds(tmp_path):
    runner = FakeRunner(invalid_once=True)
    store, runtime, _ = make(tmp_path, runner=runner)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        state = finish(runtime, run_id)
        assert state["status"] == "completed", state
        assert len(runner.calls) == 3
        assert state["agent_calls"] == 3
        assert len(store.journal(run_id)) == 2
        assert any("retry_invalid_schema" in item["message"] for item in state["progress"])
    finally:
        runtime.shutdown()


def test_schema_twice_invalid_fails_without_journal_poisoning(tmp_path):
    class InvalidRunner:
        def run(self, prompt, schema, label):
            return {"broken": "invalid"}
    store, runtime, _ = make(tmp_path, runner=InvalidRunner())
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        state = finish(runtime, run_id)
        assert state["status"] == "failed"
        assert store.journal(run_id) == {}
    finally:
        runtime.shutdown()


def test_bounded_agent_budget_stops_script_without_partial_output(tmp_path):
    store, runtime, _ = make(tmp_path, max_calls=1)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        state = finish(runtime, run_id)
        assert state["status"] == "failed"
        assert len(store.journal(run_id)) <= 1
        assert store.read_output(run_id) is None
    finally:
        runtime.shutdown()


def test_workflow_registry_validates_metadata_and_rejects_duplicates():
    r = ScriptRegistry()
    async def harmless(ctx, args):
        return {"status": "ok"}
    with pytest.raises(WorkflowInputError):
        r.register(name="../bad", description="bad", phases=("A",), fn=harmless)
    with pytest.raises(WorkflowInputError):
        r.register(name="valid", description="", phases=("A",), fn=harmless)
    r.register(name="valid", description="good", phases=("A",), fn=harmless)
    with pytest.raises(WorkflowInputError):
        r.register(name="valid", description="duplicate", phases=("A",), fn=harmless)
    with pytest.raises(WorkflowInputError):
        r.get("code_from_model")


def test_nested_scripts_limited_to_one_level(tmp_path):
    store, runtime, runner = make(tmp_path)
    snapshot = store.reserve("review-files", {"paths": [P1]})
    ctx = ExecutionState(
        run_id=snapshot["id"], store=store, runner=runner,
        meta=runtime.registry.get("review-files")[0],
    )
    registry = ScriptRegistry()
    async def inner(ctx, args):
        with pytest.raises(WorkflowInputError, match="nested"):
            await ctx.workflow("other", {}, registry)
        return {"x": 1}
    registry.register(name="other", description="allowed nested", phases=("A",), fn=inner)
    try:
        assert asyncio.run(ctx.workflow("other", {}, registry)) == {"x": 1}
    finally:
        store.unlock(snapshot["id"])
        runtime.shutdown()


def test_semantic_key_does_not_depend_on_completion_order():
    a = ExecutionState._key("inspect code", OUTPUT_SCHEMA, "audit:path")
    b = ExecutionState._key("inspect code", OUTPUT_SCHEMA, "audit:path")
    c = ExecutionState._key("inspect different code", OUTPUT_SCHEMA, "audit:path")
    assert a == b and a != c
    assert a.startswith("agent-") and len(a) == len("agent-") + 32


def test_private_workflow_path_cannot_be_written_through_s03(tmp_path):
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_workflow_directory=tmp_path / ".workflows",
    )
    for tool in ("write_file", "edit_file"):
        for path in (".workflows/wf_aaaaaaaaaaaa.journal.jsonl",
                     "sub/../.workflows/wf_aaaaaaaaaaaa.json"):
            assert policy.evaluate(tool, {"path": path, "content": "tampered"}).action == "deny"


def test_workflow_tool_is_high_risk_approval_required_and_does_not_launch_early(tmp_path):
    store, runtime, runner = make(tmp_path)
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(
        name="Workflow", description="pre-saved trusted Workflow",
        input_schema={"name": {"type": "string"}, "args": {"type": "object"}},
    ))
    registry.register_implementation(ToolImplementation(
        name="s16_test", tool_name="Workflow", handler=runtime.run_tool,
    ))
    approvals = ApprovalStore()
    executor = ToolExecutor(registry=registry, approval_store=approvals,
                            permission_policy=PermissionPolicy(workspace_root=tmp_path))
    try:
        decision = executor.execute("Workflow", {
            "name": "review-files", "args": {"paths": [P1]},
        })
        assert decision.status == "approval_required"
        assert decision.risk == "high"
        assert not runner.calls
        assert store.list() == []
        result = executor.approve(decision.approval_id)
        assert result.success
        assert result.output["launched"]
        assert finish(runtime, result.output["run_id"])["status"] == "completed"
    finally:
        runtime.shutdown()


def test_workflow_approval_denial_keeps_filesystem_untouched(tmp_path):
    store, runtime, runner = make(tmp_path)
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="Workflow", description="S16"))
    registry.register_implementation(ToolImplementation(
        name="fake_workflow", tool_name="Workflow", handler=runtime.run_tool,
    ))
    approvals = ApprovalStore()
    executor = ToolExecutor(
        registry, approval_store=approvals,
        permission_policy=PermissionPolicy(workspace_root=tmp_path),
    )
    try:
        decision = executor.execute("Workflow", {
            "name": "review-files", "args": {"paths": [P1]},
        })
        assert decision.status == "approval_required"
        assert executor.deny(decision.approval_id)
        assert not store.list()
        assert runner.calls == []
    finally:
        runtime.shutdown()


def test_readonly_worker_denies_private_state_even_without_global_hooks(tmp_path):
    worker = ReadOnlyAgentRunner(
        llm_factory=lambda: None, workspace_root=tmp_path,
        private_roots=(tmp_path / ".my-private-store",),
    )
    manager = worker._hooks()
    for path in (
        ".workflows/wf_aabbccddeeff.journal.jsonl",
        ".team/mailboxes/lead.jsonl", ".tasks/task_12345678.json",
        ".my-private-store/hidden.json", ".memory/MEMORY.md",
    ):
        report = manager.dispatch(HookContext(
            event="before_tool", tool_name="read_file", arguments={"path": path},
        ))
        assert not report.allowed
    assert manager.dispatch(HookContext(
        event="before_tool", tool_name="read_file", arguments={"path": P1},
    )).allowed


def test_s15_receives_only_current_process_workflow_completion(tmp_path):
    store, runtime, runner = make(tmp_path)
    class FakeLoop:
        def __init__(self):
            self.tool_registry = ToolRegistry()
            self.skill_loader = None
            self.planner = None
            self.context_compactor = None
            self.agent_memory_store = None
            self.task_store = None
            self.hook_manager = None

    harness = IntegratedHarness(agent=FakeLoop(), workflow_runtime=runtime)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        finish(runtime, run_id)
        collected = harness.poll()
        assert len(collected) == 1
        assert collected[0]["source"] == "s16_workflow"
        assert collected[0]["status"] == "completed"
        assert harness.poll() == []
        assert harness.capability_snapshot()["workflow_runtime"] is True
    finally:
        harness.shutdown()
        runtime.shutdown()


def test_restart_never_automatically_replays_old_workflow_runs(tmp_path):
    store, runtime, runner = make(tmp_path)
    run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
    finish(runtime, run_id)
    runtime.shutdown()
    another_store = WorkflowStore(tmp_path)
    extra_runner = FakeRunner()
    second = WorkflowRuntime(
        store=another_store, workspace_root=tmp_path,
        runner_factory=lambda: extra_runner,
    )
    try:
        assert second.current_runs() == []
        assert second.show(run_id)["status"] == "completed"
        assert extra_runner.calls == []
        assert not another_store.locked(run_id)
    finally:
        second.shutdown()


def test_locked_run_cannot_be_unlocked_without_operator_confirmation(tmp_path):
    store = WorkflowStore(tmp_path)
    record = store.reserve("review-files", {"paths": [P1]})
    run_id = record["id"]
    try:
        with pytest.raises(PermissionError):
            store.clear_orphan_lock(run_id, confirmed=False)
        with pytest.raises(PermissionError):
            store.clear_orphan_lock(run_id, confirmed=True, active=True)
        with pytest.raises(PermissionError):
            store.clear_orphan_lock(run_id, confirmed=True)
        assert store.locked(run_id)
    finally:
        store.unlock(run_id)


def test_cli_commands_normalized_and_excluded_from_memory_learning():
    for text in (
        "workflow-catalog", "workflow-list",
        'workflow-run review-files {"paths":["app/agent/schema.py"]}',
        "workflow-resume wf_0123456789ab",
        "workflow-show wf_0123456789ab",
        "workflow-events wf_0123456789ab",
        "workflow-wait wf_0123456789ab 30",
        "workflow-unlock wf_0123456789ab CONFIRM",
    ):
        assert normalize_cli_input(text) == "/" + text
        assert looks_like_operational_input(text)


def test_nested_schema_checks_primitives_without_exec():
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["status", "count"],
        "additionalProperties": False,
    }
    assert _validate_schema({"status": "ok", "count": 2}, schema)
    assert not _validate_schema({"status": "ok", "count": "bad"}, schema)
    assert not _validate_schema({"status": "ok", "count": 2, "secret": "x"}, schema)
    assert not _validate_schema("not-json", schema)


def test_real_s06_research_runner_reads_source_with_s04_hooks(tmp_path):
    from app.hooks import HookManager
    (tmp_path / "notes.py").write_text("answer = 42\n", encoding="utf-8")

    class StubLLM:
        def __init__(self):
            self.count = 0
        def json_chat(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return {
                    "action": "tool", "tool_name": "read_file",
                    "arguments": {"path": "notes.py"},
                }
            return {"action": "final", "task_complete": True}
        def chat(self, messages, **kwargs):
            return "notes.py contains answer = 42"

    hooks = HookManager()
    runner = ReadOnlyAgentRunner(
        workspace_root=tmp_path, llm_factory=StubLLM,
        hook_manager=hooks,
    )
    answer = runner.run("只读检查 notes.py", OUTPUT_SCHEMA, "audit:notes")
    assert answer["verified"] is True
    assert answer["paths"] == ["notes.py"]
    assert "answer" in answer["summary"]
    events = [(row.event, row.tool_name) for row in hooks.recent()]
    assert ("before_tool", "read_file") in events
    assert ("after_tool", "read_file") in events


def test_finished_output_is_authoritative_and_snapshot_cannot_forge_completion(tmp_path):
    store, runtime, _ = make(tmp_path)
    try:
        run_id = runtime.launch("review-files", {"paths": [P1]})["run_id"]
        finish(runtime, run_id)
        out = store.read_output(run_id)
        assert out["status"] == "reviewed"
        snapshot = store.load(run_id)
        snapshot["status"] = "completed"
        snapshot["progress"] = []  # progress is not the source of output facts
        store.save(snapshot)
        assert runtime.show(run_id)["result"] == out
    finally:
        runtime.shutdown()


def test_same_prompt_different_labels_are_independent_semantic_calls(tmp_path):
    store, runtime, runner = make(tmp_path)
    row = store.reserve("review-files", {"paths": [P1]})
    state = ExecutionState(
        run_id=row["id"], store=store, runner=runner,
        meta=runtime.registry.get("review-files")[0],
    )
    try:
        results = asyncio.run(state.parallel([
            lambda: state.agent("Read one file", label="alpha"),
            lambda: state.agent("Read one file", label="beta"),
        ]))
        assert len(results) == 2
        assert len(runner.calls) == 2
        assert len(store.journal(row["id"])) == 2
    finally:
        store.unlock(row["id"])
        runtime.shutdown()


def test_s04_before_tool_veto_blocks_approved_workflow_launch(tmp_path):
    from app.hooks import HookDecision, HookManager
    store, runtime, runner = make(tmp_path)
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(name="Workflow", description="S16"))
    registry.register_implementation(ToolImplementation(
        name="approved_workflow", tool_name="Workflow",
        handler=runtime.run_tool,
    ))
    hooks = HookManager()
    hooks.register(
        "before_tool",
        lambda ctx: HookDecision(allow=False, reason="operator safety veto")
        if ctx.tool_name == "Workflow" else HookDecision(),
        name="s16_guard",
    )
    executor = ToolExecutor(
        registry, permission_policy=PermissionPolicy(workspace_root=tmp_path),
        hook_manager=hooks, approval_store=ApprovalStore(),
    )
    try:
        pending = executor.execute("Workflow", {
            "name": "review-files", "args": {"paths": [P1]},
        })
        assert pending.status == "approval_required"
        denied = executor.approve(pending.approval_id)
        assert denied.status == "denied"
        assert denied.success is False
        assert store.list() == []
        assert runner.calls == []
    finally:
        runtime.shutdown()


def test_s11_research_cannot_read_s16_workflow_journal(tmp_path):
    from app.agent.background_tasks import BackgroundTaskService
    from app.agent.task_system import TaskStore
    service = BackgroundTaskService(
        store=TaskStore(workspace_root=tmp_path),
        workspace_root=str(tmp_path), llm_factory=lambda: None,
        workflow_root=tmp_path / ".private-workflow-records",
    )
    try:
        hooks = service._hooks()
        for path in (
            ".workflows/wf_123456789abc.journal.jsonl",
            ".private-workflow-records/wf_123456789abc.json",
        ):
            report = hooks.dispatch(HookContext(
                event="before_tool", tool_name="read_file",
                arguments={"path": path},
            ))
            assert not report.allowed
    finally:
        service.shutdown()


def test_s13_teammate_cannot_read_s16_private_workflow_state(tmp_path):
    from app.agent.team_runtime import TeamRuntime
    from app.agent.task_system import TaskStore
    team = TeamRuntime(
        store=TaskStore(workspace_root=tmp_path),
        llm_factory=lambda: None, workspace_root=tmp_path,
        workflow_root=tmp_path / ".custom-workflow-state",
    )
    try:
        hooks = team._teammate_hooks(tmp_path)
        for path in (
            ".workflows/wf_123456789abc.json",
            ".custom-workflow-state/wf_123456789abc.journal.jsonl",
        ):
            assert not hooks.dispatch(HookContext(
                event="before_tool", tool_name="read_file",
                arguments={"path": path},
            )).allowed
    finally:
        team.shutdown()


def test_s03_blocks_direct_read_of_workflow_journal(tmp_path):
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_workflow_directory=tmp_path / ".workflows",
    )
    assert policy.evaluate(
        "read_file", {"path": ".workflows/wf_aaaaaaaaaaaa.journal.jsonl"}
    ).action == "deny"
    assert policy.evaluate(
        "read_file", {"path": P1}
    ).action == "allow"
