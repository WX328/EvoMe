"""Path privacy, trace storage lifecycle and authoritative outcome regressions."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.skills.registry import SkillRegistry
from app.telemetry.tracing import TraceRecorder, current_trace
from app.tools.builtin import build_default_tool_registry
from app.tools.executor import ToolExecutor
from app.tools.filesystem import FileSystemTools
from app.tools.permissions import PermissionPolicy


def file_executor(root):
    return ToolExecutor(build_default_tool_registry(skill_registry=SkillRegistry(), workspace_root=root),
                        permission_policy=PermissionPolicy(workspace_root=root))


@pytest.mark.parametrize("pattern", ["../*", "a/../../*", "C:*.txt", "C:/work/*", "/etc/*",
    "\\\\server\\share\\*", ".env:stream", "folder. /*", "folder./*", ".GIT/**",
    ".traces/**", ".workflows/**", ".ssh/*", "**/*.pem"])
def test_glob_rejects_sensitive_and_ambiguous_patterns_without_hooks(tmp_path, pattern):
    result = file_executor(tmp_path).execute("glob", {"pattern": pattern})
    assert not result.success and result.status == "denied"


def test_recursive_glob_prunes_secrets_before_entering_directories(tmp_path, monkeypatch):
    hidden = [".GIT", ".venv", ".traces", ".ssh", ".aws", "evome_qdrant",
              ".memory", ".tasks", ".schedules", ".team", ".workflows", "nested/.env.local"]
    for name in hidden:
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "PRIVATE_NAME_CANARY.py").write_text("secret", encoding="utf-8")
    for name in [".env", ".ENV.production", "evome.db-wal", ".mcp_servers.json", "secret.key",
                 "id_ed25519", "credentials.json", "bundle.pfx", "nested/server.pem"]:
        (tmp_path / name).write_text("secret", encoding="utf-8")
    for name in [".env.example", "app.py", "nested/public.py"]:
        (tmp_path / name).write_text("public", encoding="utf-8")
    visited = []
    original = os.scandir
    def scan(path):
        visited.append(Path(path))
        return original(path)
    monkeypatch.setattr("app.tools.filesystem.os.scandir", scan)
    result = file_executor(tmp_path).execute("glob", {"pattern": "**/*"})
    assert result.success
    names = {row["path"].replace("\\", "/") for row in result.output["matches"]}
    assert names == {".env.example", "app.py", "nested", "nested/public.py"}
    assert not any(path in visited for path in [tmp_path / name for name in hidden])
    assert "PRIVATE_NAME_CANARY" not in json.dumps(result.output)
    # A wildcard spelling of a protected directory cannot bypass filtering.
    assert file_executor(tmp_path).execute("glob", {"pattern": ".[gG][iI][tT]/**/*"}).output["matches"] == []


def test_glob_preserves_patterns_limits_and_public_configuration(tmp_path):
    fs = FileSystemTools(workspace_root=tmp_path)
    (tmp_path / "src" / "deep").mkdir(parents=True)
    for name in ["src/a.py", "src/deep/b.py", "src/readme.md", ".env.example"]:
        (tmp_path / name).touch()
    paths = lambda result: {row["path"].replace("\\", "/") for row in result["matches"]}
    assert paths(fs.glob("src/**/*.py")) == {"src/a.py", "src/deep/b.py"}
    assert paths(fs.glob("src/*.py")) == {"src/a.py"}
    assert paths(fs.glob(".env.example")) == {".env.example"}
    assert paths(fs.glob("src/**/")) == {"src", "src/deep"}
    assert fs.glob("**/*", max_results=1)["count"] == 1


def test_glob_hides_configured_private_roots_including_custom_names(tmp_path):
    private = tmp_path / "custom-state"
    private.mkdir()
    (private / "PRIVATE_CANARY.json").touch()
    (tmp_path / "public.py").touch()
    registry = build_default_tool_registry(skill_registry=SkillRegistry(), workspace_root=tmp_path,
                                           listing_excludes=(private, tmp_path.parent / "outside"))
    ex = ToolExecutor(registry)
    assert ex.execute("glob", {"pattern": "**/*"}).output["matches"] == [{"path": "public.py", "type": "file"}]
    assert ex.execute("glob", {"pattern": "custom-state/**/*"}).output["matches"] == []


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length path anchors")
def test_windows_extended_path_alias_is_canonical_before_containment_check(tmp_path, monkeypatch):
    from app.tools.workspace import resolved_path
    path = tmp_path / "events.sqlite3-shm"
    resolve = Path.resolve
    def extended(self, *args, **kwargs):
        result = resolve(self, *args, **kwargs)
        return Path("\\\\?\\" + str(result)) if self == path else result
    monkeypatch.setattr(Path, "resolve", extended)
    assert resolved_path(path).is_relative_to(tmp_path)


def directory_alias(link, target):
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


def test_glob_skips_real_junction_aliases_escapes_and_cycles(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    private = root / ".traces"
    private.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (private / "PRIVATE_CANARY.txt").touch()
    (outside / "OUTSIDE_CANARY.txt").touch()
    (root / "public.txt").touch()
    links = [root / "alias", root / "escape", root / "loop"]
    try:
        for link, target in zip(links, [private, outside, root]):
            directory_alias(link, target)
        fs = FileSystemTools(workspace_root=root)
        assert fs.glob("**/*")["matches"] == [{"path": "public.txt", "type": "file"}]
        assert fs.glob("alias/**/*")["matches"] == []
        assert fs.glob("escape/**/*")["matches"] == []
        for name in ["alias/PRIVATE_CANARY.txt", "escape/OUTSIDE_CANARY.txt"]:
            assert file_executor(root).execute("read_file", {"path": name}).status == "denied"
        assert (outside / "OUTSIDE_CANARY.txt").is_file()
    finally:
        for link in links:
            if link.exists():
                link.rmdir() if os.name == "nt" else link.unlink()


def test_trace_initializes_writer_once_and_batches_retention(tmp_path, monkeypatch):
    statements, connections = [], []
    connect = sqlite3.connect
    def track(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        connections.append(args[0])
        return conn
    monkeypatch.setattr("app.telemetry.tracing.sqlite3.connect", track)
    recorder = TraceRecorder(tmp_path, max_events=9, cleanup_interval=4)
    try:
        for _ in range(10):
            with recorder.span("runtime", "fixture"):
                pass
        assert len(connections) == 1  # one writer, not one connection per event
        assert sum("CREATE TABLE" in sql for sql in statements) == 1
        assert sum("CREATE INDEX" in sql for sql in statements) == 1
        assert sum(sql.startswith("DELETE") for sql in statements) == 6  # init + 5 batches
        assert len(recorder.events(limit=20000)) == 9
        assert recorder.dropped_events == 0
    finally:
        recorder.close()
    with connect(recorder.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 9
    # Graceful close is idempotent and does not resurrect a writer.
    recorder.close()
    with recorder.span("runtime", "after_close"):
        pass
    assert recorder.dropped_events == 2


def test_trace_read_only_view_does_not_initialize_or_prune(tmp_path):
    recorder = TraceRecorder(tmp_path, read_only=True)
    assert recorder.events() == [] and not recorder.root.exists()
    with recorder.span("runtime", "forbidden_write"):
        pass
    assert not recorder.root.exists()
    writer = TraceRecorder(tmp_path, max_events=5, cleanup_interval=4)
    try:
        for _ in range(5):
            with writer.span("runtime", "fixture"):
                pass
        with sqlite3.connect(writer.path) as conn:
            before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert before == 7  # last 2 inserts await the next retention sweep
        assert recorder.traces()
        with sqlite3.connect(writer.path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == before
    finally:
        recorder.close()
        writer.close()


def test_failed_trace_initialization_closes_connection_and_recovers(tmp_path, monkeypatch):
    connections = []
    class BrokenConnection(sqlite3.Connection):
        closed = False
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("CREATE TABLE"):
                raise sqlite3.OperationalError("PRIVATE_DB_ERROR_CANARY")
            return super().execute(sql, *args, **kwargs)
        def close(self):
            self.closed = True
            super().close()
    connect = sqlite3.connect
    def broken(*args, **kwargs):
        conn = connect(*args, **kwargs, factory=BrokenConnection)
        connections.append(conn)
        return conn
    recorder = TraceRecorder(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr("app.telemetry.tracing.sqlite3.connect", broken)
        with recorder.span("runtime", "fixture"):
            pass
    assert recorder.dropped_events == 2
    assert connections and all(conn.closed for conn in connections)
    try:
        with recorder.span("runtime", "recovered"):
            pass
        assert recorder.traces()[0]["status"] == "completed"
    finally:
        recorder.close()


def test_concurrent_trace_writers_and_live_reader_keep_committed_events(tmp_path):
    writers = [TraceRecorder(tmp_path), TraceRecorder(tmp_path)]
    reader = TraceRecorder(tmp_path, read_only=True)
    def run(i):
        writer = writers[i % 2]
        with writer.span("runtime", "fixture", trace_id=f"{i:032x}"):
            writer.usage(purpose="fixture", model="fake", prompt_tokens=3, completion_tokens=2, cached_tokens=0)
        assert reader.events(limit=1)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(run, range(40)))
        assert len(reader.events(limit=20000)) == 120
        assert all(writer.dropped_events == 0 for writer in writers)
    finally:
        for recorder in [*writers, reader]:
            recorder.close()
    # All handles released: the actual file can be moved and reopened on Windows.
    original = writers[0].path
    moved = original.with_suffix(".moved")
    original.rename(moved)
    moved.rename(original)
    reopened = TraceRecorder(tmp_path, read_only=True)
    assert len(reopened.events(limit=20000)) == 120
    reopened.close()


def test_cached_trace_writer_revalidates_storage_junction(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    recorder = TraceRecorder(root)
    with recorder.span("runtime", "fixture"):
        pass
    outside = tmp_path / "outside"
    outside.mkdir()
    # A journal alias introduced AFTER initialization must be rejected even
    # while the cached WAL writer remains open.
    alias = Path(str(recorder.path) + "-journal")
    directory_alias(alias, outside)
    try:
        with recorder.span("runtime", "unsafe"):
            pass
        assert recorder.dropped_events == 2
        assert list(outside.iterdir()) == []
    finally:
        alias.rmdir() if os.name == "nt" else alias.unlink()
        recorder.close()


@pytest.mark.parametrize("goal_status,root_status,child_status", [
    ("achieved", "goal_achieved", "failed"),
    ("awaiting_approval", "approval_required", "completed"),
    ("deferred", "goal_deferred", "completed"),
    ("failed", "goal_failed", "completed"),
])
def test_goal_result_is_independent_of_late_child_completion(tmp_path, goal_status, root_status, child_status):
    recorder = TraceRecorder(tmp_path)
    try:
        with recorder.span("runtime", "goal_continue") as outcome:
            parent = current_trace()
            recorder.goal_state(goal_status)
            outcome["status"] = root_status
        with recorder.span("workflow", "run", parent=parent) as child:
            child["status"] = child_status
        row = recorder.traces()[0]
        assert row["goal_status"] == goal_status
        assert row["root_status"] == root_status
        assert row["execution_status"] == "finished"
        assert row["subtask_failures"] == int(child_status == "failed")
        assert row["status"] == "goal_" + goal_status
    finally:
        recorder.close()


def test_open_background_does_not_erase_goal_and_retention_is_partial(tmp_path):
    recorder = TraceRecorder(tmp_path, max_events=3, cleanup_interval=1)
    try:
        with recorder.span("runtime", "goal_continue") as result:
            parent = current_trace()
            recorder.goal_state("deferred")
            result["status"] = "goal_deferred"
        active = recorder.span("workflow", "run", parent=parent)
        active.__enter__()
        row = recorder.traces()[0]
        assert row["goal_status"] == "deferred"
        assert row["open_spans"] == 1
        assert row["partial"] is True
        assert row["execution_status"] == "unfinished"
        assert row["status"] == "partial"
        active.__exit__(None, None, None)
        assert recorder.traces()[0]["execution_status"] == "partial"
    finally:
        recorder.close()


def test_real_goal_loop_trace_requires_delivered_answer_to_report_achieved(tmp_path):
    from app.agent.loop import AgentLoop
    from app.agent.goal_loop import GoalController
    from tests.test_s17_1_goal_workflow_integration import (
        fixtures, ScriptedGoalWorker, EvidenceGoalEvaluator,
    )
    workflow, registry, runner = fixtures(tmp_path)
    goal = GoalController(EvidenceGoalEvaluator(), background_running=lambda: False)
    worker = ScriptedGoalWorker()
    agent = AgentLoop(llm=worker, tool_registry=registry,
                      permission_policy=PermissionPolicy(workspace_root=tmp_path), max_steps=5)
    agent.goal_controller = goal
    recorder = TraceRecorder(tmp_path)
    from app.runtime import EvoMeRuntime
    runtime = object.__new__(EvoMeRuntime)
    runtime.strategy = SimpleNamespace(agent_loop=agent)
    runtime.goal_controller = goal
    runtime.trace_recorder = recorder
    goal.set("inspect source evidence with Workflow")
    try:
        with recorder.span("runtime", "goal_continue", trace_id=goal.trace_id) as result:
            first = agent.run(user_input=goal.condition)
            result["status"] = first.termination_reason
        assert recorder.traces()[0]["goal_status"] == "awaiting_approval"
        approved = runtime.approve_tool_call(first.approval_id)
        worker.run_id = approved.output["run_id"]
        workflow.wait(worker.run_id, timeout=5)
        assert recorder.traces()[0]["goal_status"] == "ready"
        worker.decisions = ["inspect", "final"]
        goal.begin_query()
        with recorder.span("runtime", "goal_continue", trace_id=goal.trace_id) as result:
            final = agent.run(user_input=goal.condition)
            result["status"] = final.termination_reason
        row = recorder.traces()[0]
        assert goal.last_status == row["goal_status"] == "achieved"
        assert row["root_status"] == "goal_achieved"
        assert row["execution_status"] == "finished" and not row["partial"]
        assert len(runner.calls) == 2
    finally:
        workflow.shutdown()
        recorder.close()


def test_goal_clear_is_recorded_on_the_original_goal_trace(tmp_path):
    from app.runtime import EvoMeRuntime
    from app.agent.goal_loop import GoalController
    from tests.test_s17_goal_loop import FakeEvaluator
    runtime = object.__new__(EvoMeRuntime)
    goal = GoalController(FakeEvaluator([]))
    goal.set("inspect source")
    runtime.goal_controller = goal
    runtime.trace_recorder = recorder = TraceRecorder(tmp_path)
    trace_id = goal.trace_id
    try:
        with recorder.span("runtime", "goal_continue", trace_id=trace_id):
            recorder.goal_state("active")
        runtime.goal_clear()
        row = recorder.traces()[0]
        assert row["trace_id"] == trace_id
        assert row["goal_status"] == row["root_status"] == "cleared"
        assert row["execution_status"] == "finished"
    finally:
        recorder.close()


def test_provisional_evaluator_success_is_not_a_trace_goal_success(tmp_path):
    from app.agent.loop import AgentLoop
    from app.agent.goal_loop import GoalController, GoalEvaluation
    from app.tools.registry import ToolRegistry
    from tests.test_s17_goal_loop import FakeEvaluator
    class Worker:
        def json_chat(self, *args, **kwargs):
            return {"action": "final", "task_complete": True}
        def chat(self, *args, **kwargs):
            raise RuntimeError("PRIVATE_DELIVERY_ERROR")
    goal = GoalController(FakeEvaluator([GoalEvaluation(True, "evidence sufficient")]))
    goal.set("check source")
    agent = AgentLoop(llm=Worker(), tool_registry=ToolRegistry(), max_steps=3)
    agent.goal_controller = goal
    recorder = TraceRecorder(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with recorder.span("runtime", "goal_continue", trace_id=goal.trace_id):
                agent.run(user_input=goal.condition)
        row = recorder.traces()[0]
        assert goal.last_status == "achieved_pending_answer"
        assert row["goal_status"] != "achieved"
        assert row["root_status"] == row["status"] == "error"
        assert "PRIVATE_DELIVERY_ERROR" not in json.dumps(recorder.events())
    finally:
        recorder.close()
