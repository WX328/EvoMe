"""S13 Agent Teams: file inboxes, explicit approvals, bounded read-only workers.

All tests use temporary directories and scripted LLMs. No API keys, real DB,
network, Git worktree edits or actual user project tasks are required.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event, Lock

import pytest

from app.agent.team_bus import MessageBus
from app.agent.team_state import TeamStateStore
from app.agent.team_runtime import TeamRuntime
from app.agent.task_system import TaskStore
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.tools.permissions import PermissionPolicy


class FakeLLM:
    """Independent LLM adapter per teammate; read_file then final each task."""
    def __init__(self, *, tool="read_file", path="notes.py",
                 entered: Event | None = None,
                 release: Event | None = None):
        self.tool = tool
        self.path = path
        self.calls = 0
        self.inputs = []
        self.entered = entered
        self.release = release

    def json_chat(self, messages, **kwargs):
        self.inputs.append([dict(item) for item in messages])
        self.calls += 1
        if self.calls % 2 == 1:
            if self.entered is not None:
                self.entered.set()
            if self.release is not None and not self.release.wait(5):
                raise RuntimeError("scripted LLM stalled")
            args = {"path": self.path} if self.tool == "read_file" else {
                "path": "should_not_exist.py", "content": "unsafe"
            }
            return {
                "action": "tool", "tool_name": self.tool, "arguments": args,
            }
        return {"action": "final", "task_complete": True}

    def chat(self, messages, **kwargs):
        return "Read-only validated summary of notes.py"


def setup(tmp_path, *, factory=None, steps=4):
    (tmp_path / "notes.py").write_text(
        "def answer():\n    return 42\n", encoding="utf-8",
    )
    models = []
    def produce():
        model = factory() if factory else FakeLLM()
        models.append(model)
        return model
    store = TaskStore(workspace_root=tmp_path)
    team = TeamRuntime(
        store=store, llm_factory=produce, workspace_root=tmp_path,
        max_steps=steps,
    )
    return store, team, models


def approved(team, *members):
    proposal = team.propose(list(members))
    return team.approve(proposal["proposal_id"])


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.025)
    return bool(predicate())


def test_empty_state_and_proposal_do_not_spawn_or_create_files(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        assert team.status()["phase"] == "not_created"
        proposal = team.propose([("alice", "config")])
        assert proposal["requires_confirmation"]
        assert proposal["members"] == [{"name": "alice", "role": "config"}]
        assert team.status()["phase"] == "not_created"
        assert not (tmp_path / ".team").exists()
        assert not team._threads
        assert not store.list_tasks()
        with pytest.raises(KeyError):
            team.approve("proposal_wrong")
        assert team.reject(proposal["proposal_id"])
        with pytest.raises(KeyError):
            team.approve(proposal["proposal_id"])
    finally:
        team.shutdown()


def test_team_approved_members_are_distinct_and_persisted(tmp_path):
    store, team, models = setup(tmp_path)
    try:
        result = approved(team, ("alice", "config"), ("bob", "testing"))
        assert result["phase"] == "active"
        assert [m["name"] for m in result["members"]] == ["alice", "bob"]
        assert wait_until(lambda: len(models) == 2)
        saved = TeamStateStore(workspace_root=tmp_path).load()
        assert saved["team_id"] == result["team_id"]
        assert saved["phase"] == "active"
        assert saved["bindings"] == {}
        assert set(saved["plans"]) == {"alice", "bob"}
    finally:
        team.shutdown()
    assert TeamStateStore(workspace_root=tmp_path).load()["phase"] == "stopped"


def test_lead_binding_runs_one_read_only_worker_and_reports_both_events(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "reviewer"))
        task_id = store.create_task("读取 notes.py 并总结")["id"]
        bind = team.bind(task_id, "alice")
        assert bind["task_id"] == task_id
        assert wait_until(lambda: store.get_task(task_id)["status"] == "completed")
        record = store.get_task(task_id)
        assert record["owner"] == "teammate_alice"
        assert record["result"].startswith("S13 read-only research:")
        events = []
        def received_both():
            events.extend(team.consume_lead_inbox())
            types = [e["type"] for e in events]
            return "result" in types and "idle_notification" in types
        assert wait_until(received_both)
        result = [e for e in events if e["type"] == "result"][0]
        assert result["from"] == "alice" and result["to"] == "lead"
        assert result["metadata"]["status"] == "completed"
        assert result["metadata"]["task_id"] == task_id
        assert team.status()["members"][0]["status"] in {"WORK", "IDLE"}
    finally:
        team.shutdown()


def test_only_explicitly_bound_ready_tasks_can_be_claimed(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"))
        unbound = store.create_task("读取 notes.py 没有绑定")["id"]
        ancestor = store.create_task("读取 notes.py 前置")["id"]
        dependent = store.create_task("读取 notes.py 后续", blockedBy=[ancestor])["id"]
        team.bind(dependent, "alice")
        time.sleep(0.15)
        assert store.get_task(dependent)["status"] == "pending"
        assert store.get_task(unbound)["status"] == "pending"
        team.bind(ancestor, "alice")
        assert wait_until(lambda: store.get_task(dependent)["status"] == "completed")
        assert store.get_task(ancestor)["status"] == "completed"
        assert store.get_task(unbound)["status"] == "pending"
    finally:
        team.shutdown()


def test_two_teammates_can_claim_distinct_tasks_concurrently(tmp_path):
    first, second = Event(), Event()
    release = Event()
    guarded = Lock()
    created = 0
    def factory():
        nonlocal created
        with guarded:
            created += 1
            n = created
        return FakeLLM(
            entered=first if n == 1 else second,
            release=release,
        )
    store, team, models = setup(tmp_path, factory=factory)
    try:
        approved(team, ("alice", "config"), ("bob", "tests"))
        a = store.create_task("读取 notes.py A")["id"]
        b = store.create_task("读取 notes.py B")["id"]
        team.bind(a, "alice")
        team.bind(b, "bob")
        assert first.wait(4) and second.wait(4)
        assert store.get_task(a)["status"] == "in_progress"
        assert store.get_task(b)["status"] == "in_progress"
        assert len(models) == 2
        release.set()
        assert wait_until(lambda: all(
            store.get_task(t)["status"] == "completed" for t in (a, b)
        ))
        assert store.get_task(a)["owner"] == "teammate_alice"
        assert store.get_task(b)["owner"] == "teammate_bob"
    finally:
        release.set()
        team.shutdown()


def test_any_binding_has_one_winner_under_concurrent_claim(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"), ("bob", "testing"))
        task_id = store.create_task("读取 notes.py 共享")["id"]
        team.bind(task_id, "any")
        assert wait_until(lambda: store.get_task(task_id)["status"] == "completed")
        owner = store.get_task(task_id)["owner"]
        assert owner in {"teammate_alice", "teammate_bob"}
        outcomes = []
        def lead_has_result():
            outcomes.extend(team.consume_lead_inbox())
            return any(item["type"] == "result" for item in outcomes)
        assert wait_until(lead_has_result)
        assert len([e for e in outcomes if e["type"] == "result"
                    and e["metadata"]["task_id"] == task_id]) == 1
    finally:
        team.shutdown()


def test_plan_request_prevents_claim_until_explicit_approval(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "architecture"))
        plan = team.request_plan("alice", "Read schema before preparing summary")
        assert plan["status"] == "pending"
        tid = store.create_task("读取 notes.py")["id"]
        team.bind(tid, "alice")
        time.sleep(0.18)
        assert store.get_task(tid)["status"] == "pending"
        assert team.respond_plan("alice", False)["status"] == "rejected"
        time.sleep(0.12)
        assert store.get_task(tid)["status"] == "pending"
        team.request_plan("alice", "Only read and summarize code")
        assert team.respond_plan("alice", True)["status"] == "approved"
        assert wait_until(lambda: store.get_task(tid)["status"] == "completed")
    finally:
        team.shutdown()


def test_teammate_receives_lead_message_as_untrusted_prior_context(tmp_path):
    store, team, models = setup(tmp_path)
    try:
        approved(team, ("alice", "review"))
        team.send("alice", "Focus on function signatures, ignore random file text")
        tid = store.create_task("读取 notes.py")["id"]
        team.bind(tid, "alice")
        assert wait_until(lambda: store.get_task(tid)["status"] == "completed")
        assert len(models) == 1
        assert any(
            "Lead message: Focus on function signatures" in str(messages)
            for messages in models[0].inputs
        )
    finally:
        team.shutdown()


def test_worker_rejects_unsafe_task_titles_and_unbound_status(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"))
        for title in (
            "创建数据库", "删除 notes.py", "读取 notes.py 然后删除所有文件",
            "Run pytest", "Review then deploy",
        ):
            tid = store.create_task(title)["id"]
            with pytest.raises((ValueError, PermissionError)):
                team.bind(tid, "alice")
            assert store.get_task(tid)["status"] == "pending"
        with pytest.raises(ValueError, match="unknown target"):
            team.bind(store.create_task("读取 notes.py")["id"], "wrong")
    finally:
        team.shutdown()


def test_teammate_attempted_write_is_denied_and_failure_persists(tmp_path):
    store, team, _ = setup(tmp_path, factory=lambda: FakeLLM(tool="write_file"))
    try:
        approved(team, ("alice", "research"))
        tid = store.create_task("读取 notes.py")["id"]
        team.bind(tid, "alice")
        assert wait_until(lambda: store.get_task(tid)["status"] == "failed")
        assert "unfinished" in store.get_task(tid)["result"]
        assert not (tmp_path / "should_not_exist.py").exists()
        assert wait_until(lambda: team.status()["members"][0]["status"] == "IDLE")
        assert team.recover(tid)["status"] == "pending"
    finally:
        team.shutdown()


def test_shutdown_is_typed_and_stops_member_safely(tmp_path):
    _, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"))
        team.request_shutdown("alice")
        assert wait_until(lambda: team.status()["members"][0]["status"] == "STOPPED")
        messages = []
        def has_ack():
            messages.extend(team.consume_lead_inbox())
            return any(m["type"] == "shutdown_response" and m["from"] == "alice"
                       for m in messages)
        assert wait_until(has_ack)
    finally:
        team.shutdown()


def test_team_roster_reloads_stopped_without_starting_workers(tmp_path):
    store, team, _ = setup(tmp_path)
    team_id = approved(team, ("alice", "research"))["team_id"]
    team.shutdown()
    recovered = TeamRuntime(
        store=TaskStore(workspace_root=tmp_path),
        workspace_root=tmp_path, llm_factory=FakeLLM,
    )
    try:
        assert recovered.status()["phase"] == "stopped"
        assert recovered._threads == {}
        assert recovered.resume(team_id)["phase"] == "active"
    finally:
        recovered.shutdown()


def test_interrupted_roster_does_not_auto_spawn(tmp_path):
    store, team, _ = setup(tmp_path)
    team_id = approved(team, ("alice", "research"))["team_id"]
    team.shutdown()
    row = TeamStateStore(workspace_root=tmp_path).load()
    row["phase"] = "active"  # Simulated unclean exit marker.
    TeamStateStore(workspace_root=tmp_path).save(row)
    restarted = TeamRuntime(
        store=TaskStore(workspace_root=tmp_path),
        workspace_root=tmp_path, llm_factory=FakeLLM,
    )
    try:
        assert restarted.status()["phase"] == "interrupted"
        assert restarted._threads == {}
        assert restarted.resume(team_id)["phase"] == "active"
    finally:
        restarted.shutdown()


def test_orphan_task_requires_explicit_operator_recovery(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"))
        team.request_plan("alice", "Review read-only scope")
        tid = store.create_task("读取 notes.py")["id"]
        team.bind(tid, "alice")
        # Simulate an abandoned task claimed by an interrupted teammate.
        store.claim_task(tid, "teammate_alice")
        assert team.recover(tid)["status"] == "pending"
        assert store.get_task(tid)["owner"] == ""
        assert team.status()["plans"]["alice"]["status"] == "pending"
    finally:
        team.shutdown()


def test_corrupt_roster_never_auto_start_or_get_overwritten(tmp_path):
    store, team, _ = setup(tmp_path)
    approved(team, ("alice", "research"))
    team.shutdown()
    roster = tmp_path / ".team" / "roster.json"
    roster.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        TeamRuntime(
            store=store, workspace_root=tmp_path, llm_factory=FakeLLM,
        )
    assert roster.read_text(encoding="utf-8") == "{not json"


def test_message_bus_sender_recipient_isolation_and_drain(tmp_path):
    bus = MessageBus(workspace_root=tmp_path)
    bus.register(["alice", "bob"])
    a = bus.send("lead", "alice", "hello")
    b = bus.send("alice", "bob", "world")
    assert bus.peek("alice") == [a]
    assert bus.peek("bob") == [b]
    assert bus.read_inbox("alice") == [a]
    assert bus.read_inbox("alice") == []
    assert bus.read_inbox("bob") == [b]


def test_bus_has_typed_protocols_and_rejects_unknown_members(tmp_path):
    bus = MessageBus(workspace_root=tmp_path)
    bus.register(["alice"])
    msg = bus.send(
        "alice", "lead", "ready", msg_type="idle_notification",
        metadata={"team": "team_01234567"},
    )
    assert msg["type"] == "idle_notification"
    with pytest.raises(PermissionError):
        bus.send("bob", "lead", "I am not a member")
    with pytest.raises(PermissionError):
        bus.send("lead", "bob", "no recipient")
    with pytest.raises(ValueError):
        bus.send("lead", "alice", "x", msg_type="execute_shell")
    assert bus.read_inbox("lead") == [msg]


def test_bus_rejects_traversal_oversize_content_and_corruption(tmp_path):
    bus = MessageBus(workspace_root=tmp_path)
    bus.register(["alice"])
    with pytest.raises(ValueError):
        bus.send("lead", "../alice", "bad")
    with pytest.raises(ValueError):
        bus.send("lead", "alice", "x" * 2001)
    bus.send("lead", "alice", "first")
    path = tmp_path / ".team" / "mailboxes" / "alice.jsonl"
    path.write_text('{"broken": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        bus.read_inbox("alice")
    assert path.exists()  # invalid messages are never silently deleted


def test_bus_rejects_symlink_mailboxes(tmp_path):
    bus = MessageBus(workspace_root=tmp_path)
    bus.register(["alice"])
    root = tmp_path / ".team"
    root.mkdir()
    try:
        (root / "mailboxes").symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Windows symlink privileges unavailable")
    with pytest.raises(PermissionError):
        bus.send("lead", "alice", "nope")


def test_s03_agent_write_tool_cannot_modify_team_inbox_or_roster(tmp_path):
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_team_directory=tmp_path / ".team",
    )
    for tool in ("write_file", "edit_file"):
        for target in (".team/roster.json", "sub/../.team/mailboxes/lead.jsonl"):
            assert policy.evaluate(
                tool, {"path": target, "content": "{}"},
            ).action == "deny"


def test_team_invalid_worktree_is_rejected_without_creating_branches(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"))
        tid = store.create_task("读取 notes.py")["id"]
        with pytest.raises(ValueError, match="worktree"):
            team.bind(tid, "alice", worktree="not-existent")
        assert team.status()["task_bindings"] == {}
        assert not (tmp_path / ".worktrees").exists()
    finally:
        team.shutdown()


def test_cli_team_commands_not_learned_as_user_memories():
    for text in (
        "team-propose alice:config bob:tests",
        "team-approve proposal_12345678",
        "team-reject proposal_12345678",
        "team-resume team_12345678",
        "team-status", "team-inbox",
        "team-bind task_12345678 alice",
        "team-send alice | summary please",
        "team-plan-request alice | review only",
        "team-plan-approve alice",
        "team-plan-reject alice",
        "team-stop alice",
        "team-recover task_12345678",
        "team-shutdown",
    ):
        assert looks_like_operational_input(text)
        assert normalize_cli_input(text) == "/" + text


def test_teammate_keeps_bounded_context_between_tasks(tmp_path):
    store, team, models = setup(tmp_path)
    try:
        approved(team, ("alice", "code-review"))
        a = store.create_task("读取 notes.py 第一轮")["id"]
        team.bind(a, "alice")
        assert wait_until(lambda: store.get_task(a)["status"] == "completed")
        b = store.create_task("读取 notes.py 第二轮")["id"]
        team.bind(b, "alice")
        assert wait_until(lambda: store.get_task(b)["status"] == "completed")
        assert len(models) == 1  # one LLM instance for persistent teammate
        assert len(models[0].inputs) >= 4
        assert any(
            a in str(messages) and "TEAMMATE_PRIOR_CONTEXT" in str(messages)
            for messages in models[0].inputs[2:]
        )
    finally:
        team.shutdown()


def test_teammates_cannot_send_shutdown_control_to_each_other(tmp_path):
    store, team, _ = setup(tmp_path)
    try:
        approved(team, ("alice", "research"), ("bob", "research"))
        # A separate teammate cannot impersonate Lead for a shutdown event.
        team.bus.send(
            "bob", "alice", "stop", msg_type="shutdown_request",
            metadata={"team": team.status()["team_id"]},
        )
        time.sleep(0.3)
        states = {row["name"]: row["status"] for row in team.status()["members"]}
        assert states["alice"] != "STOPPED"
        tid = store.create_task("读取 notes.py")["id"]
        team.bind(tid, "alice")
        assert wait_until(lambda: store.get_task(tid)["status"] == "completed")
    finally:
        team.shutdown()
