"""S10 durable task DAG / approvals / persistence regression tests.

Owner runs pytest; no test in this file needs a real model/API or network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent import AgentLoop
from app.agent.task_system import TaskRecord, TaskStore
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookManager
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.permissions import PermissionPolicy


def store(tmp_path):
    return TaskStore(workspace_root=tmp_path)


def new(store_, name, blocked=None):
    return store_.create_task(name, blockedBy=blocked)["id"]


def test_empty_store_is_read_only_and_has_no_folder(tmp_path):
    tasks = store(tmp_path)
    assert tasks.list_tasks() == []
    assert tasks.ready_tasks() == []
    assert not (tmp_path / ".tasks").exists()


def test_durable_record_restored_after_restart(tmp_path):
    first = store(tmp_path)
    tid = new(first, "Design schema")
    path = tmp_path / ".tasks" / f"{tid}.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"] == tid
    assert data["blockedBy"] == []
    assert data["status"] == "pending"
    second = store(tmp_path)
    assert second.get_task(tid)["ready"] is True
    assert second.get_task(tid)["title"] == "Design schema"
    assert second.list_tasks()[0]["id"] == tid


def test_dependency_blocks_until_predecessor_completed(tmp_path):
    tasks = store(tmp_path)
    a = new(tasks, "Setup db")
    b = new(tasks, "Implement API", [a])
    assert [row["id"] for row in tasks.ready_tasks()] == [a]
    assert tasks.get_task(b)["blockedByOpen"] == [a]
    with pytest.raises(ValueError, match="blocked"):
        tasks.claim_task(b, "operator")
    first = tasks.claim_task(a, "operator")
    assert first["status"] == "in_progress" and not first["ready"]
    tasks.complete_task(a, "operator", "database created")
    assert tasks.get_task(b)["ready"]
    assert [row["id"] for row in tasks.ready_tasks()] == [b]
    second = store(tmp_path)
    assert second.get_task(b)["blockedByOpen"] == []


def test_diamond_dependencies_require_all_predecessors(tmp_path):
    tasks = store(tmp_path)
    source = new(tasks, "Parse")
    left = new(tasks, "Transform", [source])
    right = new(tasks, "Validate", [source])
    finish = new(tasks, "Emit", [left, right])
    tasks.claim_task(source, "dev")
    tasks.complete_task(source, "dev")
    assert set(row["id"] for row in tasks.ready_tasks()) == {left, right}
    tasks.claim_task(left, "dev")
    tasks.complete_task(left, "dev")
    assert tasks.get_task(finish)["blockedByOpen"] == [right]
    tasks.claim_task(right, "dev")
    tasks.complete_task(right, "dev")
    assert tasks.get_task(finish)["ready"]


def test_dependency_self_unknown_duplicate_and_cycle_fail_closed(tmp_path):
    tasks = store(tmp_path)
    a = new(tasks, "A")
    b = new(tasks, "B")
    tasks.add_dependency(b, a)
    for dep in (b, "task_00000000"):
        with pytest.raises((ValueError, KeyError)):
            tasks.add_dependency(b, dep)
    with pytest.raises(ValueError, match="cycle"):
        tasks.add_dependency(a, b)
    with pytest.raises(ValueError, match="already exists"):
        tasks.add_dependency(b, a)
    assert tasks.get_task(a)["blockedBy"] == []
    assert tasks.get_task(b)["blockedBy"] == [a]


def test_create_with_missing_parent_rejected_without_disk_side_effect(tmp_path):
    tasks = store(tmp_path)
    with pytest.raises(ValueError, match="already exist"):
        tasks.create_task("Missing parent", blockedBy=["task_01234567"])
    assert tasks.list_tasks() == []
    assert not (tmp_path / ".tasks").exists()


def test_owner_must_match_claim_for_completion(tmp_path):
    tasks = store(tmp_path)
    tid = new(tasks, "Implement tests")
    tasks.claim_task(tid, "alice")
    with pytest.raises(PermissionError, match="claiming owner"):
        tasks.complete_task(tid, "bob")
    assert tasks.get_task(tid)["status"] == "in_progress"
    tasks.complete_task(tid, "alice")
    assert tasks.get_task(tid)["status"] == "completed"
    assert tasks.get_task(tid)["owner"] == "alice"
    with pytest.raises(ValueError, match="in_progress"):
        tasks.complete_task(tid, "alice")


def test_claimed_task_cannot_be_claimed_again_or_edit_dependencies(tmp_path):
    tasks = store(tmp_path)
    a = new(tasks, "A")
    b = new(tasks, "B")
    tasks.claim_task(b, "worker")
    with pytest.raises(ValueError, match="already claimed"):
        tasks.claim_task(b, "other")
    with pytest.raises(ValueError, match="after task started"):
        tasks.add_dependency(b, a)


def test_release_explicit_owner_restores_pending_without_running_work(tmp_path):
    tasks = store(tmp_path)
    tid = new(tasks, "Manually retry")
    tasks.claim_task(tid, "worker")
    with pytest.raises(PermissionError):
        tasks.release_task(tid, "other")
    released = tasks.release_task(tid, "worker")
    assert released["status"] == "pending" and released["ready"]
    assert released["owner"] == ""
    tasks.claim_task(tid, "worker2")
    assert store(tmp_path).get_task(tid)["owner"] == "worker2"


def test_started_task_and_finished_task_survive_restarts(tmp_path):
    tasks = store(tmp_path)
    a = new(tasks, "A")
    b = new(tasks, "B", [a])
    tasks.claim_task(a, "worker")
    assert store(tmp_path).get_task(a)["status"] == "in_progress"
    store(tmp_path).complete_task(a, "worker", result="manual result")
    after = store(tmp_path)
    assert after.get_task(a)["result"] == "manual result"
    assert after.get_task(b)["ready"]


def test_invalid_ids_titles_owners_dependencies_rejected(tmp_path):
    tasks = store(tmp_path)
    for value in ("", " ", "a" * 201):
        with pytest.raises(ValueError):
            tasks.create_task(value)
    tid = new(tasks, "Valid")
    for value in ("../.env", "task_XX", "task_1234567", "task_12345678/abc"):
        with pytest.raises(ValueError):
            tasks.get_task(value)
    with pytest.raises(ValueError):
        tasks.create_task("List wrong type", blockedBy="no")
    with pytest.raises(ValueError):
        tasks.create_task("Duplicate", blockedBy=[tid, tid])
    with pytest.raises(ValueError):
        tasks.claim_task(tid, " ")


def test_invalid_json_and_dangling_links_not_silently_ignored(tmp_path):
    tasks = store(tmp_path)
    tid = new(tasks, "A")
    path = tmp_path / ".tasks" / f"{tid}.json"
    path.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        tasks.list_tasks()
    with pytest.raises(ValueError, match="corrupt"):
        tasks.create_task("B")
    path.write_text(json.dumps({
        **TaskRecord.from_dict({
            "id": tid, "title": "A", "description": "",
            "status": "pending", "blockedBy": [], "owner": "",
            "created_at": "2026-01-01", "updated_at": "2026-01-01",
            "result": "", "version": 1,
        }).to_dict(),
        "blockedBy": ["task_12345678"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="dangling"):
        tasks.list_tasks()


def test_symlink_root_and_symlink_record_are_rejected(tmp_path):
    root = tmp_path / ".tasks"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        root.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Windows symlink creation unavailable")
    with pytest.raises(PermissionError):
        store(tmp_path)


def test_atomic_replace_failure_preserves_existing_record(tmp_path, monkeypatch):
    tasks = store(tmp_path)
    tid = new(tasks, "do not corrupt")
    import app.agent.task_system as module

    existing = (tmp_path / ".tasks" / f"{tid}.json").read_bytes()

    def fail_replace(src, target):
        raise OSError("simulated I/O failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        tasks.claim_task(tid, "worker")
    assert (tmp_path / ".tasks" / f"{tid}.json").read_bytes() == existing
    assert not list((tmp_path / ".tasks").glob(".s10-*.tmp"))


def test_direct_file_tool_mutations_to_task_store_are_denied(tmp_path):
    tasks = store(tmp_path)
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_task_directory=tasks.root,
    )
    for name in ("write_file", "edit_file"):
        for path in (".tasks/hijack.json", "sub/../.tasks/hijack.json"):
            result = policy.evaluate(name, {"path": path, "content": "{}"})
            assert result.action == "deny"
    assert policy.evaluate(
        "write_file", {"path": "regular.py", "content": "a"}
    ).action == "ask"
    assert policy.evaluate("task_list", {}).action == "allow"
    assert policy.evaluate("task_create", {"title": "A"}).action == "ask"


class DummyLLM:
    def json_chat(self, messages, **kwargs):
        return {"action": "final", "task_complete": True}

    def chat(self, messages, **kwargs):
        return "no-op"


def make_agent(tmp_path, *, enabled=True, subagent=False, hooks=None):
    task_store = store(tmp_path) if enabled else None
    tools = build_default_tool_registry(
        skill_registry=SkillRegistry(), workspace_root=tmp_path,
        enable_bash=False,
    )
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_task_directory=task_store.root if task_store else None,
    )
    return AgentLoop(
        llm=DummyLLM(), tool_registry=tools, permission_policy=policy,
        hook_manager=hooks, task_store=task_store,
        planning_enabled=True, subagent_enabled=subagent,
    )


def test_task_tools_are_opt_in(tmp_path):
    agent = make_agent(tmp_path, enabled=False)
    names = {item.name for item in agent.tool_registry.list_tools()}
    assert "task_create" not in names
    assert "task_list" not in names
    assert agent.planner is not None


def test_read_only_task_tools_require_no_approval(tmp_path):
    agent = make_agent(tmp_path)
    result = agent.executor.execute("task_list")
    assert result.success and result.output == []
    assert agent.executor.execute("task_ready").success
    assert agent.executor.approval_store.list_pending() == []


def test_agent_mutations_require_approval_and_dispatch_hooks(tmp_path):
    hooks = HookManager()
    agent = make_agent(tmp_path, hooks=hooks)
    pending = agent.executor.execute("task_create", {"title": "API"})
    assert pending.status == "approval_required"
    assert agent.task_store.list_tasks() == []
    assert hooks.recent() == []
    approved = agent.executor.approve(pending.approval_id)
    assert approved.success
    task_id = approved.output["id"]
    assert agent.task_store.get_task(task_id)["status"] == "pending"
    events = [(entry.event, entry.tool_name) for entry in hooks.recent()]
    assert ("before_tool", "task_create") in events
    assert ("after_tool", "task_create") in events
    denied = agent.executor.execute("task_claim", {
        "task_id": task_id, "owner": "agent",
    })
    assert denied.status == "approval_required"
    assert agent.task_store.get_task(task_id)["status"] == "pending"
    agent.executor.deny(denied.approval_id)
    assert agent.task_store.get_task(task_id)["status"] == "pending"


def test_s06_subagent_cannot_create_or_claim_durable_tasks(tmp_path):
    agent = make_agent(tmp_path, subagent=True)
    assert {t.name for t in agent.subagent.registry.list_tools()} == {
        "read_file", "glob",
    }
    assert "task_create" in {t.name for t in agent.tool_registry.list_tools()}


def test_cli_command_vocabulary_does_not_become_user_memory(tmp_path):
    assert normalize_cli_input("task-list") == "/task-list"
    assert normalize_cli_input("task-create Setup environment") == "/task-create Setup environment"
    assert looks_like_operational_input("/task-depend task_01234567 task_12345678")
    assert looks_like_operational_input("/task-start task_01234567")


def test_completion_unblocks_after_multiple_restarts(tmp_path):
    original = store(tmp_path)
    a = new(original, "first")
    b = new(original, "second", [a])
    second = store(tmp_path)
    assert second.get_task(b)["ready"] is False
    second.claim_task(a, "operator")
    third = store(tmp_path)
    third.complete_task(a, "operator")
    assert store(tmp_path).get_task(b)["ready"] is True
