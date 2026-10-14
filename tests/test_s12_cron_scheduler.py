"""S12 Cron Scheduler regression tests: no network, LLM or real time sleeps.

Tests use injected clocks and a fake safe background worker. The project
owner runs pytest after applying the patch.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.cron_store import ScheduleStore, next_cron
from app.agent.cron_scheduler import CronScheduler
from app.agent.background_tasks import BackgroundTaskService
from app.agent.task_system import TaskStore
from app.cli_commands import normalize_cli_input, looks_like_operational_input
from app.tools.permissions import PermissionPolicy
from app.hooks import HookContext


T0 = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)
RESEARCH = "读取 app/agent/schema.py 并总结"


class Clock:
    def __init__(self, start=T0):
        self.current = start
    def __call__(self):
        return self.current
    def set(self, value):
        self.current = value
    def advance(self, seconds):
        self.current += timedelta(seconds=seconds)


class FakeWorker:
    def __init__(self, root, *, fail=False):
        self.store = TaskStore(workspace_root=root)
        self.fail = fail
        self.submitted = []
    def _task_prompt(self, payload):
        return BackgroundTaskService._task_prompt(payload)
    def submit(self, task_id):
        if self.fail:
            raise RuntimeError("simulated S11 dispatch failure")
        task = self.store.get_task(task_id)
        self.store.claim_task(task_id, "s11_fake")
        self.store.complete_task(task_id, "s11_fake", result="read-only test evidence")
        self.submitted.append(task_id)
        return {"job_id": "test_job"}
    def get_status(self, task_id):
        task = self.store.get_task(task_id)
        return {"task_id": task_id, "task_status": task["status"]}


def setup(tmp_path, *, clock=None, worker=None):
    clock = clock or Clock()
    store = ScheduleStore(workspace_root=tmp_path)
    worker = worker or FakeWorker(tmp_path)
    service = CronScheduler(store=store, worker=worker, clock=clock)
    return store, service, clock, worker


def test_reading_empty_schedule_store_does_not_create_directory(tmp_path):
    store, scheduler, _, _ = setup(tmp_path)
    assert store.list() == []
    assert scheduler.list() == []
    assert not (tmp_path / ".schedules").exists()


def test_once_at_dispatches_only_one_new_s10_task(tmp_path):
    store, scheduler, clock, worker = setup(tmp_path)
    created = scheduler.create_at(
        at="2026-10-14T08:00:05+08:00", title=RESEARCH,
    )
    id_ = created["id"]
    assert created["next_run_at"] == "2026-10-14T00:00:05+00:00"
    assert created["status"] == "active"
    assert scheduler.tick() == []
    clock.advance(5)
    fired = scheduler.tick()
    assert len(fired) == 1 and fired[0]["type"] == "dispatched"
    assert fired[0]["schedule_id"] == id_
    assert len(worker.submitted) == 1
    task = worker.store.get_task(worker.submitted[0])
    assert task["status"] == "completed"
    assert task["title"] == RESEARCH
    assert store.get(id_)["status"] == "completed"
    assert store.get(id_)["run_count"] == 1
    assert scheduler.tick() == []
    assert len(worker.submitted) == 1


def test_weekday_cron_and_fixed_utc_offset():
    # 2026-10-14 is Wednesday (cron weekday=3).
    now = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)
    assert next_cron("30 9 * * 1-5", now, "+08:00") == datetime(
        2026, 10, 14, 1, 30, tzinfo=timezone.utc,
    )
    assert next_cron("30 9 * * 1-5", datetime(
        2026, 10, 16, 2, 0, tzinfo=timezone.utc,
    ), "+08:00") == datetime(2026, 10, 19, 1, 30, tzinfo=timezone.utc)


def test_interval_cron_produces_fresh_task_and_honors_run_limit(tmp_path):
    store, scheduler, clock, worker = setup(tmp_path)
    created = scheduler.create_cron(
        expression="*/5 * * * *", utc_offset="+00:00",
        title=RESEARCH, max_runs=3,
    )
    for index in range(1, 4):
        clock.advance(5 * 60)
        fired = scheduler.tick()
        assert len(fired) == 1
        assert store.get(created["id"])["run_count"] == index
        assert len(worker.store.list_tasks()) == index
    assert store.get(created["id"])["status"] == "completed"
    assert len(set(worker.submitted)) == 3
    clock.advance(600)
    assert scheduler.tick() == []
    assert store.get(created["id"])["run_count"] == 3


def test_session_delay_is_not_persisted_and_fires_once(tmp_path):
    store, scheduler, clock, worker = setup(tmp_path)
    sess = scheduler.create_session(delay_seconds=5, title=RESEARCH)
    assert sess["id"].startswith("sess_")
    assert store.list() == []
    clock.advance(5)
    assert len(scheduler.tick()) == 1
    assert scheduler.tick() == []
    assert len(worker.submitted) == 1
    assert not scheduler._session
    recreated = CronScheduler(
        store=ScheduleStore(workspace_root=tmp_path),
        worker=FakeWorker(tmp_path), clock=clock,
    )
    assert recreated.list() == []


def test_pause_resume_and_cancel_block_trigger(tmp_path):
    store, scheduler, clock, worker = setup(tmp_path)
    c = scheduler.create_cron(
        expression="*/5 * * * *", utc_offset="+00:00",
        title=RESEARCH, max_runs=2,
    )
    assert scheduler.pause(c["id"])["status"] == "paused"
    clock.advance(300)
    assert scheduler.tick() == []
    resumed = scheduler.resume(c["id"])
    assert resumed["next_run_at"] == "2026-10-14T00:10:00+00:00"
    clock.advance(300)
    assert len(scheduler.tick()) == 1
    assert scheduler.cancel(c["id"])["status"] == "canceled"
    clock.advance(300)
    assert scheduler.tick() == []
    assert len(worker.submitted) == 1


def test_persistent_schedule_can_be_reloaded_after_restart(tmp_path):
    store, scheduler, clock, worker = setup(tmp_path)
    saved = scheduler.create_cron(
        expression="0 1 * * *", utc_offset="+00:00",
        title=RESEARCH, max_runs=5,
    )
    assert store.get(saved["id"])["status"] == "active"
    second = ScheduleStore(workspace_root=tmp_path)
    assert second.get(saved["id"])["next_run_at"] == saved["next_run_at"]
    assert len(second.list()) == 1


def test_restarted_scheduler_skips_overdue_at_and_cron(tmp_path):
    store, scheduler, clock, worker = setup(tmp_path)
    one = scheduler.create_at(at="2026-10-14T00:00:05+00:00", title=RESEARCH)
    recurring = scheduler.create_cron(
        expression="*/5 * * * *", utc_offset="+00:00",
        title=RESEARCH, max_runs=4,
    )
    clock.advance(15 * 60)
    recreated = CronScheduler(
        store=ScheduleStore(workspace_root=tmp_path),
        worker=FakeWorker(tmp_path), clock=clock,
        poll_seconds=3600,
    )
    try:
        assert recreated.start() == 2
        assert store.get(one["id"])["status"] == "missed"
        assert store.get(recurring["id"])["run_count"] == 0
        assert store.get(recurring["id"])["next_run_at"] > clock().isoformat()
        assert recreated.tick() == []
    finally:
        recreated.shutdown()
    assert worker.submitted == []


def test_dispatch_failure_consumes_occurrence_without_retry(tmp_path):
    clock = Clock()
    worker = FakeWorker(tmp_path, fail=True)
    store, scheduler, _, _ = setup(tmp_path, clock=clock, worker=worker)
    one = scheduler.create_at(
        at="2026-10-14T00:00:05+00:00", title=RESEARCH,
    )
    clock.advance(5)
    events = scheduler.tick()
    assert len(events) == 1
    assert events[0]["type"] == "dispatch_failed"
    assert "simulated" in events[0]["error"]
    assert scheduler.tick() == []
    assert store.get(one["id"])["status"] == "completed"
    assert "simulated" in store.get(one["id"])["last_error"]
    assert scheduler.events()[-1]["type"] == "dispatch_failed"


def test_scheduler_rejects_write_shell_or_nonresearch_tasks(tmp_path):
    store, scheduler, _, _ = setup(tmp_path)
    for title in ("修改代码", "创建数据库", "运行 pytest",
                  "读取 notes.py 后删除所有文件", "Review and delete files"):
        with pytest.raises((ValueError, PermissionError)):
            scheduler.create_cron(
                expression="* * * * *", utc_offset="+08:00",
                title=title,
            )
    assert store.list() == []


def test_invalid_cron_fields_and_unclear_timezone_are_rejected(tmp_path):
    _, scheduler, _, _ = setup(tmp_path)
    bad = ("61 * * * *", "0 25 * * *", "0 9 1 * *",
           "* * * * 7", "0 9 * * foo", "*/0 * * * *",
           "5/2 * * * *", "0 0 * *", "0 9 * * * *")
    for expr in bad:
        with pytest.raises(ValueError):
            scheduler.create_cron(
                expression=expr, utc_offset="+08:00", title=RESEARCH,
            )
    for tz in ("Asia/Shanghai", "+15:00", "+08:99", "UTC", ""):
        with pytest.raises(ValueError):
            scheduler.create_cron(
                expression="* * * * *", utc_offset=tz, title=RESEARCH,
            )


def test_at_requires_aware_future_date_and_run_limits(tmp_path):
    store, scheduler, _, _ = setup(tmp_path)
    for stamp in ("2026-10-14T00:00:05", "yesterday",
                  "2026-10-13T20:00:00+00:00"):
        with pytest.raises(ValueError):
            scheduler.create_at(at=stamp, title=RESEARCH)
    for runs in (0, 31, -1, True):
        with pytest.raises(ValueError):
            scheduler.create_cron(
                expression="* * * * *", utc_offset="+00:00",
                max_runs=runs, title=RESEARCH,
            )
    assert store.list() == []


def test_corrupt_schedule_json_is_a_hard_error(tmp_path):
    store, scheduler, _, _ = setup(tmp_path)
    record = scheduler.create_cron(
        expression="*/5 * * * *", utc_offset="+00:00", title=RESEARCH,
    )
    file = tmp_path / ".schedules" / f"{record['id']}.json"
    file.write_text("{broken json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        store.list()
    with pytest.raises(ValueError, match="corrupt"):
        scheduler.create_at(
            at="2026-10-14T00:00:05+00:00", title=RESEARCH,
        )


def test_schedule_root_symlinks_are_rejected(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    folder = tmp_path / ".schedules"
    try:
        folder.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem cannot create directory symlinks")
    with pytest.raises(PermissionError):
        ScheduleStore(workspace_root=tmp_path)


def test_s03_file_tools_cannot_modify_schedule_state(tmp_path):
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_schedule_directory=tmp_path / ".schedules",
    )
    for name in ("write_file", "edit_file"):
        for target in (".schedules/cron_bad.json",
                       "work/../.schedules/cron_bad.json"):
            result = policy.evaluate(name, {"path": target, "content": "{}"})
            assert result.action == "deny"
    assert policy.evaluate("write_file", {"path": "notes.py"}).action == "ask"


def test_s11_cannot_read_private_schedule_records(tmp_path):
    from app.agent.background_tasks import BackgroundTaskService
    hooks = None
    worker = BackgroundTaskService(
        store=TaskStore(workspace_root=tmp_path),
        workspace_root=str(tmp_path),
        llm_factory=lambda: None,
        schedule_root=tmp_path / ".schedules",
        hook_manager=hooks,
    )
    try:
        manager = worker._hooks()
        report = manager.dispatch(HookContext(
            event="before_tool", tool_name="read_file",
            arguments={"path": ".schedules/cron_12345678.json"},
        ))
        assert not report.allowed
    finally:
        worker.shutdown()


def test_cli_cron_commands_are_operational(tmp_path):
    for text in (
        "cron-list", "cron-events", "cron-tick",
        "cron-at 2026-10-15T09:00:00+08:00 | 读取 notes.py",
        'cron-add "*/5 * * * *" +08:00 3 | 读取 notes.py',
        "cron-after 15 | 读取 notes.py",
        "cron-show cron_01234567", "cron-cancel cron_01234567",
        "cron-pause cron_01234567", "cron-resume cron_01234567",
    ):
        assert normalize_cli_input(text) == "/" + text
        assert looks_like_operational_input(text)


def test_end_to_end_s12_triggers_real_s11_research_worker(tmp_path):
    """Full S10 -> S12 -> S11 -> S06 path with a scripted LLM (no API)."""
    (tmp_path / "notes.py").write_text("def answer(): return 42\n", encoding="utf-8")

    class FakeLLM:
        def __init__(self):
            self.steps = 0

        def json_chat(self, messages, **kwargs):
            self.steps += 1
            if self.steps == 1:
                return {
                    "action": "tool", "tool_name": "read_file",
                    "arguments": {"path": "notes.py"},
                }
            return {"action": "final", "task_complete": True}

        def chat(self, messages, **kwargs):
            return "已根据 notes.py 的源码验证函数定义。"

    worker = BackgroundTaskService(
        store=TaskStore(workspace_root=tmp_path),
        workspace_root=str(tmp_path),
        llm_factory=FakeLLM,
        schedule_root=tmp_path / ".schedules",
    )
    clock = Clock()
    store = ScheduleStore(workspace_root=tmp_path)
    scheduler = CronScheduler(store=store, worker=worker, clock=clock)
    try:
        scheduled = scheduler.create_at(
            at="2026-10-14T00:00:05+00:00",
            title="读取 notes.py 并总结函数",
        )
        clock.advance(5)
        fired = scheduler.tick()
        assert len(fired) == 1, fired
        task_id = fired[0]["task_id"]
        state = worker.wait(task_id, timeout=5)
        assert state["task_status"] == "completed", state
        assert state["evidence"][0]["path"] == "notes.py"
        scheduler.tick()
        assert any(
            event.get("type") == "task_finished"
            and event.get("task_id") == task_id
            for event in scheduler.events()
        )
        assert TaskStore(workspace_root=tmp_path).get_task(task_id)["status"] == "completed"
        assert store.get(scheduled["id"])["run_count"] == 1
    finally:
        scheduler.shutdown()
        worker.shutdown()
