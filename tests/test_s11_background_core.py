"""S11 worker tests; owner runs pytest. No external API."""
from __future__ import annotations
from threading import Event
import pytest

from app.agent.background_tasks import BackgroundTaskService
from app.agent.task_system import TaskStore
from app.cli_commands import normalize_cli_input, looks_like_operational_input
from app.hooks import HookManager


class FakeLLM:
    def __init__(self, *, action="read_file", entered=None, release=None):
        self.calls = 0
        self.action = action
        self.entered = entered
        self.release = release
    def json_chat(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            if self.entered: self.entered.set()
            if self.release and not self.release.wait(5):
                raise RuntimeError("fake worker timeout")
            return {"action":"tool","tool_name":self.action,
                    "arguments":{"path":"notes.py"} if self.action=="read_file"
                    else {"path":"oops.py","content":"not allowed"}}
        return {"action":"final","task_complete":True}
    def chat(self, messages, **kwargs):
        return "read-only report, grounded in notes.py"


def setup(tmp_path, llm=None, hooks=None):
    (tmp_path/"notes.py").write_text("def answer(): return 42\n", encoding="utf-8")
    store=TaskStore(workspace_root=tmp_path)
    worker=BackgroundTaskService(store=store, workspace_root=str(tmp_path),
                                 llm_factory=lambda: llm or FakeLLM(),
                                 hook_manager=hooks)
    return store,worker


def test_success_persists_and_unblocks_dependencies(tmp_path):
    store,worker=setup(tmp_path)
    a=store.create_task("读取 notes.py")["id"]
    b=store.create_task("分析 notes.py",blockedBy=[a])["id"]
    try:
        with pytest.raises(ValueError): worker.submit(b)
        job=worker.submit(a)
        assert job["job_id"]
        result=worker.wait(a,timeout=5)
        assert result["job_status"]=="completed",result
        assert result["task_status"]=="completed"
        assert result["evidence"][0]["tool"]=="read_file"
        assert TaskStore(workspace_root=tmp_path).get_task(a)["status"]=="completed"
        assert store.get_task(b)["ready"]
    finally: worker.shutdown()


def test_foreground_is_not_blocked_and_duplicate_is_rejected(tmp_path):
    entered,release=Event(),Event()
    model=FakeLLM(entered=entered,release=release)
    store,worker=setup(tmp_path,model)
    tid=store.create_task("读取 notes.py")["id"]
    try:
        worker.submit(tid)
        assert entered.wait(4)
        assert store.create_task("其他任务")["ready"]
        with pytest.raises(ValueError): worker.submit(tid)
        with pytest.raises(ValueError): worker.recover(tid)
        release.set()
        assert worker.wait(tid,timeout=5)["task_status"]=="completed"
        assert model.calls==2
    finally:
        release.set()
        worker.shutdown()


def test_only_research_tasks_are_accepted(tmp_path):
    store,worker=setup(tmp_path)
    try:
        for title in ("初始化数据库","修改 notes.py","读取 notes.py 然后删除文件"):
            tid=store.create_task(title)["id"]
            with pytest.raises((ValueError,PermissionError)): worker.submit(tid)
            assert store.get_task(tid)["status"]=="pending"
    finally: worker.shutdown()


def test_failure_is_persistent_and_requires_manual_retry(tmp_path):
    models=iter([FakeLLM(action="write_file"),FakeLLM()])
    (tmp_path/"notes.py").write_text("x=1\n",encoding="utf-8")
    store=TaskStore(workspace_root=tmp_path)
    worker=BackgroundTaskService(store=store,workspace_root=str(tmp_path),
                                 llm_factory=lambda: next(models))
    tid=store.create_task("读取 notes.py")["id"]
    try:
        worker.submit(tid)
        bad=worker.wait(tid,timeout=5)
        assert bad["task_status"]=="failed"
        assert bad["job_status"]=="failed"
        assert not (tmp_path/"oops.py").exists()
        assert TaskStore(workspace_root=tmp_path).get_task(tid)["status"]=="failed"
        with pytest.raises(ValueError): worker.submit(tid)
        assert worker.recover(tid)["status"]=="pending"
        worker.submit(tid)
        assert worker.wait(tid,timeout=5)["task_status"]=="completed"
    finally: worker.shutdown()


def test_orphan_after_restart_never_auto_reexecutes(tmp_path):
    store,worker=setup(tmp_path)
    tid=store.create_task("读取 notes.py")["id"]
    store.claim_task(tid,"s11_orphan")
    worker.shutdown()
    second=BackgroundTaskService(store=TaskStore(workspace_root=tmp_path),
                                 workspace_root=str(tmp_path),
                                 llm_factory=FakeLLM)
    try:
        assert second.get_status(tid)["job_status"]=="interrupted_or_external"
        assert second.recover(tid)["status"]=="pending"
        second.submit(tid)
        assert second.wait(tid,timeout=5)["task_status"]=="completed"
    finally: second.shutdown()


def test_secret_guard_runs_even_without_global_hooks(tmp_path):
    (tmp_path/".env").write_text("NEVER_READ_THIS_SECRET=123",encoding="utf-8")
    class SecretLLM(FakeLLM):
        def json_chat(self,messages,**kwargs):
            self.calls+=1
            if self.calls==1:
                return {"action":"tool","tool_name":"read_file",
                        "arguments":{"path":".env"}}
            assert "NEVER_READ_THIS_SECRET" not in str(messages)
            return {"action":"final","task_complete":True}
    store,worker=setup(tmp_path,SecretLLM())
    tid=store.create_task("读取 notes.py")["id"]
    try:
        worker.submit(tid)
        assert worker.wait(tid,timeout=5)["task_status"]=="failed"
    finally: worker.shutdown()


def test_s04_forwarding_and_s10_failure_state(tmp_path):
    hooks=HookManager()
    store,worker=setup(tmp_path,hooks=hooks)
    tid=store.create_task("读取 notes.py")["id"]
    try:
        worker.submit(tid)
        assert worker.wait(tid,timeout=5)["task_status"]=="completed"
        assert ("before_tool","read_file") in [(x.event,x.tool_name) for x in hooks.recent()]
    finally: worker.shutdown()
    other=store.create_task("另一个任务")["id"]
    store.claim_task(other,"s11_worker")
    store.fail_task(other,"s11_worker",error="fake failure")
    assert store.get_task(other)["status"]=="failed"
    assert store.retry_failed_task(other)["status"]=="pending"


def test_cli_operations_are_not_learning_samples():
    for cmd in ("task-bg-list","task-bg-run task_aaaaaaaa",
                "task-bg-show task_aaaaaaaa","task-bg-wait task_aaaaaaaa",
                "task-bg-recover task_aaaaaaaa"):
        assert normalize_cli_input(cmd)=="/"+cmd
        assert looks_like_operational_input(cmd)
