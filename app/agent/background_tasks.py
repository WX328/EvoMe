"""S11 opt-in asynchronous, read-only execution of S10 durable tasks.

Only an operator can enqueue a pending, unblocked research task. The worker
owns a separate LLM client and an S06 ResearchSubagent with read_file/glob
only. No automatic shell/write approvals, no silent retries, no DB session
sharing, and no process-restart replay of orphaned jobs.
"""
from __future__ import annotations

import re
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from app.agent.subagent import ResearchSubagent
from app.agent.task_system import TaskStore
from app.background import BackgroundEvolutionQueue
from app.hooks import HookDecision, HookManager, build_sensitive_path_guard
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry


_RESEARCH_PREFIX = re.compile(
    r"^(?:读取|阅读|查看|分析|研究|审查|检索|查找|盘点|总结|检查|"
    r"read\b|inspect\b|review\b|research\b|analy[sz]e\b|search\b)",
    flags=re.IGNORECASE,
)
_UNSAFE = re.compile(
    r"(?:写入|改写|修改|删除|创建|新建|安装|部署|提交代码|运行代码|执行脚本|"
    r"生成文件|上传|覆盖|重命名|编辑文件|"
    r"\b(?:write|modify|delete|remove|install|deploy|execute|run|commit|"
    r"create|upload|overwrite|rename)\b)",
    flags=re.IGNORECASE,
)


class BackgroundTaskService:
    """Single-process job queue whose durable source of truth is TaskStore."""

    def __init__(
        self,
        *,
        store: TaskStore,
        llm_factory: Callable[[], Any],
        workspace_root: str,
        hook_manager: HookManager | None = None,
        memory_root: str | Path | None = None,
        schedule_root: str | Path | None = None,
        team_root: str | Path | None = None,
        max_steps: int = 6,
        max_output_chars: int = 10000,
    ) -> None:
        self.store = store
        self.llm_factory = llm_factory
        self.workspace_root = workspace_root
        self.memory_root = Path(memory_root).resolve() if memory_root is not None else None
        self.schedule_root = Path(schedule_root).resolve() if schedule_root is not None else None
        self.team_root = Path(team_root).resolve() if team_root is not None else None
        self.external_hooks = hook_manager
        self.max_steps = max(2, min(10, int(max_steps)))
        self.max_output_chars = max(2000, min(16000, int(max_output_chars)))
        self.queue = BackgroundEvolutionQueue(max_workers=1)
        self._lock = RLock()
        self._task_jobs: dict[str, str] = {}

    def _hooks(self) -> HookManager:
        """Always protect sensitive paths, even if S04 is disabled globally."""
        manager = HookManager(history_limit=100)
        manager.register(
            "before_tool",
            build_sensitive_path_guard(self.workspace_root),
            name="s11_protect_sensitive_paths",
            priority=1000,
        )

        def protect_working_state(context):
            if context.tool_name != "read_file":
                return HookDecision()
            raw = context.arguments.get("path")
            if not isinstance(raw, str):
                return HookDecision(False, "invalid read target")
            workspace = Path(self.workspace_root).resolve()
            try:
                resolved = (workspace / raw).resolve()
                protected = [
                    self.store.root.resolve(), (workspace / ".memory").resolve(),
                ]
                if self.memory_root is not None:
                    protected.append(self.memory_root)
                if self.schedule_root is not None:
                    protected.append(self.schedule_root)
                if self.team_root is not None:
                    protected.append(self.team_root)
                if any(resolved.is_relative_to(root) for root in protected):
                    return HookDecision(False, "S11 cannot read task state or private memory")
            except (ValueError, OSError, RuntimeError):
                return HookDecision(False, "unresolvable read target")
            return HookDecision()

        manager.register(
            "before_tool", protect_working_state,
            name="s11_protect_private_state", priority=900,
        )
        if self.external_hooks is not None:
            def delegate(context):
                report = self.external_hooks.dispatch(context)
                return HookDecision(allow=report.allowed, reason=report.reason)

            for event in ("before_agent", "after_agent", "before_tool", "after_tool"):
                manager.register(event, delegate, name=f"s11_s04_{event}")
        return manager

    @staticmethod
    def _task_prompt(record: dict[str, Any]) -> str:
        title = record["title"].strip()
        desc = record["description"].strip()
        if not _RESEARCH_PREFIX.search(title):
            raise ValueError(
                "S11 only runs explicit read-only research tasks. "
                "Create a task titled with 读取/分析/研究/审查 (read/inspect/review)."
            )
        prompt = f"{title}\n{desc}".strip() if desc else title
        if len(prompt) > 1200:
            raise ValueError("research task exceeds S06 1200-character budget")
        # Negated guardrails (e.g. "不修改文件") describe read-only intent
        # rather than an instruction to mutate. The child still cannot write.
        intent = re.sub(
            r"(?:不|不要|禁止|不得|勿)(?:修改|写入|删除|执行|运行)",
            "", prompt,
        )
        if _UNSAFE.search(intent):
            raise PermissionError(
                "S11 research worker does not write, modify, delete, deploy or run code"
            )
        return prompt

    def _research(self, *, task_id: str, owner: str, prompt: str) -> dict[str, Any]:
        try:
            client = self.llm_factory()
            # SkillRegistry is empty on purpose: no SQLite-bound memory skills
            # or foreground Session may be captured across worker threads.
            registry = build_default_tool_registry(
                skill_registry=SkillRegistry(),
                workspace_root=self.workspace_root,
                enable_bash=False,
                max_output_chars=self.max_output_chars,
            )
            research = ResearchSubagent(
                llm=client,
                source_registry=registry,
                hook_manager=self._hooks(),
                max_steps=self.max_steps,
                max_summary_chars=1600,
            )
            output = research.run(prompt)
            evidence = output.get("evidence", [])
            verified = [
                row for row in evidence
                if isinstance(row, dict)
                and row.get("tool") in {"read_file", "glob"}
            ]
            if not verified:
                raise RuntimeError("S11 no verified read-only evidence")
            if re.match(
                r"^(?:读取|阅读|分析|研究|审查|总结|检查|"
                r"read\b|review\b|inspect\b|analy[sz]e\b)",
                prompt, flags=re.IGNORECASE,
            ) and not any(row["tool"] == "read_file" for row in verified):
                raise RuntimeError("S11 research requires real read_file evidence")
            summary = str(output.get("summary", "")).strip()
            if not summary:
                raise RuntimeError("S11 subagent produced an empty research report")
            # S10 result is deliberately bounded and persistent. The full
            # structured evidence is available in the in-process job result
            # until shutdown; no secret source contents enter TaskStore.
            persisted = (
                f"S11 read-only research: {summary[:1000]} "
                f"[verified_tool_calls={output.get('tool_call_count', 0)}]"
            )[:1200]
            task = self.store.complete_task(task_id, owner, result=persisted)
            return {
                "task_id": task_id, "task_status": task["status"],
                "summary": summary, "evidence": evidence,
                "tool_call_count": output.get("tool_call_count", 0),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:1000]
            try:
                self.store.fail_task(task_id, owner, error=error)
            except Exception as store_error:
                # If persistence fails, the task remains claimed. It MUST NOT
                # be automatically retried after restart.
                raise RuntimeError(
                    f"S11 research failed; durable failure write failed: "
                    f"{type(store_error).__name__}: {store_error}"
                ) from exc
            raise

    def submit(self, task_id: str) -> dict[str, str]:
        """Explicit, non-blocking operator submission; claim before scheduling."""
        with self._lock:
            existing = self._task_jobs.get(task_id)
            if existing:
                job = self.queue.get(existing)
                if job is not None and job.status in {"queued", "running"}:
                    raise ValueError("S11 task is already queued or running")
            task = self.store.get_task(task_id)
            if not task["ready"]:
                raise ValueError("S11 task must be pending and free of blockers")
            prompt = self._task_prompt(task)
            owner = f"s11_{uuid4().hex[:12]}"
            self.store.claim_task(task_id, owner)
            try:
                job_id = self.queue.submit(
                    kind="s11_read_only_research",
                    fn=lambda: self._research(
                        task_id=task_id, owner=owner, prompt=prompt,
                    ),
                )
            except Exception:
                self.store.release_task(task_id, owner)
                raise
            self._task_jobs[task_id] = job_id
            return {"task_id": task_id, "job_id": job_id, "status": "queued"}

    def get_status(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        with self._lock:
            job_id = self._task_jobs.get(task_id)
            job = self.queue.get(job_id) if job_id else None
        output = {
            "task_id": task_id, "title": task["title"],
            "task_status": task["status"], "owner": task["owner"],
            "result": task["result"], "ready": task["ready"],
            "job_id": job_id,
            "job_status": job.status if job else (
                "interrupted_or_external" if task["status"] == "in_progress"
                and str(task["owner"]).startswith("s11_") else "not_active"
            ),
        }
        if job is not None:
            output["error"] = job.error
            if job.status == "completed":
                output["summary"] = job.result.get("summary", "")
                output["evidence"] = job.result.get("evidence", [])
        return output

    def list_status(self) -> list[dict[str, Any]]:
        return [
            self.get_status(row["id"])
            for row in self.store.list_tasks()
            if str(row.get("owner", "")).startswith("s11_")
            or row["id"] in self._task_jobs
        ]

    def wait(self, task_id: str, *, timeout: float | None = 30) -> dict[str, Any]:
        with self._lock:
            job_id = self._task_jobs.get(task_id)
        if not job_id:
            raise KeyError("S11 no job for this task in the current process")
        try:
            self.queue.wait(job_id, timeout=timeout)
        except Exception:
            # Failure information is still available from job + TaskStore.
            pass
        return self.get_status(task_id)

    def recover(self, task_id: str) -> dict[str, Any]:
        """Manual-only retry for failed or orphaned in_progress S11 tasks."""
        with self._lock:
            job_id = self._task_jobs.get(task_id)
            job = self.queue.get(job_id) if job_id else None
            if job is not None and job.status in {"queued", "running"}:
                raise ValueError("cannot recover an active background task")
            task = self.store.get_task(task_id)
            if not str(task["owner"]).startswith("s11_"):
                raise PermissionError("only S11-owned tasks can be recovered here")
            if task["status"] == "failed":
                return self.store.retry_failed_task(task_id)
            if task["status"] == "in_progress":
                return self.store.release_task(task_id, task["owner"])
            raise ValueError("task is not failed or interrupted")

    def shutdown(self, *, wait: bool = True) -> None:
        self.queue.shutdown(wait=wait)
