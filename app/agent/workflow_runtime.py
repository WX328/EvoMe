"""S16 trusted script orchestration above the existing S01/S03/S04 loop.

Workflow tool dispatch schedules a single safe, predefined, read-only script.
Scripts are Python functions registered by the HOST (never submitted by LLM).
Each child call uses an isolated S06 ResearchSubagent client/registry. Parallel
and pipeline stages share a bounded resource budget and durable semantic journal.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Awaitable, Callable, Protocol

from app.agent.subagent import ResearchSubagent
from app.agent.review_quality import (
    SUMMARY_CHARS, VERIFY_PREFIX, build_verification_request,
    unpack_verification_request, summarize_review_quality,
)
from app.agent.workflow_store import WorkflowStore, json_bytes, utc_now
from app.hooks import HookDecision, HookManager, build_sensitive_path_guard
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
MAX_INPUT_CHARS = 3500
MAX_AGENTS = 12
MAX_CONCURRENT = 2


class WorkflowInputError(ValueError):
    """Untrusted arguments or malformed script configuration."""


@dataclass(frozen=True)
class WorkflowMeta:
    name: str
    description: str
    phases: tuple[str, ...]


class ScriptRegistry:
    def __init__(self):
        self._scripts: dict[str, tuple[WorkflowMeta, Callable[..., Awaitable[dict]]]] = {}

    def register(self, *, name: str, description: str,
                 phases: tuple[str, ...], fn: Callable[..., Awaitable[dict]]) -> None:
        if (not isinstance(name, str) or not NAME.fullmatch(name)
                or not isinstance(description, str) or not 1 <= len(description) <= 300
                or not isinstance(phases, tuple) or not phases
                or len(phases) > 8 or any(not isinstance(p, str) or not 1 <= len(p) <= 80 for p in phases)
                or not callable(fn)):
            raise WorkflowInputError("invalid S16 host-owned script metadata")
        if name in self._scripts:
            raise WorkflowInputError("duplicate workflow registry name")
        self._scripts[name] = (WorkflowMeta(name, description, phases), fn)

    def get(self, name: str):
        if not isinstance(name, str) or name not in self._scripts:
            raise WorkflowInputError("unknown S16 workflow name")
        return self._scripts[name]

    def list(self) -> list[dict[str, Any]]:
        return [
            {"name": meta.name, "description": meta.description,
             "phases": list(meta.phases)}
            for meta, _fn in self._scripts.values()
        ]


class AgentRunner(Protocol):
    def run(self, prompt: str, schema: dict[str, Any] | None, label: str) -> dict[str, Any]: ...


def _validate_schema(value: Any, schema: dict[str, Any] | None) -> bool:
    """Small strict subset; no untrusted references or dynamic schema execution."""
    if schema is None:
        return isinstance(value, dict)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise WorkflowInputError("S16 only supports a trusted object schema")
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if (not isinstance(props, dict) or not isinstance(required, list)
            or not all(isinstance(x, str) for x in required)):
        raise WorkflowInputError("S16 invalid trusted schema")
    if not isinstance(value, dict) or any(key not in value for key in required):
        return False
    for key, item in value.items():
        if key not in props:
            if schema.get("additionalProperties", True) is False:
                return False
            continue
        spec = props[key]
        if not isinstance(spec, dict):
            raise WorkflowInputError("S16 malformed trusted field schema")
        expected = spec.get("type")
        checks = {
            "string": lambda v: isinstance(v, str),
            "boolean": lambda v: isinstance(v, bool),
            "integer": lambda v: type(v) is int,
            "number": lambda v: type(v) in (int, float),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }
        if expected not in checks or not checks[expected](item):
            return False
        if expected == "string" and len(item) > 2200:
            return False
        if expected == "array":
            if len(item) > 20:
                return False
            sub = spec.get("items", {}).get("type")
            if sub == "string" and not all(isinstance(v, str) and len(v) <= 250 for v in item):
                return False
    return True


class ReadOnlyAgentRunner:
    """Each sub-call is independent, no inherited memory/MCP/Bash capability."""

    def __init__(self, *, llm_factory: Callable[[], Any],
                 workspace_root: str | Path, hook_manager: HookManager | None = None,
                 private_roots: tuple[Path, ...] = ()):
        self.llm_factory = llm_factory
        self.workspace = Path(workspace_root).resolve()
        self.hook_manager = hook_manager
        self.private_roots = tuple(p.resolve() for p in private_roots)

    def _hooks(self) -> HookManager:
        manager = HookManager()
        manager.register(
            "before_tool", build_sensitive_path_guard(self.workspace),
            name="s16_secret_guard", priority=1000,
        )

        def deny_private(context):
            if context.tool_name != "read_file":
                return HookDecision()
            raw = context.arguments.get("path")
            if not isinstance(raw, str):
                return HookDecision(False, "bad S16 read path")
            target = (self.workspace / raw).resolve()
            blocks = (
                self.workspace / ".env",
                self.workspace / ".workflows",
                self.workspace / ".tasks",
                self.workspace / ".team",
                self.workspace / ".memory",
                self.workspace / ".schedules",
                *self.private_roots,
            )
            if any(target.is_relative_to(p.resolve()) for p in blocks):
                return HookDecision(False, "S16 private execution state is inaccessible")
            return HookDecision()

        manager.register("before_tool", deny_private,
                         name="s16_private_guard", priority=900)
        if self.hook_manager is not None:
            def forward(context):
                verdict = self.hook_manager.dispatch(context)
                return HookDecision(allow=verdict.allowed, reason=verdict.reason)
            for event in ("before_agent", "after_agent", "before_tool", "after_tool"):
                manager.register(event, forward, name=f"s16_s04_{event}")
        return manager

    def run(self, prompt: str, schema: dict[str, Any] | None, label: str) -> dict[str, Any]:
        if not isinstance(prompt, str):
            raise WorkflowInputError("S16 research prompt must be text")
        task, prior_context = unpack_verification_request(prompt)
        registry = build_default_tool_registry(
            skill_registry=SkillRegistry(),
            workspace_root=self.workspace, enable_bash=False,
            max_output_chars=10000,
        )
        child = ResearchSubagent(
            llm=self.llm_factory(),
            source_registry=registry,
            hook_manager=self._hooks(),
            max_steps=5,
            # S06 truncates before returning. Retain a 1-char sentinel so
            # summaries longer than the S16 journal cap are detectable.
            max_summary_chars=SUMMARY_CHARS + 1,
        )
        # Audit history is delivered in the S06 user-role context, never
        # as new system/tool policy. The trusted task still fits S06's limit.
        report = child.run(task, teammate_context=prior_context)
        evidence = [
            row for row in report.get("evidence", [])
            if isinstance(row, dict) and row.get("tool") == "read_file"
        ]
        if not evidence:
            raise RuntimeError("S16 agent did not verify a read_file tool call")
        # Preserve the 1-char truncation sentinel before stripping any
        # terminal whitespace: strip() could otherwise erase the signal.
        observed_summary = str(report.get("summary", ""))
        if not observed_summary.strip():
            raise RuntimeError("S16 agent returned no grounded summary")
        summary = observed_summary[:SUMMARY_CHARS].strip()
        # Tool success is NOT a factual verification of the narrative.
        # Track both context/source truncation instead of silently slicing.
        read_truncated = any(
            row.get("truncated") is True or row.get("line_truncated") is True
            for row in evidence
        )
        result = {
            "summary": summary,
            "paths": [str(row.get("path", ""))[:220] for row in evidence[:8]],
            "verified": True,
            "summary_truncated": len(observed_summary) > SUMMARY_CHARS,
            # Lower bound when truncated; not an invented original length.
            "summary_observed_chars": len(observed_summary),
        }
        if all("truncated" in row for row in evidence):
            result["source_read_truncated"] = read_truncated
        return result


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "paths": {"type": "array", "items": {"type": "string"}},
        "verified": {"type": "boolean"},
        "summary_truncated": {"type": "boolean"},
        "summary_observed_chars": {"type": "integer"},
        "source_read_truncated": {"type": "boolean"},
    },
    "required": ["summary", "paths", "verified"],
    "additionalProperties": False,
}


class ExecutionState:
    """Trusted agent/parallel/pipeline/phase/log primitives with stable calls."""

    def __init__(self, *, run_id: str, store: WorkflowStore, runner: AgentRunner,
                 meta: WorkflowMeta, max_calls: int = 8, max_parallel: int = 2):
        self.run_id = run_id
        self.store = store
        self.runner = runner
        self.meta = meta
        self.max_calls = max(1, min(MAX_AGENTS, int(max_calls)))
        self.max_parallel = max(1, min(MAX_CONCURRENT, int(max_parallel)))
        self._sem = asyncio.Semaphore(self.max_parallel)
        self._lock = RLock()
        self._cache = store.journal(run_id)
        self._physical = 0
        self._total_keys: set[str] = set()
        self._active_keys: set[str] = set()
        self._nested = 0

    def _update(self, *, kind: str, message: str = "",
                phase: str | None = None) -> None:
        with self._lock:
            row = self.store.load(self.run_id)
            if phase is not None:
                row["phase"] = phase[:80]
            row["progress"] = [
                *row["progress"][-98:],
                {"kind": kind[:50], "message": message[:220], "at": utc_now()},
            ]
            row["updated_at"] = utc_now()
            row["version"] += 1
            row["agent_calls"] = self._physical
            self.store.save(row)

    def phase(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.meta.phases:
            raise WorkflowInputError("workflow phase is not declared in trusted metadata")
        self._update(kind="phase", phase=name, message=name)

    def log(self, message: str) -> None:
        self._update(kind="log", message=str(message)[:220])

    @staticmethod
    def _key(prompt: str, schema: dict[str, Any] | None, label: str) -> str:
        key_schema = schema
        if schema == OUTPUT_SCHEMA:
            # S16.1 adds OPTIONAL observation metadata, not different Audit
            # semantics. Preserve S16 Audit journal keys across upgrades,
            # so successfully completed Audit calls can still be reused.
            # Verify keys change because their actual prompt/data changed.
            optional_meta = {
                "summary_truncated", "summary_observed_chars",
                "source_read_truncated",
            }
            key_schema = {
                **schema,
                "properties": {
                    name: spec for name, spec in schema["properties"].items()
                    if name not in optional_meta
                },
            }
        basis = json_bytes({
            "prompt": prompt, "label": label, "schema": key_schema,
        }, limit=12000)
        return "agent-" + hashlib.sha256(basis).hexdigest()[:32]

    async def agent(self, prompt: str, *, label: str, schema: dict[str, Any] | None = OUTPUT_SCHEMA) -> dict[str, Any]:
        # A host-built S16.1 verification envelope may carry the entire
        # saved Audit summary. Its embedded S06 task remains <=1200 chars.
        if (not isinstance(prompt, str) or not 1 <= len(prompt) <= 3500
                or not isinstance(label, str) or not 1 <= len(label) <= 90):
            raise WorkflowInputError("S16 agent requires bounded prompt and label")
        key = self._key(prompt, schema, label)
        with self._lock:
            self._total_keys.add(key)
            if len(self._total_keys) > self.max_calls:
                raise WorkflowInputError("S16 unique agent call budget exceeded")
            cached = self._cache.get(key)
            if cached is not None:
                if not _validate_schema(cached, schema):
                    raise WorkflowInputError("S16 cached result violates schema")
                self._update(kind="workflow_agent", message=f"{label}: cached")
                return cached
            if key in self._active_keys:
                raise WorkflowInputError("duplicate active agent semantic key")
            self._active_keys.add(key)
        try:
            async with self._sem:
                with self._lock:
                    # Parallel tasks may enter the semaphore after another call
                    # has journaled its semantic key.
                    cached = self._cache.get(key)
                    if cached is not None:
                        self._update(kind="workflow_agent", message=f"{label}: cached")
                        return cached
                    if self._physical >= self.max_calls:
                        raise WorkflowInputError("S16 agent execution budget exceeded")
                    self._physical += 1
                    self._update(kind="workflow_agent", message=f"{label}: running")
                value = await asyncio.to_thread(self.runner.run, prompt, schema, label)
                if not _validate_schema(value, schema):
                    with self._lock:
                        if self._physical >= self.max_calls:
                            raise WorkflowInputError("S16 retry budget exceeded")
                        self._physical += 1
                        self._update(kind="workflow_agent", message=f"{label}: retry_invalid_schema")
                    # A Verify v2 prompt is a signed-by-host-style JSON
                    # envelope, not arbitrary plain text. Appending retry
                    # instructions would corrupt its JSON payload.
                    retry_prompt = (
                        prompt if prompt.startswith(VERIFY_PREFIX)
                        else prompt + "\nReturn a valid JSON object."
                    )
                    value = await asyncio.to_thread(
                        self.runner.run, retry_prompt, schema, label,
                    )
                    if not _validate_schema(value, schema):
                        raise WorkflowInputError("S16 child returned malformed structured result twice")
                json_bytes(value, 12_000)
                with self._lock:
                    self.store.append(self.run_id, key, value)
                    self._cache[key] = value
                    self._update(kind="workflow_agent", message=f"{label}: completed")
                return value
        finally:
            with self._lock:
                self._active_keys.discard(key)

    async def parallel(self, tasks: list[Callable[[], Awaitable[Any]]]) -> list[Any]:
        if not isinstance(tasks, list) or len(tasks) > 8 or not all(callable(f) for f in tasks):
            raise WorkflowInputError("parallel requires <=8 trusted callables")
        # Wait for *all* children even if one failed. Otherwise a losing
        # asyncio.to_thread() call could still be writing the journal after
        # _execute() releases its run lock.
        results = list(await asyncio.gather(
            *(fn() for fn in tasks), return_exceptions=True,
        ))
        errors = [item for item in results if isinstance(item, BaseException)]
        if errors:
            raise errors[0]
        return results

    async def pipeline(self, items: list[Any], *stages: Callable[..., Awaitable[Any]]) -> list[Any]:
        if not isinstance(items, list) or not 1 <= len(items) <= 4 or not 1 <= len(stages) <= 4:
            raise WorkflowInputError("pipeline requires 1..4 items and 1..4 stages")
        if not all(callable(stage) for stage in stages):
            raise WorkflowInputError("pipeline stages must be host-owned functions")
        async def one(item: Any, index: int):
            value = item
            for stage in stages:
                value = await stage(value, item, index)
            return value
        return await self.parallel([lambda x=x, i=i: one(x, i) for i, x in enumerate(items)])

    async def workflow(self, name: str, args: dict[str, Any],
                       registry: ScriptRegistry) -> dict[str, Any]:
        if self._nested:
            raise WorkflowInputError("S16 nested workflows limited to one level")
        self._nested += 1
        try:
            meta, script = registry.get(name)
            if meta.name == self.meta.name:
                raise WorkflowInputError("recursive workflow is forbidden")
            return await script(self, args)
        finally:
            self._nested -= 1


async def review_files(ctx: ExecutionState, args: dict[str, Any]) -> dict[str, Any]:
    """Trusted script: audit each file then verify findings independently."""
    paths = args["paths"]
    ctx.phase("Audit")

    async def audit(_value: Any, path: str, index: int):
        output = await ctx.agent(
            f"只读检查工作区文件 {path} 的代码结构、边界、潜在问题，"
            "必须实际调用 read_file，给出有证据的摘要。",
            label=f"audit:{index}:{path}", schema=OUTPUT_SCHEMA,
        )
        return {"path": path, "audit": output}

    async def verify(previous: dict, path: str, index: int):
        # Keep the semantic key stable for resumption; a changed protocol
        # intentionally invalidates only the prior Verify journal entry.
        ctx.phase("Verify")
        audit_result = previous["audit"]
        checked = await ctx.agent(
            build_verification_request(path, audit_result),
            label=f"verify:{index}:{path}", schema=OUTPUT_SCHEMA,
        )
        return {
            "path": path,
            "audit": audit_result["summary"],
            "verification": checked["summary"],
            "evidence": checked["paths"],
            "quality": summarize_review_quality(path, audit_result, checked),
        }

    results = await ctx.pipeline(paths, audit, verify)
    ctx.phase("Report")
    ctx.log(f"Verified {len(results)} file(s); every stage used read_file")
    return {"files": results, "count": len(results), "status": "reviewed"}


class WorkflowRuntime:
    """One active run per process; each workflow requires ToolExecutor approval."""

    def __init__(self, *, store: WorkflowStore, llm_factory: Callable[[], Any] | None = None,
                 runner_factory: Callable[[], AgentRunner] | None = None,
                 workspace_root: str | Path, hook_manager: HookManager | None = None,
                 max_calls: int = 8, max_parallel: int = 2):
        self.store = store
        self.workspace = Path(workspace_root).resolve()
        self.llm_factory = llm_factory
        self.runner_factory = runner_factory
        self.hook_manager = hook_manager
        self.max_calls = max(1, min(MAX_AGENTS, int(max_calls)))
        self.max_parallel = max(1, min(MAX_CONCURRENT, int(max_parallel)))
        self.registry = ScriptRegistry()
        self.registry.register(
            name="review-files", description="只读逐文件代码审查与独立复核",
            phases=("Audit", "Verify", "Report"), fn=review_files,
        )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evome-s16")
        self._lock = RLock()
        self._active: dict[str, Future] = {}
        self._closed = False

    @staticmethod
    def _validate_args(name: str, args: Any) -> dict[str, Any]:
        if name != "review-files":
            raise WorkflowInputError("unknown saved S16 workflow")
        if not isinstance(args, dict) or set(args) != {"paths"}:
            raise WorkflowInputError("review-files args must contain only paths array")
        paths = args["paths"]
        if (not isinstance(paths, list) or not 1 <= len(paths) <= 3
                or len(set(map(str, paths))) != len(paths)):
            raise WorkflowInputError("review-files requires 1..3 unique paths")
        from app.agent.subagent import _check_read_path
        from pathlib import PurePosixPath
        for path in paths:
            if (not isinstance(path, str) or not 1 <= len(path) <= 180
                    or "\x00" in path or PurePosixPath(path.replace("\\", "/")).is_absolute()
                    or ".." in PurePosixPath(path.replace("\\", "/")).parts
                    or path.startswith(("\\", "/")) or ":" in path):
                raise WorkflowInputError("S16 paths must be safe workspace-relative paths")
            _check_read_path(path)
        json_bytes(args, 8000)
        return {"paths": paths}

    def _runner(self) -> AgentRunner:
        if self.runner_factory is not None:
            return self.runner_factory()
        if self.llm_factory is None:
            raise RuntimeError("S16 no agent runner configured")
        return ReadOnlyAgentRunner(
            llm_factory=self.llm_factory, workspace_root=self.workspace,
            hook_manager=self.hook_manager,
            private_roots=(self.store.root,),
        )

    def _execute(self, run_id: str, meta: WorkflowMeta, script: Callable[..., Awaitable[dict]],
                 args: dict[str, Any]) -> None:
        try:
            row = self.store.load(run_id)
            row["status"] = "running"
            row["updated_at"] = utc_now()
            row["version"] += 1
            self.store.save(row)
            context = ExecutionState(
                run_id=run_id, store=self.store, runner=self._runner(),
                meta=meta, max_calls=self.max_calls, max_parallel=self.max_parallel,
            )
            context._update(kind="task_started", message=meta.name)
            output = asyncio.run(script(context, args))
            json_bytes(output, 24_000)
            self.store.write_output(run_id, output)
            row = self.store.load(run_id)
            row["status"] = "completed"
            row["updated_at"] = utc_now()
            row["version"] += 1
            row["error"] = ""
            row["progress"] = [
                *row["progress"][-98:],
                {"kind": "task_notification", "message": "completed", "at": utc_now()},
            ]
            self.store.save(row)
        except BaseException as exc:
            # Keep journal and prior success entries unchanged, never retry
            # or forward arbitrary private file content in the error.
            try:
                row = self.store.load(run_id)
                row["status"] = "failed"
                row["error"] = type(exc).__name__[:120]
                row["updated_at"] = utc_now()
                row["version"] += 1
                row["progress"] = [
                    *row["progress"][-98:],
                    {"kind": "task_notification", "message": "failed", "at": utc_now()},
                ]
                self.store.save(row)
            except Exception:
                pass
        finally:
            self.store.unlock(run_id)

    def launch(self, name: str, args: dict[str, Any] | None = None,
               resume_from_run_id: str = "") -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("S16 workflow runtime shut down")
            if any(not f.done() for f in self._active.values()):
                raise RuntimeError("S16 only one active workflow at a time")
            meta, script = self.registry.get(name)
            if resume_from_run_id:
                previous = self.store.load(resume_from_run_id)
                if previous["workflow"] != name:
                    raise WorkflowInputError("S16 resume workflow name mismatch")
                if args is not None and args != previous["args"]:
                    raise WorkflowInputError("S16 resume args mismatch")
                args = self._validate_args(name, previous["args"])
                # Validate all journal entries before touching lock or state.
                self.store.journal(resume_from_run_id)
                if self.store.locked(resume_from_run_id):
                    raise FileExistsError("S16 run still locked; operator must inspect before resume")
                if previous["status"] == "completed":
                    output = self.store.read_output(resume_from_run_id)
                    if output is None:
                        raise ValueError("S16 completed run is missing its saved output")
                    return {"launched": False, "run_id": resume_from_run_id,
                            "status": "completed", "result": output}
                run_id = resume_from_run_id
                self.store.lock(run_id)
            else:
                args = self._validate_args(name, args)
                run_id = self.store.reserve(name, args)["id"]
            try:
                future = self._pool.submit(self._execute, run_id, meta, script, args)
                self._active[run_id] = future
            except Exception:
                self.store.unlock(run_id)
                raise
            return {"launched": True, "run_id": run_id, "status": "queued"}

    def run_tool(self, name: str, args: dict[str, Any] | None = None,
                 resume_from_run_id: str = "") -> dict[str, Any]:
        """S03-approved implementation for the single public Workflow tool."""
        return self.launch(name, args=args, resume_from_run_id=resume_from_run_id)

    def show(self, run_id: str) -> dict[str, Any]:
        row = self.store.load(run_id)
        row["journal_entries"] = len(self.store.journal(run_id))
        if row["status"] == "completed":
            row["result"] = self.store.read_output(run_id)
        return row

    def list(self) -> list[dict[str, Any]]:
        return [{
            "id": row["id"], "workflow": row["workflow"],
            "status": row["status"], "phase": row["phase"],
            "agent_calls": row["agent_calls"],
        } for row in self.store.list()]

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.load(run_id)["progress"]

    def current_runs(self) -> list[dict[str, Any]]:
        """Only this process's runs are eligible for S15 notifications.

        Do not replay old completed workflow results after an EvoMe restart.
        """
        with self._lock:
            ids = list(self._active)
        return [
            {"id": run_id, "status": self.store.load(run_id)["status"]}
            for run_id in ids
        ]

    def wait(self, run_id: str, timeout: float = 30.0) -> dict[str, Any]:
        if not 0 <= timeout <= 120:
            raise ValueError("S16 wait timeout must be 0..120 seconds")
        with self._lock:
            future = self._active.get(run_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.show(run_id)

    def unlock_orphan(self, run_id: str, *, confirmed: bool) -> dict[str, Any]:
        with self._lock:
            future = self._active.get(run_id)
            active = future is not None and not future.done()
            self.store.clear_orphan_lock(run_id, confirmed=confirmed, active=active)
            return self.show(run_id)

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=wait)
