"""S05 bounded, in-memory task plans executed through the existing AgentLoop.

Plans are explicit operator state, not new long-term Personal Memory.
A plan never executes a tool directly: every subtask uses AgentLoop and its
S03 PermissionPolicy and S04 Hooks. Approval resumes only after a human acts.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.agent.schema import AgentCheckpoint, AgentStep

if TYPE_CHECKING:
    from app.agent.loop import AgentLoop
    from app.tools.base import ToolResult


TERMINAL = frozenset({"completed", "cancelled"})
SAFE_REPLAN_TOOLS = frozenset({"glob", "read_file", "recall_relevant_memory", "plan_show"})


@dataclass
class TodoStep:
    number: int
    title: str
    status: str = "pending"
    note: str = ""
    attempts: int = 0
    approval_id: str = ""
    checkpoint: AgentCheckpoint | None = field(default=None, repr=False)
    tool_steps: tuple[AgentStep, ...] = field(default=(), repr=False)
    pause_reason: str = ""


@dataclass
class TaskPlan:
    id: str
    goal: str
    steps: list[TodoStep]
    status: str = "pending"
    replans: int = 0
    error: str = ""

    @property
    def completed_count(self) -> int:
        return sum(step.status == "completed" for step in self.steps)


class PlanStore:
    """Bounded per-runtime store; plans vanish on process restart (S05 scope)."""

    def __init__(self, *, capacity: int = 50) -> None:
        self.capacity = max(1, int(capacity))
        self._plans: OrderedDict[str, TaskPlan] = OrderedDict()
        self._lock = RLock()

    def add(self, plan: TaskPlan) -> None:
        with self._lock:
            if len(self._plans) >= self.capacity:
                for key, old in tuple(self._plans.items()):
                    if old.status in TERMINAL:
                        del self._plans[key]
                        break
                else:
                    raise ValueError("plan store is full; cancel or complete an active plan")
            self._plans[plan.id] = plan

    def get(self, plan_id: str) -> TaskPlan:
        with self._lock:
            if plan_id not in self._plans:
                raise KeyError(f"unknown plan id: {plan_id}")
            return self._plans[plan_id]

    def list(self) -> list[TaskPlan]:
        with self._lock:
            return list(self._plans.values())


class PlanningService:
    def __init__(
        self,
        agent: AgentLoop,
        *,
        store: PlanStore | None = None,
        max_tasks: int = 5,
        step_max_steps: int = 4,
        step_max_continuations: int = 2,
        max_replans: int = 1,
        auto_replan: bool = False,
        require_file_evidence: bool = False,
    ) -> None:
        self.agent = agent
        self.store = store if store is not None else PlanStore()
        self.max_tasks = max(1, min(10, int(max_tasks)))
        self.step_max_steps = max(1, min(12, int(step_max_steps)))
        self.step_max_continuations = max(0, min(3, int(step_max_continuations)))
        self.max_replans = max(0, min(3, int(max_replans)))
        self.auto_replan = bool(auto_replan)
        self.require_file_evidence = bool(require_file_evidence)
        self._running: set[str] = set()

    def _validate_steps(self, raw: Any, limit: int | None = None) -> list[str]:
        if not isinstance(raw, list):
            raise ValueError("planning response must contain a steps list")
        limit = self.max_tasks if limit is None else max(0, limit)
        if not 1 <= len(raw) <= limit:
            raise ValueError(f"plan requires 1-{limit} steps")
        titles: list[str] = []
        for value in raw:
            title = value.get("title") if isinstance(value, dict) else value
            if not isinstance(title, str):
                raise ValueError("each plan step must be a title string")
            title = title.strip()
            if not title or len(title) > 220:
                raise ValueError("each plan step needs 1-220 characters")
            titles.append(title)
        return titles

    def _generate(self, goal: str, *, prompt_extra: str = "", limit: int | None = None) -> list[str]:
        limit = self.max_tasks if limit is None else limit
        catalog = [tool.name for tool in self.agent.tool_registry.list_tools()]
        response = self.agent._decision_chat([
            {
                "role": "system",
                "content": (
                    "你是 EvoMe 的任务规划器，只返回 JSON："
                    '{"steps":["步骤一","步骤二"]}。'
                    f"生成 1 到 {limit} 个按顺序执行的具体、可验证步骤。"
                    "每步用清晰的中文任务描述，不包含工具调用的虚构结果。"
                    f"每步最多调用 {min(self.agent.max_steps, self.step_max_steps)} 次工具；"
                    "读取多个大文件时拆成较小步骤，最后单独综合分析。"
                    "优先只读检查；用户没有明确要求时不要规划改写文件或执行命令。"
                    "若需要 /plan-show，最后使用 plan_show 工具查看记录；"
                    "真实计划 ID 由运行时提供，不需要查找、猜测或执行 Shell。"
                    f"可用工具：{catalog}。"
                ),
            },
            {"role": "user", "content": goal[:2000] + "\n" + prompt_extra[:2000]},
        ], purpose="agent_plan")
        if not isinstance(response, dict) or response.get("_json_repaired"):
            raise ValueError("planner returned malformed JSON; plan not trusted")
        if response.get("action") == "output_limit":
            raise ValueError("planner output reached token limit")
        return self._validate_steps(response.get("steps"), limit)

    def create(self, goal: str, steps: list[Any]) -> TaskPlan:
        goal = str(goal).strip()
        if not goal or len(goal) > 2000:
            raise ValueError("plan goal must be 1-2000 characters")
        titles = self._validate_steps(steps)
        plan = TaskPlan(
            id=uuid4().hex[:10],
            goal=goal,
            steps=[TodoStep(number=i, title=title) for i, title in enumerate(titles, 1)],
        )
        self.store.add(plan)
        return plan

    def create_from_llm(self, goal: str) -> TaskPlan:
        return self.create(goal, self._generate(goal))

    def list(self) -> list[TaskPlan]:
        return self.store.list()

    def get(self, plan_id: str) -> TaskPlan:
        return self.store.get(plan_id)

    def inspect(self, plan_id: str) -> dict[str, str]:
        plan = self.store.get(plan_id)
        return {"plan_id": plan.id, "status": plan.status, "report": self.describe(plan, detail=True)}

    @staticmethod
    def _short(value: str, limit: int = 450) -> str:
        text = str(value or "").strip()
        return text if len(text) <= limit else text[:limit] + "…（已省略）"

    def describe(self, plan: TaskPlan, *, detail: bool = False) -> str:
        # Honor a natural-language request to inspect the actual execution log.
        detail = detail or (plan.status == "completed" and "/plan-show" in plan.goal)
        lines = [f"计划 {plan.id}｜{plan.status}｜目标：{plan.goal}"]
        for step in plan.steps:
            line = f"  {step.number}. [{step.status}] {step.title}"
            if step.note:
                line += f"\n{step.note}" if detail else f" — {self._short(step.note, 120)}"
            lines.append(line)
            if detail and step.tool_steps:
                lines.append(f"  执行批次：{step.attempts}；工具调用记录：")
                for entry in step.tool_steps:
                    call = entry.tool_call
                    if call is None:
                        if entry.action == "invalid_json_retry":
                            lines.append(f"    {entry.index}. JSON 格式重试（未执行工具）")
                        continue
                    args = call.arguments
                    target = args.get("path", args.get("pattern", ""))
                    location = f" {target}" if target else ""
                    if call.name == "read_file":
                        output = entry.tool_output if isinstance(entry.tool_output, dict) else {}
                        start = output.get("start_line", args.get("start_line", 1))
                        end = output.get("end_line", args.get("end_line") or "end")
                        location += f" [{start}:{end}]"
                        if output.get("truncated"):
                            next_line = output.get("next_start_line")
                            location += f"（未读完，下一段从 {next_line} 行开始）" if next_line else "（单行内容被截断）"
                    lines.append(f"    {entry.index}. {call.name}{location} — {entry.error or '成功'}")
                    if call.name == "delegate_research" and isinstance(entry.tool_output, dict):
                        delegated = entry.tool_output
                        if delegated.get("status") == "completed":
                            lines.append(
                                f"      Research Subagent："
                                f"{delegated.get('tool_call_count', 0)} 次工具调用"
                            )
                            for source in delegated.get("evidence", [])[:20]:
                                if not isinstance(source, dict):
                                    continue
                                source_path = source.get("path", "")
                                span = (
                                    f" [{source['start_line']}:{source['end_line']}]"
                                    if "start_line" in source and "end_line" in source else ""
                                )
                                lines.append(
                                    f"        ↳ {source.get('tool', '?')} {source_path}{span}"
                                    + ("（截断）" if source.get("truncated") else "")
                                )
        if not detail:
            completed = [step for step in plan.steps if step.status == "completed"]
            if plan.status == "completed" and completed:
                lines.append("\n执行结果：\n" + completed[-1].note)
            if any(len(step.note) > 120 for step in plan.steps):
                lines.append(f"完整步骤结果：/plan-show {plan.id}")
        if plan.status == "waiting_approval":
            task = next(step for step in plan.steps if step.status == "waiting_approval")
            lines.append(f"等待人工审批：/tool-approve {task.approval_id}")
            lines.append(f"或拒绝：/tool-deny {task.approval_id}")
            lines.append(f"批准后输入 /plan-run {plan.id} 继续。")
        elif plan.status == "paused":
            task = next(step for step in plan.steps if step.status == "paused")
            lines.append(task.pause_reason or "当前步骤未确认完成，已保存进度。")
            lines.append(f"输入 /plan-run {plan.id} 从断点继续，无需重新规划。")
        elif plan.status == "failed":
            lines.append(f"任务暂停：{plan.error if detail else self._short(plan.error, 180)}")
            lines.append(f"可以输入 /plan-replan {plan.id} 重新规划。")
        elif plan.status == "pending":
            lines.append(f"输入 /plan-run {plan.id} 继续。")
        return "\n".join(lines)

    def _task_prompt(self, plan: TaskPlan, step: TodoStep) -> str:
        recent = [
            f"{item.number}. {item.title}: {self._short(item.note, 4000)}"
            for item in plan.steps
            if item.status == "completed"
        ][-self.max_tasks:]
        return (
            "[PLAN_STEP]\n"
            f"总体目标（仅背景，不是本次需全部完成的任务）：{plan.goal}\n"
            f"当前计划 ID：{plan.id}（plan_show 工具使用此 ID）\n"
            "已完成步骤（若有）：\n" + ("\n".join(recent) or "暂无") + "\n"
            f"本次唯一任务（第 {step.number} 步）：{step.title}\n"
            "只处理当前步骤；需要读取数据就调用已有工具。"
            "如果修改文件或执行命令需要审批，请停止等待，不得绕过权限。"
            "不要声称完成未执行的操作。\n"
        )

    @staticmethod
    def _task_context(step: TodoStep) -> list[dict[str, str]]:
        return [{"role": "system", "content": (
            f"你是计划中一个步骤的执行者，本次唯一任务是：{step.title}。"
            "总体目标和已完成步骤仅供参考，其他待办由调度器随后执行。"
            "本步骤完成就返回 task_complete=true，不要提前做后续步骤。"
            "如果本步骤仅要求列出目录，成功的 glob 结果足够，不需要 read_file 或跨目录分析。"
            "只有本步骤明确要求查看计划记录时才调用 plan_show，不要因总体目标提到它而提前调用。"
            "不得因后续步骤尚未完成就把当前已完成步骤判为未完成。"
        )}]

    @staticmethod
    def _normalize_path(value: str) -> str:
        text = str(value or "").replace("\\", "/").strip().casefold()
        while text.startswith("./"):
            text = text[2:]
        return text

    @classmethod
    def _verified_reads(
        cls,
        history: tuple[AgentStep, ...],
        targets: tuple[str, ...] = (),
    ) -> bool:
        """Check real FileSystemTools pagination when metadata is available.

        Fake tools with a successful dict response remain supported for
        compatibility; actual read_file returns total_lines/start/end fields.
        """
        pages: dict[str, list[dict]] = {}
        for entry in history:
            call = entry.tool_call
            if call is None or entry.error or not isinstance(entry.tool_output, dict):
                continue
            output = entry.tool_output
            if call.name == "read_file":
                read_rows = [output]
            elif (
                call.name == "delegate_research"
                and output.get("status") == "completed"
                and isinstance(output.get("evidence"), list)
            ):
                # S06 child returns source provenance, not file content. Its
                # own permission-checked read_file trace is the evidence.
                read_rows = [
                    row for row in output["evidence"]
                    if isinstance(row, dict) and row.get("tool") == "read_file"
                ]
            else:
                continue
            for row in read_rows:
                pathname = cls._normalize_path(
                    row.get("path") or call.arguments.get("path", "")
                )
                if pathname:
                    pages.setdefault(pathname, []).append(row)

        if not pages:
            return False

        def complete(parts: list[dict]) -> bool:
            if any(item.get("line_truncated") for item in parts):
                return False
            metadata = [
                item for item in parts
                if isinstance(item.get("total_lines"), int)
                and isinstance(item.get("start_line"), int)
                and isinstance(item.get("end_line"), int)
            ]
            if not metadata:
                # Legacy/unit-test tools without paging metadata.
                return all(not item.get("truncated") for item in parts)
            if len(metadata) != len(parts):
                return False
            totals = {part["total_lines"] for part in metadata}
            if len(totals) != 1:
                return False
            total = totals.pop()
            if total == 0:
                return any(part["start_line"] == 1 for part in metadata)
            covered = 0
            ordered = sorted(metadata, key=lambda item: item["start_line"])
            for part in ordered:
                if part["start_line"] > covered + 1:
                    return False
                covered = max(covered, part["end_line"])
            # Reaching the last line is not enough when a tool reports a
            # truncated final page; do not treat partial output as complete.
            last_page = max(metadata, key=lambda item: item["end_line"])
            return covered >= total and not last_page.get("truncated", False)

        verified = {path for path, parts in pages.items() if complete(parts)}
        if not targets:
            return bool(verified)
        normalized_targets = [cls._normalize_path(path) for path in targets]
        return all(
            any(
                actual == target
                or ("/" not in target and actual.rsplit("/", 1)[-1] == target)
                for actual in verified
            )
            for target in normalized_targets
        )

    @classmethod
    def _same_target(cls, actual: str, requested: str) -> bool:
        # Mutations must match the requested relative path, not merely the
        # basename (subdir/demo.txt is not the requested root demo.txt).
        return cls._normalize_path(actual) == cls._normalize_path(requested)

    @classmethod
    def _verified_writes(
        cls,
        history: tuple[AgentStep, ...],
        targets: tuple[str, ...] = (),
    ) -> bool:
        """A successful write/edit is evidence of mutation, not of later readback.

        For S03 approvals, the actual executed ToolResult is reinserted into the
        pending AgentStep by checkpoint_after_approval. An approval request alone
        has step.error and no output, and never satisfies this check.
        """
        written: set[str] = set()
        for entry in history:
            call = entry.tool_call
            if call is None or call.name not in {"write_file", "edit_file"}:
                continue
            if entry.error or not isinstance(entry.tool_output, dict):
                continue
            requested_by_tool = call.arguments.get("path")
            if not isinstance(requested_by_tool, str) or not requested_by_tool:
                continue
            actual = entry.tool_output.get("path", requested_by_tool)
            if not isinstance(actual, str) or not cls._same_target(actual, requested_by_tool):
                continue
            written.add(cls._normalize_path(actual))
        if not targets:
            return bool(written)
        return all(
            any(cls._same_target(actual, wanted) for actual in written)
            for wanted in targets
        )

    @classmethod
    def _verified_readback_matches_write(
        cls,
        plan: TaskPlan,
        step: TodoStep,
        targets: tuple[str, ...],
    ) -> bool:
        """For a read-and-confirm step, compare actual content to prior writes.

        FileSystemTools returns numbered read_file lines. When both exact write
        content and numbered read content exist, validate the logical lines.
        Legacy tool adapters without 'content' retain their prior behavior.
        """
        relevant_steps = [
            item
            for old_step in plan.steps
            if old_step.status == "completed"
            for item in old_step.tool_steps
        ] + list(step.tool_steps)
        for requested in targets:
            expected_content: str | None = None
            wrote_after_rewrite = False
            read_pages: list[dict] = []
            for item in relevant_steps:
                call = item.tool_call
                if call is None or item.error or not isinstance(item.tool_output, dict):
                    continue
                source = item.tool_output.get("path") or call.arguments.get("path")
                if not isinstance(source, str) or not cls._same_target(source, requested):
                    continue
                if call.name == "write_file":
                    data = call.arguments.get("content")
                    append = bool(call.arguments.get("append", False))
                    if isinstance(data, str) and not append:
                        expected_content = data
                        wrote_after_rewrite = False
                        read_pages = []
                    elif append:
                        wrote_after_rewrite = True
                elif call.name == "edit_file":
                    wrote_after_rewrite = True
                elif call.name == "read_file" and item in step.tool_steps:
                    read_pages.append(item.tool_output)

            if expected_content is None or wrote_after_rewrite or not read_pages:
                continue
            # Actual FileSystemTools output includes numbered text; mocks
            # lacking content cannot be compared (but must pass read evidence).
            if any(not isinstance(page.get("content"), str) for page in read_pages):
                continue
            lines: list[str] = []
            for page in sorted(read_pages, key=lambda x: x.get("start_line", 1)):
                raw = page["content"]
                for line in raw.splitlines():
                    match = re.match(r"^\d+: (.*)$", line)
                    if match is None:
                        return False
                    lines.append(match.group(1))
            if lines != expected_content.splitlines():
                return False
        return True

    @classmethod
    def _file_tool_evidence(cls, history: tuple[AgentStep, ...]) -> bool:
        if cls._verified_reads(history):
            return True
        for step in history:
            call = step.tool_call
            if call is None or step.error or not isinstance(step.tool_output, dict):
                continue
            if call.name == "glob":
                return True
            if (
                call.name == "delegate_research"
                and step.tool_output.get("status") == "completed"
                and isinstance(step.tool_output.get("evidence"), list)
                and any(
                    isinstance(row, dict) and row.get("tool") == "glob"
                    for row in step.tool_output["evidence"]
                )
            ):
                return True
        return False

    def _has_required_file_evidence(self, plan: TaskPlan, step: TodoStep) -> bool:
        if not self.require_file_evidence:
            return True

        title = step.title.lower()
        goal = plan.goal.lower()
        inspects_plan = "/plan-show" in title or "plan_show" in title
        if inspects_plan:
            has_plan_record = any(
                entry.tool_call is not None
                and entry.tool_call.name == "plan_show"
                and not entry.error
                and isinstance(entry.tool_output, dict)
                and entry.tool_output.get("plan_id") == plan.id
                for entry in step.tool_steps
            )
            if not has_plan_record:
                return False

        file_goal = any(
            word in goal
            for word in ("文件", "代码", "源码", "目录", "仓库", "readme", "app/")
        )
        # Named file requirements are checked against the path actually read.
        targets = tuple(re.findall(
            r"[a-z0-9_.-]+(?:[/\\][a-z0-9_.-]+)*\.(?:py|md|txt|json|toml|yaml|yml)\b",
            title,
            flags=re.IGNORECASE,
        ))
        explicitly_writing = any(
            word in title
            for word in ("write_file", "edit_file", "写入", "创建", "新建",
                         "编辑", "修改", "追加", "保存到")
        )
        explicitly_reading = any(
            word in title
            for word in ("读取", "阅读", "打开", "read_file", "查看源码", "查看代码")
        )
        # Mutation steps require evidence from the mutation tool itself.
        # A write-only step must NOT be rejected for lack of read_file/glob.
        if explicitly_writing and not self._verified_writes(step.tool_steps, targets):
            return False
        if explicitly_reading:
            if not self._verified_reads(step.tool_steps, targets):
                return False
            if any(word in title for word in ("确认", "核对", "验证", "比对", "一致")):
                return self._verified_readback_matches_write(plan, step, targets)
            return True
        if explicitly_writing:
            return True

        if targets:
            # A pure analysis step may reuse earlier actual file reads,
            # but must still match the named path in the task.
            previous = tuple(
                item
                for finished in plan.steps
                if finished.status == "completed"
                for item in finished.tool_steps
            )
            return self._verified_reads(previous + step.tool_steps, targets)

        # A plan-record-only inspection never substitutes for source content.
        if inspects_plan and not any(
            word in title for word in ("分析", "综合", "汇总", "总结", "读取", "文件", "源码", "代码")
        ):
            return True

        asks_listing = any(
            word in title
            for word in ("列出", "目录结构", "扫描", "搜索", "检索", "glob")
        )
        if asks_listing:
            return self._file_tool_evidence(step.tool_steps)

        history = tuple(
            entry
            for completed in plan.steps
            if completed.status == "completed"
            for entry in completed.tool_steps
        )
        source_analysis = (
            any(word in goal for word in ("源码", "代码", "架构"))
            or any(word in title for word in ("源码", "代码", "架构"))
        )
        if source_analysis:
            # Summaries may reuse earlier verified code reads, but a glob of
            # filenames alone is not enough to substantiate source analysis.
            return self._verified_reads(history + step.tool_steps)
        if file_goal or any(word in title for word in ("目录", "文件", "查看", "检查", "分析")):
            return self._file_tool_evidence(history + step.tool_steps)
        return True

    def run(self, plan_id: str) -> TaskPlan:
        plan = self.store.get(plan_id)
        if plan.status in TERMINAL | {"waiting_approval"}:
            return plan
        if plan.status == "failed":
            return plan
        if plan_id in self._running:
            raise ValueError("plan is already running")
        self._running.add(plan_id)
        continuations: dict[int, int] = {}
        try:
            if plan.status == "paused":
                for step in plan.steps:
                    if step.status == "paused":
                        step.status = "pending"
            while True:
                step = next((s for s in plan.steps if s.status == "pending"), None)
                if step is None:
                    plan.status = "completed"
                    plan.error = ""
                    return plan

                plan.status = "running"
                step.status = "running"
                step.attempts += 1
                previous_calls = sum(entry.tool_call is not None for entry in step.tool_steps)
                try:
                    result = self.agent.run(
                        user_input=self._task_prompt(plan, step),
                        context_messages=self._task_context(step),
                        allow_planning=False,
                        max_steps_override=self.step_max_steps,
                        resume_from=step.checkpoint,
                    )
                except Exception as exc:
                    step.status = "failed"
                    step.note = self._short(f"{type(exc).__name__}: {exc}")
                    plan.status = "failed"
                    plan.error = step.note
                    return plan

                step.tool_steps = result.steps
                step.checkpoint = None
                if result.termination_reason == "approval_required":
                    if result.checkpoint is None:
                        step.status = "failed"
                        plan.status = "failed"
                        plan.error = "approval requested without resumable checkpoint"
                        return plan
                    step.checkpoint = result.checkpoint
                    step.status = "waiting_approval"
                    step.approval_id = result.approval_id
                    step.note = "waiting for explicit approval"
                    plan.status = "waiting_approval"
                    return plan

                if result.termination_reason == "final_answer" and result.tool_error_count == 0:
                    if not self._has_required_file_evidence(plan, step):
                        step.status = "failed"
                        step.note = (
                            "文件操作证据不足：缺少必要的真实写入/读取结果，"
                            "或读取内容与写入预期不一致，不能标记步骤完成。"
                        )
                        plan.status = "failed"
                        plan.error = f"step {step.number}: missing_file_evidence"
                        return plan
                    step.status = "completed"
                    step.note = str(result.answer or "").strip()
                    continue

                resumable = (
                    result.termination_reason in {"max_steps", "incomplete"}
                    and result.checkpoint is not None
                    and result.tool_error_count == 0
                    and bool(result.tool_calls)
                    and all(call.name in SAFE_REPLAN_TOOLS for call in result.tool_calls)
                )
                if resumable:
                    step.checkpoint = result.checkpoint
                    step.note = str(result.answer or "").strip()
                    made_progress = len(result.tool_calls) > previous_calls
                    used = continuations.get(step.number, 0)
                    if made_progress and used < self.step_max_continuations:
                        continuations[step.number] = used + 1
                        step.status = "pending"
                        continue
                    step.status = "paused"
                    step.pause_reason = (
                        "自动续跑预算已用完，当前步骤未确认完成，已保存进度。" if made_progress else
                        "模型没有产生新的工具调用，也未确认当前步骤完成；已停止自动重复并保存进度。"
                    )
                    plan.status = "paused"
                    plan.error = f"step {step.number}: {result.termination_reason}; progress saved"
                    return plan

                step.status = "failed"
                step.note = str(result.answer or result.termination_reason).strip()
                plan.status = "failed"
                plan.error = (
                    f"step {step.number}: {result.termination_reason}; "
                    f"tool_errors={result.tool_error_count}"
                )
                safe = all(call.name in SAFE_REPLAN_TOOLS for call in result.tool_calls)
                if self.auto_replan and safe and result.termination_reason not in {"protocol_error", "output_limit"} and plan.replans < self.max_replans:
                    try:
                        self.replan(plan.id)
                    except (ValueError, TypeError, KeyError) as exc:
                        plan.status = "failed"
                        plan.error = self._short(f"replan failed: {exc}")
                        return plan
                    continue
                return plan
        finally:
            self._running.remove(plan_id)

    def on_approval(self, approval_id: str, result: ToolResult | None, *, denied: bool = False) -> bool:
        for plan in self.store.list():
            if plan.status != "waiting_approval":
                continue
            task = next((s for s in plan.steps if s.approval_id == approval_id), None)
            if task is None:
                continue
            task.approval_id = ""
            if denied or result is None or not result.success:
                task.status = "failed"
                task.note = self._short(result.error if result else "approval denied")
                task.checkpoint = None
                plan.status = "failed"
                plan.error = task.note
            else:
                try:
                    if task.checkpoint is None:
                        raise ValueError("approval checkpoint missing")
                    continuation = self.agent.checkpoint_after_approval(
                        task.checkpoint, approval_id, result,
                    )
                except (ValueError, TypeError) as exc:
                    # The tool has already executed; never retry it implicitly.
                    task.status = "failed"
                    task.note = self._short(f"cannot resume approved tool: {exc}")
                    task.checkpoint = None
                    plan.status = "failed"
                    plan.error = task.note
                else:
                    task.checkpoint = continuation
                    task.tool_steps = continuation.steps
                    task.status = "pending"
                    task.note = (
                        "工具已批准并执行；原步骤尚未验证完成。"
                        "请使用 /plan-run 继续执行剩余操作。"
                    )
                    plan.status = "pending"
                    plan.error = ""
            return True
        return False

    def replan(self, plan_id: str) -> TaskPlan:
        plan = self.store.get(plan_id)
        if plan.status not in {"failed", "paused"}:
            raise ValueError("only failed or paused plans may be replanned")
        if plan.replans >= self.max_replans:
            raise ValueError("replan budget exhausted")
        completed = [step for step in plan.steps if step.status == "completed"]
        remaining = self.max_tasks - len(completed)
        if remaining < 1:
            raise ValueError("no remaining task budget")
        context = "\n".join(
            f"{task.number}. [{task.status}] {task.title}: {self._short(task.note, 200)}"
            for task in plan.steps
        )
        new_titles = self._generate(
            plan.goal,
            prompt_extra=(
                "之前执行失败。不要重复已经完成的步骤；根据以下记录"
                "重新规划尚未完成的任务，优先只读安全路径：\n" + context
            ),
            limit=remaining,
        )
        for step in plan.steps:
            if step.status in {"failed", "pending", "paused"}:
                step.status = "skipped"
                step.checkpoint = None
        next_number = 1 + max((step.number for step in plan.steps), default=0)
        plan.steps.extend(
            TodoStep(number=next_number + i, title=title)
            for i, title in enumerate(new_titles)
        )
        plan.replans += 1
        plan.status = "pending"
        plan.error = ""
        return plan

    def cancel(self, plan_id: str) -> TaskPlan:
        plan = self.store.get(plan_id)
        if plan.status in TERMINAL:
            return plan
        if plan_id in self._running:
            raise ValueError("cannot cancel a plan during a synchronous run")
        for step in plan.steps:
            if step.status in {"pending", "waiting_approval", "failed", "paused"}:
                step.status = "cancelled"
                step.approval_id = ""
                step.checkpoint = None
        plan.status = "cancelled"
        return plan
