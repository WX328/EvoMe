"""S05 bounded, in-memory task plans executed through the existing AgentLoop.

Plans are explicit operator state, not new long-term Personal Memory.
A plan never executes a tool directly: every subtask uses AgentLoop and its
S03 PermissionPolicy and S04 Hooks. Approval resumes only after a human acts.
"""
from __future__ import annotations

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
    def _file_tool_evidence(steps: tuple[AgentStep, ...]) -> bool:
        return any(
            s.tool_call is not None
            and s.tool_call.name in {"glob", "read_file"}
            and not s.error
            and isinstance(s.tool_output, dict)
            for s in steps
        )

    def _has_required_file_evidence(self, plan: TaskPlan, step: TodoStep) -> bool:
        if not self.require_file_evidence:
            return True
        inspects_plan = "/plan-show" in step.title or "plan_show" in step.title
        if inspects_plan:
            has_plan_record = any(
                entry.tool_call is not None and entry.tool_call.name == "plan_show"
                and not entry.error and isinstance(entry.tool_output, dict)
                and entry.tool_output.get("plan_id") == plan.id
                for entry in step.tool_steps
            )
            if not has_plan_record:
                return False
            if not any(word in step.title for word in ("读取", "文件", "源码", "目录", "代码", "分析", "综合", "总结")):
                return True
        file_goal = any(
            word in plan.goal.lower()
            for word in ("文件", "代码", "源码", "目录", "仓库", "readme", "app/")
        )
        file_step = any(
            word in step.title.lower()
            for word in ("读取", "查看", "列出", "搜索", "扫描", "检查", "检索", "打开")
        )
        if inspects_plan and any(word in step.title for word in ("综合", "汇总", "总结")):
            # A combined summary + plan inspection may "view" the execution
            # log while reusing earlier source evidence. An explicit request
            # to read/check files still requires this step's own file result.
            file_step = any(word in step.title.lower() for word in (
                "读取", "列出", "搜索", "扫描", "检查", "检索", "打开", "read_file", "glob",
                "查看文件", "查看源码", "查看代码", "查看目录", "查看 app/",
            ))
        if file_step:
            # Each file-reading step must have its own successful tool result.
            return self._file_tool_evidence(step.tool_steps)
        if file_goal:
            # Pure summary steps may reuse previously grounded evidence.
            return any(self._file_tool_evidence(s.tool_steps) for s in plan.steps)
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
                    step.status = "waiting_approval"
                    step.approval_id = result.approval_id
                    step.note = "waiting for explicit approval"
                    plan.status = "waiting_approval"
                    return plan

                if result.termination_reason == "final_answer" and result.tool_error_count == 0:
                    if not self._has_required_file_evidence(plan, step):
                        step.status = "failed"
                        step.note = (
                            "缺少实际文件工具结果：本步骤尚未成功执行 "
                            "read_file / glob，不能根据模型描述判定完成。"
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
                plan.status = "failed"
                plan.error = task.note
            else:
                task.status = "completed"
                task.note = "Approved tool executed successfully; result available in CLI."
                plan.status = "pending"
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
