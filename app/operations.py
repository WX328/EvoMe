"""Operator help and read-only diagnostics. Never sends a model request."""
from __future__ import annotations
import sqlite3

from app.version import APP_VERSION

HELP_GROUPS = {
    "start": ("开始与继续", "/run <完成条件>", "/continue", "/goal", "/goal clear", "/doctor", "/trace [trace_id]", "/exit"),
    "approval": ("人工审批", "/tool-approvals", "/approve <id>", "/deny <id>", "审批十分钟内有效；失效后重新提交操作。"),
    "workflow": ("工作流", "/workflow-catalog", "/workflow-list", "/workflow-show <id>", "/workflow-resume <id>", "/workflow-unlock <id> CONFIRM"),
    "tasks": ("任务与协作", "/task-list", "/task-ready", "/task-bg-list", "/cron-list", "/team-status", "/harness-status"),
    "memory": ("记忆与学习", "/memory", "/agent-memory-list", "/experience", "/reflection", "/hypothesis", "/candidates"),
    "tools": ("能力与用量", "/skills", "/skill-docs", "/mcp-servers", "/hooks", "/context-stats", "/token-usage"),
}


def command_help(topic=""):
    from app.cli_commands import COMMAND_NAMES
    if topic == "all":
        return "全部命令：\n" + "\n".join("/" + name for name in sorted(COMMAND_NAMES))
    if not topic:
        return "常用：/run <目标> → /approve <id> → /continue\n" + "\n".join(
            f"/help {name}  {items[0]}" for name, items in HELP_GROUPS.items()) + "\n/help all  全部命令"
    if topic not in HELP_GROUPS:
        return "未知帮助分类；输入 /help 查看分类。"
    return "\n".join(HELP_GROUPS[topic])


def diagnose(runtime):
    """Report state without changing jobs, locks, approvals, or memory."""
    findings = []
    pending = []
    store = runtime.tool_approval_store
    for item in store.list_pending():
        expired = store.expired(item)
        pending.append({"id": item.id, "tool": item.tool_name, "risk": item.risk,
                        "status": "expired" if expired else "awaiting_approval"})
        if expired:
            findings.append({"kind": "expired_approval", "ref": item.id,
                "next": [f"/deny {item.id}", "/continue"],
                "note": "审批已过期，需重新提出操作；不会自动执行。"})
    workflow = getattr(runtime, "workflow_runtime", None)
    if workflow:
        try:
            active = {row["id"] for row in workflow.current_runs()
                      if row["status"] in {"queued", "running"}}
            for row in workflow.store.list():
                if row["status"] in {"queued", "running"} and row["id"] not in active:
                    findings.append({"kind": "workflow_needs_inspection", "ref": row["id"],
                        "next": [f"/workflow-show {row['id']}",
                                 f"/workflow-unlock {row['id']} CONFIRM", f"/workflow-resume {row['id']}"],
                        "note": "可能由其他进程运行或已经中断。先核实原进程退出，再手工解锁并重新审批续跑。"})
                elif row["status"] == "failed":
                    findings.append({"kind": "workflow_failed", "ref": row["id"],
                        "next": [f"/workflow-show {row['id']}", f"/workflow-resume {row['id']}"]})
        except (OSError, ValueError, RuntimeError) as exc:
            findings.append({"kind": "workflow_store_error", "error_type": type(exc).__name__, "next": []})
    tracer = getattr(runtime, "trace_recorder", None)
    if getattr(runtime, "trace_error", "") or (tracer and tracer.dropped_events):
        findings.append({"kind": "trace_sink_error", "dropped_events": tracer.dropped_events if tracer else 0,
                         "next": ["检查 .traces 目录权限和 EVOME_TOKEN_PRICES_JSON"]})
    elif tracer:
        try:
            tracer.events(limit=1)
        except (OSError, ValueError, sqlite3.Error) as exc:
            findings.append({"kind": "trace_store_error", "error_type": type(exc).__name__,
                             "next": ["检查 .traces 账本，先备份再修复；业务操作不会自动重放。"]})
    goal = getattr(runtime, "goal_controller", None)
    goal_state = goal.status() if goal else {"active": False, "status": "disabled"}
    # A completion condition may contain private source/user text.
    goal_state.pop("condition", None)
    goal_state.pop("last_reason", None)
    return {"version": APP_VERSION, "status": "attention" if findings else "ok",
            "goal": goal_state, "pending_approvals": pending,
            "tracing": "enabled" if tracer else "disabled", "findings": findings}
