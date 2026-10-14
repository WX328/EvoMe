"""Local read-only runtime dashboard. Opening it never creates a model client."""
from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from app.telemetry.tracing import TraceRecorder
from app.version import APP_VERSION


def status_text(status):
    return {None: "—", "completed": "已完成", "final_answer": "已回答",
            "achieved": "目标已达成", "goal_achieved": "目标已达成", "failed": "失败",
            "error": "异常", "awaiting_approval": "等待审批", "approval_required": "等待审批",
            "executed": "已执行", "goal_deferred": "等待后台", "deferred": "等待后台",
            "finished": "已结束", "unfinished": "尚未结束", "partial": "记录不完整",
            "running": "运行中", "queued": "已排队", "active": "进行中",
            "ready": "等待继续核验", "cleared": "已清除",
            "achieved_pending_answer": "等待交付回答"}.get(status, status)


class RuntimeDashboard:
    def __init__(self, root, workspace, *, demo=False):
        self.root = root
        self.recorder = TraceRecorder(workspace, read_only=True)
        self.selected = ""
        root.title(f"EvoMe · 运行面板 {APP_VERSION}" + (" · 演示数据" if demo else ""))
        root.geometry("1120x900")
        root.minsize(900, 820)
        root.configure(bg="#0c1424")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#0c1424")
        style.configure("TLabel", background="#0c1424", foreground="#dce6f5", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Muted.TLabel", foreground="#9aaec7")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(14, 7))
        style.configure("Treeview", background="#142136", fieldbackground="#142136",
                        foreground="#dce6f5", rowheight=29, borderwidth=0, font=("Microsoft YaHei UI", 10))
        style.configure("Treeview.Heading", background="#20314a", foreground="#b7cbe4", padding=7)
        style.map("Treeview", background=[("selected", "#174f61")])
        shell = ttk.Frame(root, padding=24)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text="EvoMe  /  运行概览", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="刷新", command=self.refresh).pack(side="right")
        ttk.Label(shell, text=("演示数据 · 无模型调用" if demo else "本地只读面板") +
                  "   ·   Goal → Workflow → Agent → Tool", style="Muted.TLabel").pack(anchor="w", pady=(8, 15))
        self.summary = ttk.Label(shell, text="", font=("Microsoft YaHei UI", 13, "bold"))
        self.summary.pack(anchor="w", pady=(0, 16))
        self.traces = ttk.Treeview(shell, columns=("time", "goal", "result", "activity", "tokens", "cost"), height=5)
        self.traces.heading("#0", text="调用链 Trace")
        self.traces.column("#0", width=210)
        for key, title, width in (("time", "最近活动", 130), ("goal", "Goal 状态", 140),
                                  ("result", "入口结果", 125), ("activity", "执行 / 子任务", 155),
                                  ("tokens", "Token", 75), ("cost", "估算 USD", 100)):
            self.traces.heading(key, text=title)
            self.traces.column(key, width=width, anchor="w")
        self.traces.pack(fill="x")
        self.traces.bind("<<TreeviewSelect>>", self.select_trace)
        ttk.Label(shell, text="调用关系与执行结果", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(19, 9))
        self.chain = ttk.Treeview(shell, columns=("status", "ms", "tokens"), height=8)
        self.chain.heading("#0", text="组件 / 操作")
        self.chain.column("#0", width=610)
        for key, title, width in (("status", "状态", 150), ("ms", "耗时 ms", 105), ("tokens", "Token", 95)):
            self.chain.heading(key, text=title)
            self.chain.column(key, width=width)
        self.chain.pack(fill="both", expand=True)
        self.chain.tag_configure("error", foreground="#ffad9d")
        self.chain.tag_configure("ok", foreground="#7cdec6")
        self.chain.tag_configure("pending", foreground="#f2cd81")
        footer = ttk.Frame(shell)
        footer.pack(fill="x", pady=(15, 7))
        ttk.Label(footer, text="CLI：/run <目标>   /approve <id>   /continue   /doctor", style="Muted.TLabel").pack(side="left")
        ttk.Button(footer, text="复制查询命令", command=self.copy_command).pack(side="right")
        self.message = ttk.Label(shell, text="", style="Muted.TLabel", wraplength=1030)
        self.message.pack(anchor="w")
        self.refresh()
        self.timer = root.after(2000, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self.root.after_cancel(self.timer)
        self.recorder.close()
        self.root.destroy()

    def refresh(self):
        try:
            rows = self.recorder.traces()
            self.traces.delete(*self.traces.get_children())
            for row in rows:
                cost = "未提供估算" if row["estimated_usd"] is None else f"{row['estimated_usd']:.6f}"
                activity = status_text(row["execution_status"])
                if row["subtask_failures"]:
                    activity += f" · {row['subtask_failures']} 异常"
                observed = datetime.fromtimestamp(row["timestamp"])
                activity_time = observed.strftime("%H:%M:%S" if observed.date() == datetime.now().date() else "%m-%d %H:%M")
                self.traces.insert("", "end", iid=row["trace_id"], text=row["trace_id"][:12] + "…", values=(
                    activity_time,
                    status_text(row["goal_status"]) if row["goal_status"] else "无 Goal 记录",
                    status_text(row["root_status"]), activity, row["tokens"], cost))
            self.summary.configure(text=f"最近 {len(rows)} 条调用链     ·     {sum(r['tokens'] for r in rows):,} Token     ·     "
                f"{sum(r['goal_status'] == 'achieved' for r in rows)} 个目标达成     ·     "
                f"{sum(r['execution_status'] == 'unfinished' for r in rows)} 条尚未结束")
            if self.selected in {r["trace_id"] for r in rows}:
                self.traces.selection_set(self.selected)
            elif rows:
                self.selected = rows[0]["trace_id"]
                self.traces.selection_set(self.selected)
            else:
                self.selected = ""
            self.render_chain()
            self.message.configure(text=("Goal 结果独立于子任务；子任务完成不代表目标达成。尚未结束或记录不完整时请结合 /doctor 核查。"
                "Token 为已观测用量；缺少 usage 或价格时不估算金额。" if rows else
                "暂无运行记录。启动 python main.py 并执行任务后，调用链将显示在这里。"))
        except Exception as exc:
            # Do not expose arbitrary exception text or stale successful data.
            self.traces.delete(*self.traces.get_children())
            self.chain.delete(*self.chain.get_children())
            self.summary.configure(text="运行记录暂不可用")
            self.message.configure(text=f"读取运行记录失败：{type(exc).__name__}。请检查目录权限并运行 /doctor。")

    def select_trace(self, _event=None):
        chosen = self.traces.selection()
        if chosen:
            self.selected = chosen[0]
            self.render_chain()

    def render_chain(self):
        self.chain.delete(*self.chain.get_children())
        if not self.selected:
            return
        rows = self.recorder.events(self.selected, limit=2000)
        for row in rows:
            key = row["span_id"]
            if row["phase"] in {"start", "usage", "goal"} or not self.chain.exists(key):
                parent = row["parent_id"] if self.chain.exists(row["parent_id"]) else ""
                self.chain.insert(parent, "end", iid=key, open=True,
                    text=f"{row['component']} / {row['operation']}" + (f"  ·  {row['reference']}" if row["reference"] else "") +
                         ("  (起点已过期)" if row["phase"] == "end" else ""),
                    values=(status_text(row["status"]), "", row["prompt_tokens"] + row["completion_tokens"] if row["phase"] == "usage" else ""))
            if row["phase"] == "end":
                self.chain.item(key, values=(status_text(row["status"]), f"{row['elapsed_ms']:.1f}", ""))
            self.chain.item(key, tags=("error" if row["status"] in {
                "error", "failed", "denied", "expired", "invalidated", "goal_failed", "goal_error"}
                else "pending" if row["status"] in {"running", "queued", "approval_required", "awaiting_approval", "deferred", "goal_deferred"}
                else "ok",))

    def copy_command(self):
        self.root.clipboard_clear()
        self.root.clipboard_append("/trace " + self.selected if self.selected else "/doctor")

    def poll(self):
        self.refresh()
        self.timer = self.root.after(2000, self.poll)


def seed_demo(workspace):
    recorder = TraceRecorder(workspace)
    with recorder.span("runtime", "goal_continue", trace_id="a" * 32) as result:
        with recorder.span("agent", "run"):
            with recorder.span("approval", "approve", reference="a1b2c3d4"):
                with recorder.span("workflow", "run", reference="wf_aabbccddeeff"):
                    with recorder.span("agent", "workflow_agent"):
                        with recorder.span("tool", "execute", reference="read_file"):
                            recorder.usage(purpose="agent_loop", model="demo", prompt_tokens=1200,
                                           completion_tokens=180, cached_tokens=400)
        recorder.goal_state("achieved")
        result["status"] = "goal_achieved"
    # Demonstrate that successful background work cannot complete a Goal.
    with recorder.span("runtime", "goal_continue", trace_id="b" * 32) as result:
        recorder.goal_state("deferred")
        result["status"] = "goal_deferred"
        from app.telemetry.tracing import current_trace
        waiting_parent = current_trace()
    with recorder.span("workflow", "run", parent=waiting_parent, reference="wf_112233445566"):
        pass
    # Goal and subtask outcomes remain independently visible even on failure.
    with recorder.span("runtime", "goal_continue", trace_id="c" * 32) as result:
        with recorder.span("workflow", "run", reference="wf_665544332211") as child:
            child["status"] = "failed"
        recorder.goal_state("awaiting_approval")
        result["status"] = "approval_required"
    recorder.close()


def main():
    parser = argparse.ArgumentParser(description="EvoMe 本地运行面板（只读，无模型调用）")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="evome-dashboard-") as temporary:
        workspace = Path(temporary) if args.demo else args.workspace
        if args.demo:
            seed_demo(workspace)
        root = tk.Tk()
        dashboard = RuntimeDashboard(root, workspace, demo=args.demo)
        if args.smoke_test:
            root.update()
            if args.demo:
                assert len(dashboard.traces.get_children()) == 3
                assert len(dashboard.chain.get_children()) == 1
                assert dashboard.chain.exists(dashboard.chain.get_children()[0])
            root.after(300, dashboard.close)
        root.mainloop()
        if args.smoke_test:
            print("Desktop dashboard rendered successfully; no model client created.")


if __name__ == "__main__":
    main()
