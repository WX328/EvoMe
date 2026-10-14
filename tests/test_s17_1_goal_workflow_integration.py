"""S17.1 offline end-to-end tests: Goal -> S03 -> S16 -> inspect -> Goal.

No API keys, network, pytest run by assistant, or access to user's .env.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.agent.goal_loop import GoalController, GoalEvaluation
from app.agent.loop import AgentLoop
from app.agent.workflow_runtime import WorkflowRuntime
from app.agent.workflow_store import WorkflowStore
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.permissions import PermissionPolicy
from app.tools.registry import ToolRegistry


PATH = "app/agent/schema.py"


class ResearchRunner:
    def __init__(self):
        self.calls = []
    def run(self, prompt, schema, label):
        self.calls.append(label)
        return {
            "summary": f"Source-backed review for {label}.",
            "paths": [PATH], "verified": True,
            "summary_truncated": False, "source_read_truncated": False,
        }


def fixtures(tmp_path):
    runner = ResearchRunner()
    workflow = WorkflowRuntime(
        store=WorkflowStore(tmp_path), workspace_root=tmp_path,
        runner_factory=lambda: runner,
    )
    registry = ToolRegistry()
    registry.register_tool(ToolDefinition(
        name="Workflow", description="S16 host-script, requires approval",
        input_schema={
            "name": {"type": "string"},
            "args": {"type": "object"},
            "resume_from_run_id": {"type": "string"},
        },
    ))
    registry.register_implementation(ToolImplementation(
        name="s16_launch_fixture", tool_name="Workflow",
        handler=workflow.run_tool,
    ))
    registry.register_tool(ToolDefinition(
        name="workflow_get", description="Read-only S16 status/result",
        input_schema={"run_id": {"type": "string"}},
    ))
    registry.register_implementation(ToolImplementation(
        name="s16_inspect_fixture", tool_name="workflow_get",
        handler=workflow.inspect_for_agent,
    ))
    return workflow, registry, runner


def test_completed_workflow_safe_read_only_status_and_review_quality(tmp_path):
    workflow, registry, runner = fixtures(tmp_path)
    try:
        run_id = workflow.launch("review-files", {"paths": [PATH]})["run_id"]
        workflow.wait(run_id, timeout=5)
        details = workflow.inspect_for_agent(run_id)
        assert details["completed"] is True
        assert details["status"] == "completed"
        assert details["result_status"] == "reviewed"
        assert details["result_count"] == 1
        assert details["journal_entries"] == 2
        assert details["files"][0]["path"] == PATH
        assert details["files"][0]["quality_status"] == "unverified"
        assert "audit_excerpt" in details["files"][0]
        assert "verification_excerpt" in details["files"][0]
        assert "journal" not in details
        assert "progress" not in details
        assert "result" not in details  # only curated fields
    finally:
        workflow.shutdown()


def test_queued_workflow_not_mistaken_for_completed(tmp_path):
    workflow, registry, runner = fixtures(tmp_path)
    try:
        row = workflow.store.reserve("review-files", {"paths": [PATH]})
        details = workflow.inspect_for_agent(row["id"])
        assert details["completed"] is False
        assert details["status"] == "queued"
        assert "files" not in details
        workflow.store.unlock(row["id"])
    finally:
        workflow.shutdown()


def test_failed_workflow_cannot_claim_verified_findings(tmp_path):
    class FailedRunner:
        def run(self, prompt, schema, label):
            raise RuntimeError("PRIVATE_CANARY_VALUE_SHOULD_NOT_LEAK")
    workflow = WorkflowRuntime(
        store=WorkflowStore(tmp_path), workspace_root=tmp_path,
        runner_factory=FailedRunner,
    )
    try:
        run_id = workflow.launch("review-files", {"paths": [PATH]})["run_id"]
        workflow.wait(run_id, timeout=5)
        details = workflow.inspect_for_agent(run_id)
        assert details["status"] == "failed"
        assert details["completed"] is False
        assert details["error_type"] == "RuntimeError"
        assert "files" not in details
        assert "PRIVATE_CANARY" not in str(details)
    finally:
        workflow.shutdown()


@pytest.mark.parametrize("bad_id", ["../private", "wf_123", "not-an-id", "wf_AAAAAAAAAAAA"])
def test_workflow_get_rejects_invalid_run_ids(tmp_path, bad_id):
    workflow, _, _ = fixtures(tmp_path)
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            workflow.inspect_for_agent(bad_id)
    finally:
        workflow.shutdown()


def test_only_read_only_workflow_get_is_auto_allowed():
    policy = PermissionPolicy()
    lookup = policy.evaluate("workflow_get", {"run_id": "wf_123456789abc"})
    launch = policy.evaluate(
        "Workflow", {"name": "review-files", "args": {"paths": [PATH]}}
    )
    assert lookup.action == "allow" and lookup.risk == "low"
    assert launch.action == "ask" and launch.risk == "high"
    assert policy.evaluate("write_file", {"path": "unsafe.txt", "content": "x"}).action == "ask"


def test_goal_approval_note_contains_only_valid_workflow_id_not_raw_output():
    class NoopEvaluator:
        def evaluate(self, *_args):
            return GoalEvaluation(False, "not confirmed")
    goal = GoalController(NoopEvaluator())
    goal.set("wait for Workflow")
    goal.waiting_approval_id = "approvalX"
    goal.note_approval(
        "approvalX", success=True, status="executed", tool_name="Workflow",
        output={
            "run_id": "wf_123456789abc",
            "status": "queued", "payload": "PRIVATE_TOOL_CANARY",
        },
    )
    assert goal.waiting_approval_id == ""
    assert goal._approval_events
    assert "wf_123456789abc" in goal._approval_events[0]
    assert "workflow_get" in goal._approval_events[0]
    assert "PRIVATE_TOOL_CANARY" not in goal._approval_events[0]
    assert "NOT completion" in goal._approval_events[0]


def test_other_approvals_cannot_forge_workflow_id_into_goal():
    class NoopEvaluator:
        def evaluate(self, *_args):
            return GoalEvaluation(False, "not complete")
    goal = GoalController(NoopEvaluator())
    goal.set("read source")
    goal.waiting_approval_id = "a123"
    goal.note_approval(
        "a123", success=True, status="executed", tool_name="write_file",
        output={"run_id": "wf_123456789abc", "secret": "CANARY"},
    )
    assert "wf_123456789abc" not in goal._approval_events[0]
    assert "CANARY" not in goal._approval_events[0]


def test_workflow_status_poll_bounded_per_goal_query_and_reset_on_manual_continue():
    class NoopEvaluator:
        def evaluate(self, *_args):
            return GoalEvaluation(False, "not complete")
    goal = GoalController(NoopEvaluator())
    goal.set("workflow")
    assert [goal.allow_workflow_status_poll() for _ in range(4)] == [
        True, True, True, False,
    ]
    goal.begin_query()
    assert goal.allow_workflow_status_poll() is True
    goal.clear()
    assert goal.allow_workflow_status_poll() is False


class ScriptedGoalWorker:
    def __init__(self):
        self.decisions = []
        self.transcripts = []
        self.run_id = ""
    def json_chat(self, messages, **kwargs):
        self.transcripts.append([dict(item) for item in messages])
        if not self.decisions:
            return {
                "action": "tool", "tool_name": "Workflow",
                "arguments": {
                    "name": "review-files",
                    "args": {"paths": [PATH]},
                },
            }
        nxt = self.decisions.pop(0)
        if nxt == "inspect":
            assert self.run_id
            return {
                "action": "tool", "tool_name": "workflow_get",
                "arguments": {"run_id": self.run_id},
            }
        return {
            "action": "final", "task_complete": True,
            "answer": "Verified script execution status with evidence.",
        }
    def chat(self, messages, **kwargs):
        return "S16 Workflow is completed with source-backed review output."


class EvidenceGoalEvaluator:
    def __init__(self):
        self.calls = []
    def evaluate(self, condition, messages):
        all_text = "\n".join(row["content"] for row in messages)
        self.calls.append(all_text)
        # Script approval alone is insufficient. Need read-only durable result.
        seen_result = "workflow_get" in all_text and "journal_entries" in all_text
        completed = '"completed": true' in all_text.lower() or '"completed":true' in all_text.lower()
        if seen_result and completed:
            return GoalEvaluation(True, "WorkflowStore completed; structured audit evidence received")
        return GoalEvaluation(False, "missing completed WorkflowStore output")


def test_real_agent_loop_goal_workflow_approval_inspection_completion(tmp_path):
    workflow, registry, runner = fixtures(tmp_path)
    worker = ScriptedGoalWorker()
    evaluator = EvidenceGoalEvaluator()
    goal = GoalController(
        evaluator,
        background_running=lambda: bool(
            [row for row in workflow.current_runs()
             if row["status"] in {"queued", "running"}]
        ),
        block_cap=2,
    )
    agent = AgentLoop(
        llm=worker, tool_registry=registry,
        permission_policy=PermissionPolicy(workspace_root=tmp_path),
        max_steps=5,
    )
    agent.goal_controller = goal
    try:
        goal.set("运行只读 review-files 工作流，检查真正完成且结果带路径证据")
        first = agent.run(user_input=goal.condition)
        assert first.termination_reason == "approval_required"
        assert evaluator.calls == []
        assert workflow.list() == []  # still awaiting S03 operator
        assert first.approval_id

        approval = agent.executor.approve(first.approval_id)
        assert approval.success and approval.status == "executed"
        run_id = approval.output["run_id"]
        worker.run_id = run_id
        goal.note_approval(
            first.approval_id, success=True, status=approval.status,
            tool_name=approval.tool_name, output=approval.output,
        )
        assert "run_id=" + run_id in goal._approval_events[0]
        workflow.wait(run_id, timeout=5)
        assert workflow.show(run_id)["status"] == "completed"
        assert len(runner.calls) == 2

        worker.decisions = ["inspect", "final"]
        goal.begin_query()
        second = agent.run(user_input=goal.condition)
        assert second.termination_reason == "goal_achieved"
        assert goal.status()["active"] is False
        assert len(evaluator.calls) == 1
        assert "journal_entries" in evaluator.calls[0]
        assert len(runner.calls) == 2  # never restarted the script
    finally:
        workflow.shutdown()


def test_same_workflow_get_repeat_allowed_for_active_goal_without_reexecuting_script(tmp_path):
    workflow, registry, runner = fixtures(tmp_path)
    worker = ScriptedGoalWorker()
    evaluator = EvidenceGoalEvaluator()
    goal = GoalController(evaluator, background_running=lambda: False)
    agent = AgentLoop(
        llm=worker, tool_registry=registry,
        permission_policy=PermissionPolicy(workspace_root=tmp_path),
        max_steps=5,
    )
    agent.goal_controller = goal
    try:
        run_id = workflow.launch("review-files", {"paths": [PATH]})["run_id"]
        workflow.wait(run_id, timeout=5)
        worker.run_id = run_id
        # Two identical read-only lookups must not hit the S01 duplicate gate.
        worker.decisions = ["inspect", "inspect", "final"]
        goal.set("Check persisted Workflow status twice, then finish")
        answer = agent.run(user_input=goal.condition)
        assert answer.termination_reason == "goal_achieved"
        lookups = [
            step for step in answer.steps
            if step.tool_call and step.tool_call.name == "workflow_get"
        ]
        assert len(lookups) == 2
        assert all(not step.error for step in lookups)
        assert len(runner.calls) == 2
    finally:
        workflow.shutdown()


def test_regular_chat_duplicate_workflow_get_still_blocked(tmp_path):
    workflow, registry, runner = fixtures(tmp_path)
    worker = ScriptedGoalWorker()
    agent = AgentLoop(
        llm=worker, tool_registry=registry,
        permission_policy=PermissionPolicy(workspace_root=tmp_path),
        max_steps=3,
    )
    try:
        run_id = workflow.launch("review-files", {"paths": [PATH]})["run_id"]
        workflow.wait(run_id, timeout=5)
        worker.run_id = run_id
        worker.decisions = ["inspect", "inspect", "final"]
        result = agent.run(user_input="just chat, no active goal")
        lookups = [
            step for step in result.steps
            if step.tool_call and step.tool_call.name == "workflow_get"
        ]
        assert len(lookups) == 2
        assert "duplicate tool call blocked" in lookups[1].error
    finally:
        workflow.shutdown()
