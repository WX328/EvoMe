import json
from types import SimpleNamespace

from app.agent import AgentLoop
from app.llm import DeepSeekClient
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.permissions import PermissionPolicy
from tests.test_planning import make_agent


def raw_client(responses):
    queue = list(responses)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        content, finish_reason = queue.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason,
        )])

    client = object.__new__(DeepSeekClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._record_usage = lambda **kwargs: None
    return client, requests, queue


def test_malformed_prose_recovers_through_short_control_and_plain_answer(tmp_path, capsys):
    (tmp_path / "module.py").write_text("ACTUAL_FILE = True\n", encoding="utf-8")
    prose = '第 1 步已完成。\n文件是 "module.py"。\n路径含反斜杠：app\\tools。'
    client, requests, queue = raw_client([
        (json.dumps({"action": "tool", "tool_name": "glob", "arguments": {"pattern": "*.py"}}), "stop"),
        ('{"action":"final","answer":"目录是 "module.py"，已完成"}', "stop"),
        ('{"action":"final","task_complete":true}', "stop"),
        (prose, "stop"),
    ])
    agent = AgentLoop(
        llm=client,
        tool_registry=build_default_tool_registry(skill_registry=SkillRegistry(), workspace_root=tmp_path),
        permission_policy=PermissionPolicy(), planning_enabled=True,
        plan_step_max_steps=2, plan_require_file_evidence=True,
    )
    plan = agent.planner.create("分析目录", ["列出目录"])
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert plan.steps[0].note == prose
    assert len([s for s in plan.steps[0].tool_steps if s.tool_call]) == 1
    assert any(s.action == "invalid_json_retry" for s in plan.steps[0].tool_steps)
    assert len(requests) == 4 and queue == []
    assert "response_format" not in requests[-1]
    assert "[TOOL_RESULT]" in str(requests[-1]["messages"])
    assert "必须只返回 JSON" not in requests[-1]["messages"][0]["content"]
    logged = capsys.readouterr().out
    assert logged.count("JSON 格式异常") == 1
    assert "\\n⚠" not in logged
    assert prose not in logged


def test_persistent_malformed_control_fails_as_protocol_error_without_unbounded_retry():
    malformed = {"action": "final", "answer": "已完成", "_json_repaired": True}
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "glob", "arguments": {"pattern": "app/tools/*"}},
        malformed, malformed,
    ], step_max_steps=2, auto_replan=True)
    plan = agent.planner.create("列出 app/tools", ["列出目录"])
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert "protocol_error" in plan.error
    assert "max_steps" not in plan.error
    assert plan.completed_count == 0 and plan.replans == 0
    assert len(calls) == 1 and len(agent.llm.requests) == 3
    assert agent.llm.chat_calls == 0


def test_final_only_turn_also_generates_plain_answer():
    agent, calls = make_agent([
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "a.py"}},
        {"action": "final", "task_complete": True},
    ], step_max_steps=1)
    agent.llm.chat = lambda messages, **kwargs: "完整分析\n包含任意未转义引号：\"abc\""
    plan = agent.planner.create("读取源码", ["读取 a.py"])
    agent.planner.run(plan.id)
    assert plan.status == "completed" and len(calls) == 1
    assert "完整分析" in plan.steps[0].note


def test_truncated_plain_answer_never_completes_task():
    client, requests, _ = raw_client([
        ('{"action":"final","task_complete":true}', "stop"),
        ("cut short", "length"),
    ])
    agent, _ = make_agent([])
    agent.llm = client
    result = agent.run(user_input="解释概念")
    assert result.termination_reason == "output_limit"
    assert "cut short" in result.answer
    assert len(requests) == 2


def test_plan_show_is_real_read_only_tool_using_runtime_plan_id():
    agent, calls = make_agent([], permission_policy=PermissionPolicy())
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析 app/tools，然后 /plan-show 查看记录", [
        "列出 app/tools", "使用 /plan-show 查看工具调用记录",
    ])
    agent.llm.decisions = [
        {"action": "tool", "tool_name": "glob", "arguments": {"pattern": "app/tools/*"}},
        {"action": "final", "answer": "目录已列出"},
        {"action": "tool", "tool_name": "plan_show", "arguments": {"plan_id": plan.id}},
        {"action": "final", "answer": "已检查实际记录"},
    ]
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    inspection = plan.steps[1].tool_steps[0]
    assert inspection.tool_call.name == "plan_show" and inspection.error == ""
    assert inspection.tool_output["plan_id"] == plan.id
    assert "glob app/tools/*" in inspection.tool_output["report"]
    assert plan.id in str(agent.llm.requests[2])
    assert "glob app/tools/*" in agent.planner.describe(plan)
    assert len(calls) == 1


def test_inspecting_plan_does_not_replace_file_reading_evidence():
    agent, _ = make_agent([])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("读取文件并查看记录", ["读取源码并使用 /plan-show 查看记录"])
    agent.llm.decisions = [
        {"action": "tool", "tool_name": "plan_show", "arguments": {"plan_id": plan.id}},
        {"action": "final", "answer": "读完了"},
    ]
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert "missing_file_evidence" in plan.error


def test_wrong_plan_id_cannot_produce_successful_inspection():
    agent, _ = make_agent([], permission_policy=PermissionPolicy())
    plan = agent.planner.create("查看记录", ["使用 /plan-show 查看记录"])
    agent.llm.decisions = [
        {"action": "tool", "tool_name": "plan_show", "arguments": {"plan_id": "<实际计划ID>"}},
        {"action": "final", "answer": "没有找到计划"},
    ]
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert "unknown plan id" in plan.steps[0].tool_steps[0].error


def test_no_progress_pause_is_distinguished_from_budget_exhaustion():
    agent, _ = make_agent([
        {"action": "tool", "tool_name": "glob", "arguments": {"pattern": "*"}},
        {"action": "final", "task_complete": False, "answer": "还未完成"},
        {"action": "final", "task_complete": False, "answer": "还未完成"},
    ], step_max_steps=2)
    plan = agent.planner.create("分析", ["列出目录"])
    agent.planner.run(plan.id)
    assert plan.status == "paused"
    assert "没有产生新的工具调用" in agent.planner.describe(plan)
    assert "自动续跑预算已用完" not in agent.planner.describe(plan)


def test_current_step_scope_is_system_context_not_only_global_user_goal():
    agent, _ = make_agent([{"action": "final", "answer": "目录已列出"}])
    plan = agent.planner.create("分析目录，然后 /plan-show 查看记录", ["列出 app/tools 目录"])
    agent.planner.run(plan.id)
    contexts = [message["content"] for message in agent.llm.requests[0] if message["role"] == "system"]
    assert any("本次唯一任务是：列出 app/tools 目录" in text for text in contexts)
    assert any("成功的 glob 结果足够" in text for text in contexts)


def test_inspection_alone_cannot_ground_an_architecture_analysis():
    agent, _ = make_agent([])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析 app/agent 并查看记录", ["分析 app/agent 并 /plan-show"])
    agent.llm.decisions = [
        {"action": "tool", "tool_name": "plan_show", "arguments": {"plan_id": plan.id}},
        {"action": "final", "answer": "分析完成"},
    ]
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert "missing_file_evidence" in plan.error


def test_summary_and_plan_inspection_reuse_completed_source_evidence():
    agent, calls = make_agent([])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析 app/agent 并查看记录", [
        "读取 app/agent/loop.py",
        "综合分析 app/agent 的职责，并调用 plan_show 查看本次计划每一步真实的工具调用记录。",
    ])
    agent.llm.decisions = [
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "app/agent/loop.py"}},
        {"action": "final", "answer": "已读取源码"},
        {"action": "tool", "tool_name": "plan_show", "arguments": {"plan_id": plan.id}},
        {"action": "final", "answer": "根据已读源码综合分析，并核对真实记录"},
    ]
    agent.planner.run(plan.id)
    assert plan.status == "completed"
    assert len(calls) == 1
    assert "read_file app/agent/loop.py" in plan.steps[1].tool_steps[0].tool_output["report"]


def test_combined_summary_requiring_new_read_cannot_reuse_old_evidence():
    agent, _ = make_agent([])
    agent.planner.require_file_evidence = True
    plan = agent.planner.create("分析 app/agent 并查看记录", [
        "读取 app/agent/loop.py",
        "读取 app/agent/schema.py，综合分析并调用 plan_show 查看记录",
    ])
    agent.llm.decisions = [
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "app/agent/loop.py"}},
        {"action": "final", "answer": "已读取 loop"},
        {"action": "tool", "tool_name": "plan_show", "arguments": {"plan_id": plan.id}},
        {"action": "final", "answer": "已读取 schema 并综合分析"},
    ]
    agent.planner.run(plan.id)
    assert plan.status == "failed"
    assert "missing_file_evidence" in plan.error
