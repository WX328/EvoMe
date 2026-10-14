"""S14 MCP Plugin: offline protocol, namespace, approval and isolation tests.

Pytest is intentionally run by the EvoMe project owner after patch apply.
The test MCP process is a local fake server with no network/file mutation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from app.agent.loop import AgentLoop
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookManager
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.executor import ToolExecutor
from app.tools.mcp_plugin import (
    MCPConfig, MCPPluginManager, MCPServerSpec, MCPStdioClient,
    _minimal_process_env,
)
from app.tools.permissions import ApprovalStore, PermissionPolicy
from app.tools.registry import ToolRegistry

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "s14_stdio_fixture.py"


def config(tmp_path, mode="normal", servers=None):
    data = servers or {
        "demo": {"command": sys.executable, "args": ["-u", str(FIXTURE), mode]},
    }
    (tmp_path / ".mcp_servers.json").write_text(
        json.dumps({"servers": data}), encoding="utf-8",
    )


def manager(tmp_path, mode="normal", *, registry=None, approval_store=None, hooks=None, timeout=3):
    config(tmp_path, mode=mode)
    registry = registry if registry is not None else ToolRegistry()
    approvals = approval_store or ApprovalStore()
    def revoke(names):
        for pending in list(approvals.list_pending()):
            if pending.tool_name in names:
                approvals.pop(pending.id)
    plugin = MCPPluginManager(
        workspace_root=tmp_path,
        registry=registry,
        approval_revoke=revoke,
        timeout=timeout,
    )
    executor = ToolExecutor(
        registry=registry, approval_store=approvals,
        permission_policy=PermissionPolicy(workspace_root=tmp_path),
        hook_manager=hooks,
    )
    return plugin, registry, executor, approvals


def test_empty_config_is_discovery_only_no_process_spawn(tmp_path):
    registry = ToolRegistry()
    plugin = MCPPluginManager(workspace_root=tmp_path, registry=registry)
    try:
        assert plugin.servers() == []
        assert plugin.connected_tools() == []
        assert list(tmp_path.iterdir()) == []
    finally:
        plugin.close()


def test_stdio_initialize_discover_and_namespaced_registry(tmp_path):
    plugin, registry, _, _ = manager(tmp_path)
    try:
        before = list(registry.list_tools())
        servers = plugin.servers()
        assert servers == [{"name": "demo", "connected": False, "tool_count": 0}]
        assert len(registry.list_tools()) == len(before)
        names = plugin.connect("demo")
        assert names == ["mcp__demo__echo"]
        assert registry.get_tool(names[0]).input_schema["text"]["type"] == "string"
        assert "approval required" in registry.get_tool(names[0]).description
        assert plugin.servers()[0]["connected"]
        assert len(plugin.connected_tools()) == 1
    finally:
        plugin.close()
    assert registry.list_tools() == []


def test_external_tools_require_s03_approval_and_after_approve_s04_hooks(tmp_path):
    hooks = HookManager()
    plugin, registry, executor, approvals = manager(tmp_path, hooks=hooks)
    try:
        plugin.connect("demo")
        pending = executor.execute("mcp__demo__echo", {"text": "hello MCP"})
        assert pending.status == "approval_required"
        assert pending.risk == "high"
        assert len(approvals.list_pending()) == 1
        resolved = executor.approve(pending.approval_id)
        assert resolved.success and resolved.status == "executed"
        assert resolved.output["content"][0]["text"] == "hello MCP"
        assert approvals.list_pending() == []
        events = [(event.event, event.tool_name) for event in hooks.recent()]
        assert ("before_tool", "mcp__demo__echo") in events
        assert ("after_tool", "mcp__demo__echo") in events
    finally:
        plugin.close()


def test_rejected_external_approval_does_not_invoke_mcp(tmp_path):
    plugin, registry, executor, approvals = manager(tmp_path)
    try:
        plugin.connect("demo")
        pending = executor.execute("mcp__demo__echo", {"text": "do not run"})
        assert pending.status == "approval_required"
        assert executor.deny(pending.approval_id)
        assert approvals.list_pending() == []
        assert executor.approve(pending.approval_id).status == "missing"
    finally:
        plugin.close()


def test_disconnect_removes_tools_and_pending_approvals(tmp_path):
    plugin, registry, executor, approvals = manager(tmp_path)
    try:
        plugin.connect("demo")
        pending = executor.execute("mcp__demo__echo", {"text": "old"})
        plugin.disconnect("demo")
        assert registry.list_tools() == []
        assert approvals.list_pending() == []
        assert executor.approve(pending.approval_id).status == "missing"
        assert plugin.servers()[0]["connected"] is False
        with pytest.raises(LookupError):
            registry.resolve("mcp__demo__echo")
    finally:
        plugin.close()


def test_reconnecting_stale_handler_cannot_dispatch(tmp_path):
    plugin, registry, executor, _ = manager(tmp_path)
    try:
        plugin.connect("demo")
        old_handler = registry.resolve("mcp__demo__echo")
        plugin.disconnect("demo")
        plugin.connect("demo")
        with pytest.raises(PermissionError, match="stale"):
            old_handler.execute(text="must not run")
        pending = executor.execute("mcp__demo__echo", {"text": "new"})
        result = executor.approve(pending.approval_id)
        assert result.success
        assert result.output["content"][0]["text"] == "new"
    finally:
        plugin.close()


def test_two_servers_namespace_isolation_and_tool_pool_assembly(tmp_path):
    servers = {
        "alpha": {"command": sys.executable, "args": ["-u", str(FIXTURE), "normal"]},
        "beta": {"command": sys.executable, "args": ["-u", str(FIXTURE), "normal"]},
    }
    config(tmp_path, servers=servers)
    registry = ToolRegistry()
    plugin = MCPPluginManager(workspace_root=tmp_path, registry=registry, timeout=3)
    try:
        plugin.connect("alpha")
        plugin.connect("beta")
        assert {t.name for t in registry.list_tools()} == {
            "mcp__alpha__echo", "mcp__beta__echo",
        }
        plugin.disconnect("alpha")
        assert {t.name for t in registry.list_tools()} == {"mcp__beta__echo"}
    finally:
        plugin.close()


def test_discovered_pagination_assembles_both_tools(tmp_path):
    plugin, registry, executor, _ = manager(tmp_path, mode="pagination")
    try:
        names = plugin.connect("demo")
        assert names == ["mcp__demo__echo", "mcp__demo__uppercase"]
        pending = executor.execute("mcp__demo__uppercase", {"text": "abc"})
        result = executor.approve(pending.approval_id)
        assert result.success
        assert result.output["content"][0]["text"] == "ABC"
    finally:
        plugin.close()


def test_server_reverse_sampling_requests_are_rejected(tmp_path):
    plugin, registry, executor, _ = manager(tmp_path, mode="reverse")
    try:
        plugin.connect("demo")
        pending = executor.execute("mcp__demo__echo", {"text": "safe"})
        result = executor.approve(pending.approval_id)
        assert result.success
        assert result.output["content"][0]["text"] == "safe"
    finally:
        plugin.close()


def test_malformed_discovery_and_duplicate_names_never_pollute_registry(tmp_path):
    for mode in ("bad_metadata", "duplicate", "bogus_jsonrpc"):
        subdir = tmp_path / mode
        subdir.mkdir()
        plugin, registry, _, _ = manager(subdir, mode=mode, timeout=0.5)
        try:
            with pytest.raises((ValueError, ConnectionError, TimeoutError)):
                plugin.connect("demo")
            assert registry.list_tools() == []
            assert plugin.connected_tools() == []
        finally:
            plugin.close()


def test_collision_does_not_remove_existing_builtin_tool(tmp_path):
    registry = ToolRegistry()
    fullname = "mcp__demo__echo"
    registry.register_tool(ToolDefinition(
        name=fullname, description="existing reserved entry",
    ))
    registry.register_implementation(ToolImplementation(
        name="existing", tool_name=fullname, handler=lambda **kwargs: "safe",
    ))
    plugin, _, _, _ = manager(tmp_path, registry=registry)
    try:
        with pytest.raises(ValueError, match="conflicts"):
            plugin.connect("demo")
        assert registry.resolve(fullname).execute() == "safe"
        assert registry.get_tool(fullname).description == "existing reserved entry"
    finally:
        plugin.close()


def test_tool_reported_error_does_not_produce_false_success(tmp_path):
    plugin, _, executor, _ = manager(tmp_path, mode="tool_error")
    try:
        plugin.connect("demo")
        pending = executor.execute("mcp__demo__echo", {"text": "x"})
        result = executor.approve(pending.approval_id)
        assert not result.success
        assert result.status == "failed"
        assert "reported error" in result.error
    finally:
        plugin.close()


def test_server_crash_and_timeout_fail_closed(tmp_path):
    for mode in ("crash", "hang"):
        sub = tmp_path / mode
        sub.mkdir()
        plugin, registry, executor, _ = manager(
            sub, mode=mode, timeout=0.4,
        )
        try:
            if mode == "hang":
                with pytest.raises((TimeoutError, ConnectionError)):
                    plugin.connect("demo")
                assert registry.list_tools() == []
            else:
                plugin.connect("demo")
                pending = executor.execute("mcp__demo__echo", {"text": "x"})
                output = executor.approve(pending.approval_id)
                assert not output.success
                assert output.status == "failed"
        finally:
            plugin.close()


def test_invalid_config_unknown_server_and_no_shell_launch(tmp_path):
    cases = [
        {"servers": {"A": {"command": "python", "args": []}}},
        {"servers": {"alpha": {"command": "python", "args": "hello"}}},
        {"servers": {"alpha": {"command": "python", "args": [],
                               "env": {"SECRET": "no"}}}},
        {"servers": {"alpha": {"command": "python", "args": ["x\x00"]}}},
    ]
    for item in cases:
        (tmp_path / ".mcp_servers.json").write_text(
            json.dumps(item), encoding="utf-8",
        )
        with pytest.raises(ValueError):
            MCPConfig(tmp_path).load()
    config(tmp_path)
    plugin = MCPPluginManager(workspace_root=tmp_path, registry=ToolRegistry())
    try:
        with pytest.raises(KeyError):
            plugin.connect("unknown")
        with pytest.raises(ValueError):
            plugin.connect("../demo")
    finally:
        plugin.close()


def test_plugin_child_does_not_inherit_secret_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "TEST_MUST_NOT_LEAK")
    monkeypatch.setenv("OPENAI_API_KEY", "TEST_MUST_NOT_LEAK")
    monkeypatch.setenv("SOME_BEARER_TOKEN", "TEST_MUST_NOT_LEAK")
    environ = _minimal_process_env()
    assert not any("TEST_MUST_NOT_LEAK" in v for v in environ.values())
    assert "PATH" in {key.upper() for key in environ}


class ScriptedAgentLLM:
    def __init__(self):
        self.n = 0

    def json_chat(self, messages, **kwargs):
        self.n += 1
        if self.n == 1:
            return {
                "action": "tool", "tool_name": "mcp__demo__echo",
                "arguments": {"text": "from agent"},
            }
        return {"action": "final", "task_complete": True}

    def chat(self, messages, **kwargs):
        return "A short summary."


def test_agent_catalog_includes_mcp_and_approval_checkpoint(tmp_path):
    plugin, registry, executor, _ = manager(tmp_path)
    try:
        plugin.connect("demo")
        loop = AgentLoop(
            llm=ScriptedAgentLLM(),
            tool_registry=registry,
            permission_policy=PermissionPolicy(workspace_root=tmp_path),
            max_steps=3,
        )
        assert "mcp__demo__echo" in loop._tool_catalog()
        decision = loop.executor.execute("mcp__demo__echo", {"text": "from agent"})
        assert decision.status == "approval_required"
    finally:
        plugin.close()


def test_cli_commands_are_operational_and_never_learning_samples():
    for command in (
        "mcp-servers",
        "mcp-connect demo",
        "mcp-tools",
        'mcp-call mcp__demo__echo {"text":"test"}',
        "mcp-disconnect demo",
    ):
        assert normalize_cli_input(command) == "/" + command
        assert looks_like_operational_input(command)
