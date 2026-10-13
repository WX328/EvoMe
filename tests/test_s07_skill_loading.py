"""S07 Skill Loading: metadata catalog, on-demand body, and S03/S04 boundaries."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.agent import AgentLoop
from app.cli_commands import looks_like_operational_input, normalize_cli_input
from app.hooks import HookManager
from app.skills.loader import SkillLoader
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.permissions import PermissionPolicy


BODY_CANARY = "ONLY_LOADED_AFTER_EXPLICIT_TOOL_CALL_9871"


def add_skill(root: Path, name: str = "code-review", body: str = BODY_CANARY, *,
              description: str = "只读审查 Python 模块") -> Path:
    folder = root / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "SKILL.md"
    manifest.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return manifest


class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.text_requests = []

    def json_chat(self, messages, **kwargs):
        self.requests.append([dict(item) for item in messages])
        if not self.decisions:
            raise AssertionError("Unexpected extra decision")
        return self.decisions.pop(0)

    def chat(self, messages, **kwargs):
        self.text_requests.append([dict(item) for item in messages])
        return "基于按需加载的审查技能完成任务。"


def make_agent(tmp_path, decisions, *, hooks=None, subagent_enabled=False):
    registry = build_default_tool_registry(
        skill_registry=SkillRegistry(),
        workspace_root=tmp_path,
        enable_bash=False,
    )
    llm = ScriptedLLM(decisions)
    loader = SkillLoader(workspace_root=tmp_path)
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_policy=PermissionPolicy(),
        hook_manager=hooks,
        skill_loader=loader,
        subagent_enabled=subagent_enabled,
    )
    return agent, llm


def test_catalog_contains_only_metadata_without_body(tmp_path):
    add_skill(tmp_path)
    loader = SkillLoader(workspace_root=tmp_path)
    with patch.object(Path, "read_bytes", side_effect=AssertionError("body eagerly loaded")):
        catalog = loader.catalog()
    assert catalog == [
        {"name": "code-review", "description": "只读审查 Python 模块"}
    ]
    assert BODY_CANARY not in str(catalog)


def test_loading_one_skill_returns_body_and_relative_path(tmp_path):
    add_skill(tmp_path)
    output = SkillLoader(workspace_root=tmp_path).load_skill("code-review")
    assert output["name"] == "code-review"
    assert BODY_CANARY in output["content"]
    assert "name: code-review" not in output["content"]
    assert output["relative_path"] == "skills/code-review/SKILL.md"
    assert "Tool permissions" in output["notice"]


def test_agent_requests_load_skill_then_uses_tool_result(tmp_path):
    add_skill(tmp_path)
    agent, llm = make_agent(tmp_path, [
        {"action": "tool", "tool_name": "load_skill",
         "arguments": {"name": "code-review"}},
        {"action": "final", "task_complete": True},
    ])
    result = agent.run(user_input="先加载 code-review 指南再总结")
    assert result.termination_reason == "final_answer"
    assert [step.tool_call.name for step in result.steps if step.tool_call] == [
        "load_skill"
    ]
    assert BODY_CANARY not in str(llm.requests[0])
    assert "code-review" in str(llm.requests[0])
    assert BODY_CANARY in str(llm.requests[1])
    # ToolResult is conversation data, never a new privileged SYSTEM message.
    assert BODY_CANARY not in str([
        item for item in llm.requests[1] if item.get("role") == "system"
    ])
    assert BODY_CANARY in str(llm.text_requests[0])


def test_skill_does_not_load_automatically_without_tool_call(tmp_path):
    add_skill(tmp_path)
    agent, llm = make_agent(tmp_path, [
        {"action": "final", "task_complete": True},
    ])
    result = agent.run(user_input="解释一下这份 Skill")
    assert result.termination_reason == "final_answer"
    assert result.tool_calls == ()
    assert BODY_CANARY not in str(llm.requests)
    assert BODY_CANARY not in str(llm.text_requests)


def test_existing_executable_skill_registry_is_not_replaced(tmp_path):
    from app.skills.base import SkillDefinition, SkillImplementation
    registry = SkillRegistry()
    registry.register_skill(SkillDefinition(
        name="legacy_ability", description="Existing executable skill",
    ))
    registry.register_implementation(SkillImplementation(
        name="old_impl", skill_name="legacy_ability",
        kind="python_function", handler=lambda: 42,
    ))
    add_skill(tmp_path)
    loader = SkillLoader(workspace_root=tmp_path)
    assert registry.resolve("legacy_ability").execute() == 42
    assert [item.name for item in registry.list_skills()] == ["legacy_ability"]
    assert "legacy_ability" not in str(loader.catalog())


def test_subagent_cannot_load_instruction_skill(tmp_path):
    add_skill(tmp_path)
    agent, _ = make_agent(tmp_path, [], subagent_enabled=True)
    assert "load_skill" in [tool.name for tool in agent.tool_registry.list_tools()]
    assert "load_skill" not in [
        tool.name for tool in agent.subagent.registry.list_tools()
    ]


def test_load_skill_uses_s03_policy_and_s04_hooks(tmp_path):
    add_skill(tmp_path)
    hooks = HookManager()
    agent, _ = make_agent(tmp_path, [], hooks=hooks)
    result = agent.executor.execute("load_skill", {"name": "code-review"})
    assert result.success
    assert not agent.executor.approval_store.list_pending()
    events = [(entry.event, entry.tool_name) for entry in hooks.recent()]
    assert ("before_tool", "load_skill") in events
    assert ("after_tool", "load_skill") in events
    pending = agent.executor.execute(
        "write_file", {"path": "unsafe.txt", "content": "x"}
    )
    assert pending.status == "approval_required"
    assert not (tmp_path / "unsafe.txt").exists()


def test_invalid_skill_name_never_resolves_path(tmp_path):
    add_skill(tmp_path)
    loader = SkillLoader(workspace_root=tmp_path)
    for name in ("../.env", r"..\secret", "CODE-REVIEW", "", "code-review/../../etc"):
        with pytest.raises(ValueError):
            loader.load_skill(name)
    with pytest.raises(LookupError):
        loader.load_skill("uninstalled")


def test_path_escape_in_directory_config_rejected(tmp_path):
    with pytest.raises(ValueError):
        SkillLoader(workspace_root=tmp_path, directory="../elsewhere")


def test_symlinked_skill_is_not_discovered(tmp_path):
    real = tmp_path / "outside"
    real.mkdir()
    (real / "SKILL.md").write_text(
        "---\nname: leaked\ndescription: should not load\n---\nSECRET",
        encoding="utf-8",
    )
    target = tmp_path / "skills" / "leaked"
    target.parent.mkdir()
    try:
        target.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable on this platform")
    loader = SkillLoader(workspace_root=tmp_path)
    assert loader.catalog() == []
    with pytest.raises(LookupError):
        loader.load_skill("leaked")


def test_oversized_manifest_is_not_loaded(tmp_path):
    add_skill(tmp_path, body="X" * 100000)
    loader = SkillLoader(workspace_root=tmp_path)
    assert loader.catalog() == []
    with pytest.raises(LookupError):
        loader.load_skill("code-review")


def test_frontmatter_invalid_or_missing_description_is_ignored(tmp_path):
    file = add_skill(tmp_path)
    loader = SkillLoader(workspace_root=tmp_path)
    file.write_text("---\nname: code-review\n---\n# body", encoding="utf-8")
    assert loader.catalog() == []
    file.write_text("---not frontmatter\nname: code-review\n"
                    "description: no\n---\nbody", encoding="utf-8")
    assert loader.catalog() == []


def test_catalog_finds_new_skill_without_restarting(tmp_path):
    loader = SkillLoader(workspace_root=tmp_path)
    assert loader.catalog() == []
    add_skill(tmp_path)
    assert len(loader.catalog()) == 1
    assert loader.load_skill("code-review")["name"] == "code-review"


def test_s07_cli_commands_are_operational_not_learning_samples():
    assert normalize_cli_input("skill-docs") == "/skill-docs"
    assert normalize_cli_input("skill-load code-review") == "/skill-load code-review"
    assert looks_like_operational_input("/skill-load code-review")
    assert looks_like_operational_input("skill-docs")
