from pathlib import Path

import pytest

from app.tools.builtin import build_default_tool_registry
from app.tools.executor import ToolExecutor
from app.skills.registry import SkillRegistry


def build_registry(tmp_path: Path, *, enable_bash: bool = False):
    return build_default_tool_registry(
        skill_registry=SkillRegistry(),
        workspace_root=tmp_path,
        enable_bash=enable_bash,
        bash_timeout_seconds=3,
        max_output_chars=4000,
    )


def test_read_write_edit_and_glob(tmp_path):
    executor = ToolExecutor(build_registry(tmp_path))

    written = executor.execute(
        "write_file",
        {"path": "demo/a.txt", "content": "hello\nworld\n"},
    )
    assert written.success

    read = executor.execute("read_file", {"path": "demo/a.txt"})
    assert read.success
    assert "1: hello" in read.output["content"]

    edited = executor.execute(
        "edit_file",
        {
            "path": "demo/a.txt",
            "old_string": "world",
            "new_string": "EvoMe",
            "expected_replacements": 1,
        },
    )
    assert edited.success

    matches = executor.execute("glob", {"pattern": "demo/*.txt"})
    assert matches.success
    assert matches.output["matches"][0]["path"].endswith("demo\\a.txt") or matches.output["matches"][0]["path"].endswith("demo/a.txt")


def test_workspace_guard_blocks_parent_escape(tmp_path):
    executor = ToolExecutor(build_registry(tmp_path))
    result = executor.execute("read_file", {"path": "../secret.txt"})
    assert not result.success
    assert "PermissionError" in result.error


def test_edit_file_requires_exact_match_count(tmp_path):
    target = tmp_path / "x.txt"
    target.write_text("a a", encoding="utf-8")
    executor = ToolExecutor(build_registry(tmp_path))

    result = executor.execute(
        "edit_file",
        {
            "path": "x.txt",
            "old_string": "a",
            "new_string": "b",
            "expected_replacements": 1,
        },
    )
    assert not result.success
    assert "expected 1 matches, found 2" in result.error


def test_bash_is_not_registered_by_default(tmp_path):
    registry = build_registry(tmp_path)
    names = {tool.name for tool in registry.list_tools()}
    assert "bash" not in names


def test_bash_can_be_explicitly_enabled(tmp_path):
    registry = build_registry(tmp_path, enable_bash=True)
    names = {tool.name for tool in registry.list_tools()}
    assert "bash" in names
