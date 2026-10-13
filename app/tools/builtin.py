from __future__ import annotations

from pathlib import Path

from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.filesystem import FileSystemTools
from app.tools.registry import ToolRegistry
from app.tools.shell import BashTool
from app.tools.skill_adapter import build_tool_registry_from_skills


def build_default_tool_registry(
    *,
    skill_registry,
    workspace_root: str | Path,
    enable_bash: bool = False,
    bash_timeout_seconds: int = 20,
    max_output_chars: int = 12000,
) -> ToolRegistry:
    registry = build_tool_registry_from_skills(skill_registry)
    fs = FileSystemTools(
        workspace_root=workspace_root,
        max_output_chars=max_output_chars,
    )

    definitions = [
        ToolDefinition(
            name="read_file",
            description=(
                "读取 EvoMe 工作区内的 UTF-8 文本文件，可指定行范围。"
                "truncated=true 时内容不完整；有 next_start_line 时从该行继续读取。"
                "line_truncated=true 表示单行超过输出上限，不能当作完整内容。"
            ),
            input_schema={
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": ["integer", "null"], "default": None},
            },
        ),
        ToolDefinition(
            name="write_file",
            description="在 EvoMe 工作区内创建或写入文本文件；append=true 时追加。",
            input_schema={
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
        ),
        ToolDefinition(
            name="edit_file",
            description="对工作区文本文件执行精确字符串替换，并验证匹配次数。",
            input_schema={
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "expected_replacements": {"type": "integer", "default": 1},
            },
        ),
        ToolDefinition(
            name="glob",
            description="按 glob 模式查找 EvoMe 工作区内的文件和目录。",
            input_schema={
                "pattern": {"type": "string"},
                "max_results": {"type": "integer", "default": 100},
            },
        ),
    ]
    for definition in definitions:
        registry.register_tool(definition)

    registry.register_implementation(
        ToolImplementation(
            name="workspace_read_file_v1",
            tool_name="read_file",
            handler=fs.read_file,
        )
    )
    registry.register_implementation(
        ToolImplementation(
            name="workspace_write_file_v1",
            tool_name="write_file",
            handler=fs.write_file,
        )
    )
    registry.register_implementation(
        ToolImplementation(
            name="workspace_edit_file_v1",
            tool_name="edit_file",
            handler=fs.edit_file,
        )
    )
    registry.register_implementation(
        ToolImplementation(
            name="workspace_glob_v1",
            tool_name="glob",
            handler=fs.glob,
        )
    )

    if enable_bash:
        bash = BashTool(
            workspace_root=workspace_root,
            timeout_seconds=bash_timeout_seconds,
            max_output_chars=max_output_chars,
        )
        registry.register_tool(
            ToolDefinition(
                name="bash",
                description=(
                    "在 EvoMe 工作区执行 shell 命令并返回 stdout/stderr。"
                    "S2 默认关闭；S3 将补充权限确认。"
                ),
                input_schema={"command": {"type": "string"}},
            )
        )
        registry.register_implementation(
            ToolImplementation(
                name="workspace_bash_v1",
                tool_name="bash",
                handler=bash.run,
            )
        )

    return registry
