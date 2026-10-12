"""Built-in before_tool guard for sensitive paths inside EvoMe's workspace."""
from __future__ import annotations

from pathlib import Path

from app.hooks.manager import HookContext, HookDecision
from app.tools.workspace import WorkspaceGuard


def _sensitive_parts(parts: tuple[str, ...]) -> HookDecision:
    if any(part in {".git", ".venv", "evome_qdrant"} for part in parts):
        return HookDecision(False, "protected workspace directory")
    if parts:
        filename = parts[-1]
        if filename == ".env" or (filename.startswith(".env.") and filename != ".env.example"):
            return HookDecision(False, "protected environment configuration")
        if filename.endswith((".pem", ".key")) or filename == "evome.db":
            return HookDecision(False, "protected local secret or database")
    return HookDecision()


def protect_sensitive_paths(context: HookContext) -> HookDecision:
    """Reject known sensitive direct paths; mainly useful for unit testing."""
    if context.tool_name not in {"read_file", "write_file", "edit_file"}:
        return HookDecision()
    raw = context.arguments.get("path")
    if not isinstance(raw, str):
        return HookDecision()
    parts = tuple(part.lower() for part in raw.replace("\\", "/").split("/") if part)
    return _sensitive_parts(parts)


def build_sensitive_path_guard(workspace_root: str | Path):
    """Resolve aliases/symlinks before classifying protected paths."""
    workspace = WorkspaceGuard(workspace_root)

    def guard(context: HookContext) -> HookDecision:
        if context.tool_name not in {"read_file", "write_file", "edit_file"}:
            return HookDecision()
        raw = context.arguments.get("path")
        if not isinstance(raw, str):
            return HookDecision()
        # A workspace escape takes precedence over the sensitive-file check.
        # Resolve first so aliases and symlinks cannot bypass the guard.
        try:
            resolved = workspace.safe_path(raw)
        except (ValueError, PermissionError):
            return HookDecision(False, "path escapes workspace")
        direct = protect_sensitive_paths(context)
        if not direct.allow:
            return direct
        parts = tuple(part.lower() for part in resolved.relative_to(workspace.root).parts)
        return _sensitive_parts(parts)

    return guard
