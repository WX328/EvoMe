from __future__ import annotations

from pathlib import Path

from app.path_safety import resolved_path


def sensitive_path_reason(parts: tuple[str, ...], *, listing: bool = False) -> str:
    """Shared, case-insensitive policy for direct access and directory listings."""
    parts = tuple(part.lower() for part in parts)
    protected = {".git", ".venv", "evome_qdrant", ".traces", ".ssh", ".aws"}
    if listing:
        protected |= {".memory", ".tasks", ".schedules", ".team", ".workflows"}
    if any(part in protected for part in parts):
        return "protected workspace directory"
    for filename in parts:
        if filename == ".env" or (filename.startswith(".env.") and filename != ".env.example"):
            return "protected environment configuration"
        if filename.endswith((".pem", ".key", ".p12", ".pfx")) or filename in {
            "evome.db", "evome.db-wal", "evome.db-shm", "evome.db-journal",
            ".mcp_servers.json", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
            "credentials.json", "secrets.json",
        }:
            return "protected local secret or database"
    return ""


def safe_glob_pattern(raw: str) -> str:
    pattern = str(raw).replace("\\", "/")
    if not pattern.strip():
        raise ValueError("glob pattern cannot be empty")
    parts = tuple(part for part in pattern.split("/") if part and part != ".")
    if (pattern.startswith("/") or not parts or any(
        part == ".." or ":" in part or part.endswith((" ", ".")) for part in parts
    )):
        raise PermissionError("glob pattern must stay inside workspace without ambiguous paths")
    if sensitive_path_reason(parts, listing=True):
        raise PermissionError("protected glob target")
    return "/".join(parts) + ("/" if pattern.endswith("/") else "")


class WorkspaceGuard:
    """Resolve file-tool paths inside a configured workspace root."""

    def __init__(self, root: str | Path) -> None:
        self.root = resolved_path(Path(root).expanduser())

    def safe_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        # Windows alternate streams and device-style names must not alias a
        # protected .env/key under an apparently harmless filename.
        parts = candidate.parts[1:] if candidate.anchor else candidate.parts
        if any(":" in part or (part not in {".", ".."} and part.endswith((" ", "."))) for part in parts):
            raise PermissionError("ambiguous workspace path")
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = resolved_path(candidate)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(
                f"path escapes workspace: {raw_path}"
            ) from exc
        return resolved
