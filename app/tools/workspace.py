from __future__ import annotations

from pathlib import Path


class WorkspaceGuard:
    """Resolve file-tool paths inside a configured workspace root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def safe_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(
                f"path escapes workspace: {raw_path}"
            ) from exc
        return resolved
