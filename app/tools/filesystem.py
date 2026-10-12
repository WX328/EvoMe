from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.workspace import WorkspaceGuard


class FileSystemTools:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        max_output_chars: int = 12000,
    ) -> None:
        self.guard = WorkspaceGuard(workspace_root)
        self.max_output_chars = max(1000, int(max_output_chars))

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        remaining = len(text) - self.max_output_chars
        return text[: self.max_output_chars] + f"\n...[truncated {remaining} chars]"

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        target = self.guard.safe_path(path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_file():
            raise IsADirectoryError(str(target))

        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else max(start, int(end_line))
        selected = lines[start - 1 : end]
        numbered = "\n".join(
            f"{line_no}: {line}"
            for line_no, line in enumerate(selected, start=start)
        )
        return {
            "path": str(target.relative_to(self.guard.root)),
            "start_line": start,
            "end_line": start + max(0, len(selected) - 1),
            "total_lines": len(lines),
            "content": self._clip(numbered),
        }

    def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> dict[str, Any]:
        target = self.guard.safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8", newline="") as handle:
            handle.write(str(content))
        return {
            "path": str(target.relative_to(self.guard.root)),
            "bytes": target.stat().st_size,
            "mode": "append" if append else "rewrite",
        }

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        expected_replacements: int = 1,
    ) -> dict[str, Any]:
        target = self.guard.safe_path(path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_file():
            raise IsADirectoryError(str(target))

        text = target.read_text(encoding="utf-8")
        count = text.count(old_string)
        expected = int(expected_replacements)
        if count != expected:
            raise ValueError(
                f"expected {expected} matches, found {count} in "
                f"{target.relative_to(self.guard.root)}"
            )
        target.write_text(text.replace(old_string, new_string), encoding="utf-8")
        return {
            "path": str(target.relative_to(self.guard.root)),
            "replacements": count,
        }

    def glob(self, pattern: str, max_results: int = 100) -> dict[str, Any]:
        limit = min(500, max(1, int(max_results)))
        raw_pattern = str(pattern).strip()
        if not raw_pattern:
            raise ValueError("glob pattern cannot be empty")
        if Path(raw_pattern).is_absolute() or ".." in Path(raw_pattern).parts:
            raise PermissionError("glob pattern must stay inside workspace")

        matches = []
        for item in self.guard.root.glob(raw_pattern):
            try:
                relative = item.resolve(strict=False).relative_to(self.guard.root)
            except ValueError:
                continue
            matches.append(
                {
                    "path": str(relative),
                    "type": "dir" if item.is_dir() else "file",
                }
            )
            if len(matches) >= limit:
                break
        return {
            "pattern": raw_pattern,
            "count": len(matches),
            "matches": matches,
        }
