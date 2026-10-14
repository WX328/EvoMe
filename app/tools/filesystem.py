from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from app.tools.workspace import WorkspaceGuard, safe_glob_pattern, sensitive_path_reason


class FileSystemTools:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        max_output_chars: int = 12000,
        listing_excludes: tuple[str | Path, ...] = (),
    ) -> None:
        self.guard = WorkspaceGuard(workspace_root)
        self.max_output_chars = max(1000, int(max_output_chars))
        self.listing_excludes = tuple(self.guard.safe_path(path) for path in listing_excludes
            if (self.guard.root / path).resolve().is_relative_to(self.guard.root))

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
        # Page on line boundaries so end_line describes content actually sent
        # to the model, rather than the full requested range before clipping.
        rendered: list[str] = []
        size = 0
        partial_line = False
        for line_no, line in enumerate(selected, start=start):
            numbered = f"{line_no}: {line}"
            needed = len(numbered) + bool(rendered)
            if size + needed > self.max_output_chars:
                if not rendered:
                    rendered.append(self._clip(numbered))
                    partial_line = True
                break
            rendered.append(numbered)
            size += needed
        actual_end = start + len(rendered) - 1
        truncated = partial_line or len(rendered) < len(selected)
        return {
            "path": str(target.relative_to(self.guard.root)),
            "start_line": start,
            "end_line": actual_end,
            "total_lines": len(lines),
            "content": "\n".join(rendered),
            "truncated": truncated,
            "line_truncated": partial_line,
            "next_start_line": actual_end + 1 if truncated and not partial_line else None,
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
        raw_pattern = safe_glob_pattern(pattern)
        directories_only = raw_pattern.endswith("/")
        parts = raw_pattern.rstrip("/").split("/")
        matches = []
        seen = set()
        # Match one path component at a time, pruning before scandir. Never
        # enter links/junctions, even when their target is inside the workspace.
        pending = [(self.guard.root, 0)]
        visited = set()
        while pending and len(matches) < limit:
            directory, index = pending.pop()
            if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
                continue
            try:
                self.guard.safe_path(directory)
            except (OSError, ValueError, RuntimeError):
                continue
            state = (directory, index)
            if state in visited:
                continue
            visited.add(state)
            if index == len(parts):
                entries = [(directory, True)]
            else:
                segment = parts[index]
                if segment == "**":
                    pending.append((directory, index + 1))
                entries = []
                with os.scandir(directory) as scan:
                    for entry in scan:
                        item = Path(entry.path)
                        relative = item.relative_to(self.guard.root)
                        if sensitive_path_reason(relative.parts, listing=True):
                            continue
                        if any(item.is_relative_to(path) for path in self.listing_excludes):
                            continue
                        if entry.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
                            continue
                        try:
                            self.guard.safe_path(item)
                        except (OSError, ValueError, RuntimeError):
                            continue
                        is_dir = entry.is_dir(follow_symlinks=False)
                        if segment == "**":
                            if is_dir:
                                pending.append((item, index))
                            elif index == len(parts) - 1:
                                entries.append((item, False))
                        elif fnmatch.fnmatch(entry.name, segment):
                            if index == len(parts) - 1:
                                entries.append((item, is_dir))
                            elif is_dir:
                                pending.append((item, index + 1))
            for item, is_dir in entries:
                if (item == self.guard.root or item in seen or
                        (directories_only and not is_dir)):
                    continue
                seen.add(item)
                matches.append({"path": str(item.relative_to(self.guard.root)),
                                "type": "dir" if is_dir else "file"})
                if len(matches) == limit:
                    break
        return {
            "pattern": raw_pattern,
            "count": len(matches),
            "matches": matches,
        }
