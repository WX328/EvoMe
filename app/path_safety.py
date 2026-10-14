"""Canonical filesystem identities shared by storage and tool boundaries."""
import os
from pathlib import Path


def resolved_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    # Windows can retain the extended-length prefix when a concurrently opened
    # file prevents realpath's final existence check. Compare equivalent anchors.
    if os.name == "nt":
        value = str(resolved)
        if value.startswith("\\\\?\\UNC\\"):
            return Path("\\\\" + value[8:])
        if value.startswith("\\\\?\\") and len(value) > 6 and value[5] == ":":
            return Path(value[4:])
    return resolved
