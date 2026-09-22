from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class TokenUsageTracker:
    """Process-local token telemetry shared by foreground/background LLM clients."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        purpose: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        finish_reason: str = "",
    ) -> None:
        reason = str(finish_reason or "unknown")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": str(purpose or "unknown"),
            "model": str(model or "unknown"),
            "prompt_tokens": max(0, int(prompt_tokens or 0)),
            "completion_tokens": max(0, int(completion_tokens or 0)),
            "total_tokens": max(0, int(total_tokens or 0)),
            "cached_tokens": max(0, int(cached_tokens or 0)),
            "reasoning_tokens": max(0, int(reasoning_tokens or 0)),
            "visible_completion_tokens": max(
                0,
                int(completion_tokens or 0) - int(reasoning_tokens or 0),
            ),
            "finish_reason": reason,
            "truncated": reason == "length",
        }
        with self._lock:
            self._records.append(row)

    def record_usage(
        self,
        *,
        purpose: str,
        model: str,
        usage: Any,
        finish_reason: str = "",
    ) -> None:
        if usage is None:
            return
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            getattr(completion_details, "reasoning_tokens", 0)
            if completion_details is not None
            else 0
        )
        self.record(
            purpose=purpose,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            cached_tokens=cached or 0,
            reasoning_tokens=reasoning_tokens or 0,
            finish_reason=finish_reason,
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        totals = {
            "calls": len(records),
            "prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "completion_tokens": sum(r["completion_tokens"] for r in records),
            "total_tokens": sum(r["total_tokens"] for r in records),
            "cached_tokens": sum(r["cached_tokens"] for r in records),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in records),
            "visible_completion_tokens": sum(
                r["visible_completion_tokens"] for r in records
            ),
            "truncated_calls": sum(1 for r in records if r["truncated"]),
        }
        grouped: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "visible_completion_tokens": 0,
                "truncated_calls": 0,
            }
        )
        for row in records:
            bucket = grouped[row["purpose"]]
            bucket["calls"] += 1
            bucket["prompt_tokens"] += row["prompt_tokens"]
            bucket["completion_tokens"] += row["completion_tokens"]
            bucket["total_tokens"] += row["total_tokens"]
            bucket["cached_tokens"] += row["cached_tokens"]
            bucket["reasoning_tokens"] += row["reasoning_tokens"]
            bucket["visible_completion_tokens"] += row["visible_completion_tokens"]
            bucket["truncated_calls"] += int(row["truncated"])
        by_purpose = [
            {"purpose": purpose, **values}
            for purpose, values in sorted(
                grouped.items(), key=lambda item: item[1]["total_tokens"], reverse=True
            )
        ]
        return {"totals": totals, "by_purpose": by_purpose, "recent": records[-20:]}

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


GLOBAL_TOKEN_USAGE = TokenUsageTracker()
