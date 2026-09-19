from __future__ import annotations

import json
from typing import Any


IMPORTANT_CONTEXT_KEYS = (
    "task_type",
    "complexity",
    "interaction_mode",
    "project_name",
    "project_stage",
)


def normalize_context(context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    normalized: dict[str, Any] = {}
    for key in IMPORTANT_CONTEXT_KEYS:
        value = context.get(key)
        if value in (None, "", "unknown", [], {}):
            continue
        normalized[key] = value
    return normalized


def build_canonical_signal(
    *,
    applicable_context: dict[str, Any] | None,
    proposed_change: str,
    expected_effect: str = "",
    insight: str = "",
) -> str:
    context = normalize_context(applicable_context)
    context_text = json.dumps(context, ensure_ascii=False, sort_keys=True)
    effect = expected_effect.strip() or insight.strip()
    return (
        f"WHEN {context_text}\n"
        f"DO {proposed_change.strip()}\n"
        f"EXPECT {effect}"
    ).strip()


def build_hypothesis_title(insight: str, proposed_change: str) -> str:
    base = insight.strip() or proposed_change.strip() or "未命名策略假设"
    base = " ".join(base.split())
    return base[:120]
