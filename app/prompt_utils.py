"""Helpers for keeping LLM prompt payloads bounded.

Phase 5.7C: long assistant responses were being embedded verbatim into
internal JSON-helper prompts (post-turn assessor, experience gate). Those
calls run every turn, so the full response was paid for on every turn.
These helpers keep the head and the tail of a long text — enough for a
judge to see what was addressed and how it concluded — while dropping the
middle.
"""

from __future__ import annotations

from app.config import EVOME_PROMPT_RESPONSE_CHAR_LIMIT


def compact_for_prompt(text: str, *, limit: int | None = None) -> str:
    """Bound a long text to roughly `limit` characters, keeping head and tail.

    Short text is returned unchanged. Long text keeps the first 2/3 and the
    last 1/3 of the budget, with an explicit marker recording how much was
    omitted so the model knows the text was elided.
    """
    budget = EVOME_PROMPT_RESPONSE_CHAR_LIMIT if limit is None else limit
    value = (text or "").strip()
    if budget <= 0 or len(value) <= budget:
        return value
    head_len = (budget * 2) // 3
    tail_len = budget - head_len
    omitted = len(value) - head_len - tail_len
    return (
        f"{value[:head_len]}\n"
        f"...[已省略 {omitted} 字]...\n"
        f"{value[-tail_len:]}"
    )
