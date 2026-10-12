from __future__ import annotations

import re
from typing import Any

from app.gates.preference_guard import (
    explicitly_disallows_storage,
    has_ambiguous_preference_negation,
    is_quoted_preference_discussion,
)


PERSONAL_MEMORY_GATE_PROMPT = """
你是 EvoMe 的 Personal Memory Gate。

只从“用户当前这一句话”中判断是否存在值得长期保存的用户信息。
禁止把 AI 回复、AI 推测或性格猜测当成用户事实。

允许 memory_type：fact / preference / goal / project / decision / knowledge

值得保存：长期目标、重要项目、明确决策、稳定偏好、明确完成/掌握的重要阶段、长期重要背景。
特别注意：用户明确说“以后/今后给我……”“这样我更容易理解”“默认按……”时，若表达的是未来跨会话可复用的交互/讲解偏好，应保存为 preference；只保存用户偏好本身，不把 AI 的回复细节扩写成用户事实。
通常不保存：普通知识问题、一次性临时请求、当前情绪、没有新增信息的旧事实。

严格 JSON：
{
  "should_save": true,
  "memories": [
    {
      "memory_type": "project",
      "content": "",
      "importance": 0.8,
      "confidence": 0.95
    }
  ]
}

没有新长期信息时：
{"should_save": false, "memories": []}
"""


class PersonalMemoryGate:
    """Hybrid gate: deterministic for explicit durable preferences, LLM otherwise."""

    _DURABLE_PREFERENCE_MARKERS = (
        "以后给我",
        "你以后给我",
        "今后给我",
        "以后帮我",
        "今后帮我",
        "以后默认",
        "今后默认",
        "默认按",
        "默认先",
        "我更喜欢",
        "这样我更容易",
        "我更容易理解",
        "这样更适合我",
        "对我更有帮助",
        "以后都",
        "今后都",
    )
    _TESTING_MARKERS = (
        "这只是测试",
        "只是测试",
        "不是真的",
        "假设我",
        "模拟用户",
        "测试记忆",
        "mock data",
        "synthetic data",
        "test only",
    )
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _contains_any(cls, text: str, markers: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in markers)

    @classmethod
    def _deterministic_extract(cls, user_input: str) -> dict[str, Any] | None:
        text = cls._normalize(user_input)
        if not text:
            return {"should_save": False, "memories": [], "decision_source": "deterministic_hard_skip"}

        if cls._contains_any(text, cls._TESTING_MARKERS) or explicitly_disallows_storage(text):
            return {
                "should_save": False,
                "memories": [],
                "decision_source": "deterministic_hard_skip",
            }

        # Negated/contrastive wording can reverse the apparent preference.
        # Leave those cases to the semantic gate instead of guessing.
        if has_ambiguous_preference_negation(text):
            return None

        # A preference mentioned as the subject of a question/example is not
        # itself a user preference. Let the semantic gate decide if needed.
        if is_quoted_preference_discussion(text):
            return None

        durable_preference = cls._contains_any(text, cls._DURABLE_PREFERENCE_MARKERS)
        if not durable_preference:
            lowered = text.lower()
            has_future = ("以后" in lowered) or ("今后" in lowered)
            has_user_reference = any(
                marker in lowered for marker in ("给我", "帮我", "我更", "对我", "我希望")
            )
            has_preference_shape = any(
                marker in lowered for marker in ("先", "默认", "更容易", "更喜欢", "都")
            )
            durable_preference = has_future and has_user_reference and has_preference_shape

        if not durable_preference:
            return None

        # Preserve the user's own wording instead of asking an LLM to paraphrase
        # it. This avoids hallucinated detail and makes provenance obvious.
        return {
            "should_save": True,
            "memories": [
                {
                    "memory_type": "preference",
                    "content": text[:500],
                    "importance": 0.9,
                    "confidence": 0.95,
                }
            ],
            "decision_source": "deterministic_explicit_preference",
        }

    def extract(self, user_input: str) -> dict[str, Any]:
        deterministic = self._deterministic_extract(user_input)
        if deterministic is not None:
            return deterministic

        result = self.llm.json_chat(
            [
                {"role": "system", "content": PERSONAL_MEMORY_GATE_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )

        memories = []
        for item in result.get("memories", []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            memories.append(
                {
                    "memory_type": str(item.get("memory_type", "fact")),
                    "content": content,
                    "importance": float(item.get("importance", 0.5)),
                    "confidence": float(item.get("confidence", 0.8)),
                }
            )

        return {
            "should_save": bool(result.get("should_save", False)) and bool(memories),
            "memories": memories,
            "decision_source": "llm",
        }
