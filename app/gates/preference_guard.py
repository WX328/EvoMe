from __future__ import annotations

import re


NO_STORE_MARKERS = (
    "不要记住",
    "别记住",
    "不要保存",
    "别保存",
    "不用记住",
    "不用保存",
)

AMBIGUOUS_NEGATION_MARKERS = (
    "不是说",
    "并不是",
    "不代表",
)

PREFERENCE_DISCUSSION_MARKERS = (
    "请判断",
    "请分析",
    "请评价",
    "是不是",
    "是否",
    "举个例子",
    "例如",
    "比如",
)

PREFERENCE_META_MARKERS = (
    "偏好",
    "这句话",
    "这个说法",
    "这条指令",
    "这个表达",
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(marker.lower() in lowered for marker in markers)


def explicitly_disallows_storage(text: str) -> bool:
    return _contains_any(text, NO_STORE_MARKERS)


def has_ambiguous_preference_negation(text: str) -> bool:
    return _contains_any(text, AMBIGUOUS_NEGATION_MARKERS)


def is_quoted_preference_discussion(text: str) -> bool:
    value = str(text or "")
    has_quote = bool(re.search(r"[‘’“”「」『』\"']", value))
    has_discussion = _contains_any(value, PREFERENCE_DISCUSSION_MARKERS)
    has_meta_subject = _contains_any(value, PREFERENCE_META_MARKERS)
    return has_discussion and (has_quote or has_meta_subject)
