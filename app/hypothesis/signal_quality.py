from __future__ import annotations

from typing import Any

from app.context.schema import APPLICABILITY_CONTEXT_FIELDS, EXECUTION_CONTEXT_FIELDS


REVISION_DIMENSIONS = {"CONTEXT", "ACTION", "OUTCOME", "ABSTRACTION"}
REVISION_OPERATIONS = {
    "EXPAND",
    "NARROW",
    "ADD_CONDITION",
    "REMOVE_CONDITION",
    "CHANGE",
    "GENERALIZE",
    "SPECIALIZE",
}

# 第一版故意使用有限、稳定的 target_field taxonomy。
# Aggregator 需要能对 RevisionSignal 做确定性聚合，因此不允许模型自由发明字段名。
ACTION_TARGET_FIELDS = {
    "workflow_structure",
    "response_structure",
    "scope_alignment",
    "verification_step",
    "tool_usage",
    "memory_usage",
    "boundary_handling",
    "data_flow_handling",
    "error_handling",
}

OUTCOME_TARGET_FIELDS = {
    "clarity",
    "executability",
    "correctness",
    "efficiency",
    "boundary_consistency",
    "scope_drift",
    "responsibility_confusion",
    "evaluation_performance",
    "user_satisfaction",
}

ABSTRACTION_TARGET_FIELDS = {
    "context_scope",
    "action_scope",
    "outcome_scope",
    "hypothesis_scope",
}

TARGET_FIELDS_BY_DIMENSION = {
    "CONTEXT": set(APPLICABILITY_CONTEXT_FIELDS),
    "ACTION": ACTION_TARGET_FIELDS,
    "OUTCOME": OUTCOME_TARGET_FIELDS,
    "ABSTRACTION": ABSTRACTION_TARGET_FIELDS,
}

# 这些字段属于运行时基础设施，不是 Strategy Hypothesis 的适用边界。
# 即使模型把它们输出为 CONTEXT signal，也必须拒绝。
EXECUTION_ONLY_FIELDS = set(EXECUTION_CONTEXT_FIELDS)

# Phase 2 真实测试中出现过的过宽字段。保留显式黑名单，便于旧数据质量检查。
GENERIC_TARGET_FIELDS = {
    "context",
    "applicable_context",
    "when",
    "do",
    "expect",
    "do_data",
    "do_data.text",
    "expect_data",
    "expect_data.text",
    "action",
    "outcome",
}


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _normalize_atomic_value(value: Any, *, allow_multi_value: bool) -> tuple[Any, str | None]:
    """保证一个 RevisionSignal 只表达一个语义变化。

    CONTEXT 的枚举集合（如 medium/high）允许短列表；ACTION/OUTCOME/ABSTRACTION
    必须是单值。如果模型输出多个动作/结果，应拆成多个 signal，而不是塞在一条里。
    """
    if _is_empty(value):
        return value, "observed_data 为空"

    if isinstance(value, list):
        cleaned = [item for item in value if not _is_empty(item)]
        if not cleaned:
            return cleaned, "observed_data 为空"
        if allow_multi_value:
            if len(cleaned) > 4:
                return cleaned, "CONTEXT observed_data 过宽，超过 4 个值"
            return cleaned, None
        if len(cleaned) != 1:
            return cleaned, "一个 RevisionSignal 只能包含一个语义变化"
        return cleaned[0], None

    if isinstance(value, dict):
        cleaned = {k: v for k, v in value.items() if not _is_empty(v)}
        if not cleaned:
            return cleaned, "observed_data 为空"
        if len(cleaned) != 1:
            return cleaned, "一个 RevisionSignal 只能包含一个语义变化"
        return cleaned, None

    return value, None


def _context_value(payload: dict[str, Any] | None, field: str) -> Any:
    if not payload:
        return None
    context = payload.get("applicable_context") or payload.get("when_data") or {}
    if not isinstance(context, dict):
        return None
    return context.get(field)


def normalize_revision_signal(
    raw: Any,
    *,
    reflection_payload: dict[str, Any] | None = None,
    hypothesis_payload: dict[str, Any] | None = None,
    explicit_user_feedback: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """结构化校验 RevisionSignal；不调用 LLM。

    返回 (normalized_signal, reason)。normalized_signal=None 表示拒绝。
    """
    if not isinstance(raw, dict):
        return None, "signal 不是对象"

    dimension = str(raw.get("dimension", "")).strip().upper()
    operation = str(raw.get("operation", "")).strip().upper()
    target_field = str(raw.get("target_field", "")).strip()
    canonical_change = str(raw.get("canonical_change", "")).strip()

    if dimension not in REVISION_DIMENSIONS:
        return None, f"非法 dimension: {dimension or '<empty>'}"
    if operation not in REVISION_OPERATIONS:
        return None, f"非法 operation: {operation or '<empty>'}"
    if not target_field:
        return None, "target_field 为空"
    if target_field in GENERIC_TARGET_FIELDS:
        return None, f"target_field 过宽: {target_field}"
    if target_field in EXECUTION_ONLY_FIELDS:
        return None, f"Execution Context 不能作为适用条件: {target_field}"

    allowed_fields = TARGET_FIELDS_BY_DIMENSION[dimension]
    if target_field not in allowed_fields:
        return None, f"{dimension} 不允许 target_field={target_field}"
    if not canonical_change:
        return None, "canonical_change 为空"
    if len(canonical_change) > 280:
        return None, "canonical_change 过长"

    before_data = raw.get("before_data")
    observed_data = raw.get("observed_data")
    observed_data, atomic_error = _normalize_atomic_value(
        observed_data,
        allow_multi_value=(dimension == "CONTEXT"),
    )
    if atomic_error:
        return None, atomic_error

    # CONTEXT signal 必须与结构化 Context 对得上；避免把自然语言猜测写成适用边界。
    if dimension == "CONTEXT":
        reflection_value = _context_value(reflection_payload, target_field)
        if reflection_value is not None and observed_data != reflection_value:
            # list/scalar 的最小兼容：reflection=medium, observed=[medium]
            equivalent = (
                isinstance(observed_data, list)
                and len(observed_data) == 1
                and observed_data[0] == reflection_value
            )
            if not equivalent:
                return None, f"observed_data 与 Reflection Context.{target_field} 不一致"

        hypothesis_value = _context_value(hypothesis_payload, target_field)
        if hypothesis_value is not None and not _is_empty(before_data):
            equivalent_before = before_data == hypothesis_value or (
                isinstance(before_data, list)
                and len(before_data) == 1
                and before_data[0] == hypothesis_value
            )
            if not equivalent_before:
                return None, f"before_data 与 Hypothesis Context.{target_field} 不一致"

    # “用户满意度”只能由显式反馈支撑，不能从 evaluator score 自行推断。
    explicit_user_feedback = explicit_user_feedback or bool(
        reflection_payload and str(reflection_payload.get("user_feedback", "")).strip()
    )
    if dimension == "OUTCOME" and target_field == "user_satisfaction":
        if not explicit_user_feedback:
            return None, "user_satisfaction 缺少显式用户反馈"

    return {
        "dimension": dimension,
        "operation": operation,
        "target_field": target_field,
        "before_data": before_data,
        "observed_data": observed_data,
        "canonical_change": canonical_change,
        "confidence": _clamp_confidence(raw.get("confidence", 0.5)),
    }, "valid"


def persisted_signal_payload(signal: Any) -> dict[str, Any]:
    return {
        "dimension": getattr(signal, "dimension", ""),
        "operation": getattr(signal, "operation", ""),
        "target_field": getattr(signal, "target_field", ""),
        "before_data": getattr(signal, "before_data", None),
        "observed_data": getattr(signal, "observed_data", None),
        "canonical_change": getattr(signal, "canonical_change", ""),
        "confidence": getattr(signal, "confidence", 0.5),
    }
