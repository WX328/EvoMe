from typing import Any

TASK_TYPES = {
    "qa", "debugging", "coding", "architecture_design", "project_planning",
    "learning", "research", "decision_support", "execution", "other",
}
COMPLEXITY_LEVELS = {"low", "medium", "high"}
RISK_LEVELS = {"low", "medium", "high"}
INTERACTION_MODES = {"direct", "collaborative", "teaching", "exploratory"}

APPLICABILITY_CONTEXT_FIELDS = {
    "task_type", "complexity", "risk_level", "interaction_mode", "project_stage",
    "cross_module", "multi_step", "tradeoff", "ambiguity", "long_term_state",
    "explicit_instruction_present", "scope_bounded",
}
EXECUTION_CONTEXT_FIELDS = {
    "available_skills", "available_tools", "model", "relevant_memory_count",
    "memory_quality", "external_tool",
}

def clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))

def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default

def split_context(context: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    context = context or {}
    explicit_app = context.get("applicability_context")
    explicit_exec = context.get("execution_context")
    if isinstance(explicit_app, dict) and isinstance(explicit_exec, dict):
        return dict(explicit_app), dict(explicit_exec)

    complexity_indicators = context.get("complexity_indicators") or {}
    if not isinstance(complexity_indicators, dict):
        complexity_indicators = {}

    applicability: dict[str, Any] = {}
    for field in APPLICABILITY_CONTEXT_FIELDS:
        if field in context:
            applicability[field] = context[field]
        elif field in complexity_indicators:
            applicability[field] = bool(complexity_indicators[field])

    execution: dict[str, Any] = {}
    for field in EXECUTION_CONTEXT_FIELDS:
        if field in context:
            execution[field] = context[field]
        elif field == "external_tool" and field in complexity_indicators:
            execution[field] = bool(complexity_indicators[field])
    return applicability, execution

def sanitize_applicable_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    return {k: v for k, v in context.items() if k in APPLICABILITY_CONTEXT_FIELDS}
