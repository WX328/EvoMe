from typing import Any

TASK_TYPES = {
    "qa",
    "debugging",
    "coding",
    "architecture_design",
    "project_planning",
    "learning",
    "research",
    "decision_support",
    "execution",
    "other",
}

COMPLEXITY_LEVELS = {"low", "medium", "high"}
RISK_LEVELS = {"low", "medium", "high"}
INTERACTION_MODES = {"direct", "collaborative", "teaching", "exploratory"}


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default
