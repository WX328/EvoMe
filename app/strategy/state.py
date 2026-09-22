from typing import Any, TypedDict


class EvoMeState(TypedDict, total=False):
    user_input: str
    strategy_name: str
    strategy_version: int
    session_id: str

    relevant_memories: list[str]
    memory_quality: float
    skills_used: list[str]
    response: str
    learning_eligibility: dict[str, Any]
    evaluation: dict[str, Any]
    post_turn_assessment: dict[str, Any]
    interaction_id: int

    personal_memory_candidates: list[dict[str, Any]]
    personal_memories_saved: int

    experience_gate: dict[str, Any]
    context_data: dict[str, Any]
    experience_id: int | None
    experience_saved: bool

    reflection: dict[str, Any]
    reflection_id: int | None
    reflection_saved: bool

    hypothesis_id: int | None
    hypothesis_action: str
    revision_signal_count: int
