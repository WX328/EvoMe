from typing import Any, TypedDict


class EvoMeState(TypedDict, total=False):
    user_input: str
    strategy_name: str
    strategy_version: int
    session_id: str

    relevant_memories: list[str]
    skills_used: list[str]
    response: str
    evaluation: dict[str, Any]
    interaction_id: int

    personal_memory_candidates: list[dict[str, Any]]
    personal_memories_saved: int

    experience_gate: dict[str, Any]
    experience_id: int | None
    experience_saved: bool

    reflection: dict[str, Any]
    reflection_saved: bool
