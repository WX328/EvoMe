from typing import Any, TypedDict


class EvoMeState(TypedDict, total=False):
    user_input: str
    strategy_name: str

    relevant_memories: list[str]
    response: str

    evaluation: dict[str, Any]
    reflection: dict[str, Any]

    memory_saved: bool
