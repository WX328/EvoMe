from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillImplementation:
    name: str
    skill_name: str
    kind: str
    handler: Callable[..., Any]
    version: int = 1
    enabled: bool = True

    def execute(self, **kwargs) -> Any:
        return self.handler(**kwargs)
