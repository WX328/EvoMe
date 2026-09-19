from app.skills.base import SkillDefinition, SkillImplementation


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._implementations: dict[str, list[SkillImplementation]] = {}

    def register_skill(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill
        self._implementations.setdefault(skill.name, [])

    def register_implementation(
        self,
        implementation: SkillImplementation,
    ) -> None:
        if implementation.skill_name not in self._skills:
            raise KeyError(
                f"Skill 未注册: {implementation.skill_name}"
            )
        self._implementations[implementation.skill_name].append(
            implementation
        )

    def get_skill(self, name: str) -> SkillDefinition:
        return self._skills[name]

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def list_implementations(
        self,
        skill_name: str,
    ) -> list[SkillImplementation]:
        return list(self._implementations.get(skill_name, []))

    def resolve(
        self,
        skill_name: str,
    ) -> SkillImplementation:
        candidates = [
            impl
            for impl in self._implementations.get(skill_name, [])
            if impl.enabled
        ]

        if not candidates:
            raise LookupError(
                f"Skill '{skill_name}' 没有可用实现"
            )

        # V0.1：先选择版本号最高的实现。
        # 后续再替换成按成功率/成本/权限动态选择。
        return max(candidates, key=lambda impl: impl.version)
