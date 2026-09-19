from typing import Any


REFLECTION_PROMPT = """
你是 EvoMe 的 Reflection 模块。
只对已经被 Experience Gate 判定为“有学习价值”的经验进行反思。

回答：这次经历说明什么？有什么可泛化经验？是否有 Strategy/Skill 调整信号？
允许 reflection_type：failure / success / pattern / conflict

严格 JSON：
{
  "reflection_type": "success",
  "observation": "",
  "lesson": "",
  "strategy_signal": "",
  "skill_signal": "",
  "confidence": 0.8
}
"""


class ReflectionService:
    def __init__(self, llm):
        self.llm = llm

    def reflect(
        self,
        *,
        task_type: str,
        result_summary: str,
        evaluation: dict[str, Any],
        strategy_name: str,
        skills_used: list[str],
    ) -> dict[str, Any]:
        result = self.llm.json_chat(
            [
                {"role": "system", "content": REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task Type：{task_type}\n"
                        f"Strategy：{strategy_name}\n"
                        f"Skills：{skills_used}\n"
                        f"Result：{result_summary}\n"
                        f"Evaluation：{evaluation}"
                    ),
                },
            ],
            temperature=0.1,
        )
        reflection_type = str(result.get("reflection_type", "success")).strip()
        if reflection_type not in {"failure", "success", "pattern", "conflict"}:
            reflection_type = "success"

        return {
            "reflection_type": reflection_type,
            "observation": str(result.get("observation", "")).strip(),
            "lesson": str(result.get("lesson", "")).strip(),
            "strategy_signal": str(result.get("strategy_signal", "")).strip(),
            "skill_signal": str(result.get("skill_signal", "")).strip(),
            "confidence": float(result.get("confidence", 0.8)),
        }
