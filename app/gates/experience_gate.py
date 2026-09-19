from typing import Any


EXPERIENCE_GATE_PROMPT = """
你是 EvoMe 的 Experience Gate。

判断“EvoMe 这次执行本身”是否值得成为进化经验。

值得保存：失败/部分失败、用户纠正、明确的成功模式、Strategy/Skill 有意义地参与且有结果、与历史经验冲突。
通常不保存：普通定义问答、重复回答、低价值闲聊、没有学习信号的普通成功。

严格 JSON：
{
  "should_save": true,
  "task_type": "general",
  "result_summary": "",
  "reason": ""
}
"""


class ExperienceGate:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(
        self,
        *,
        user_input: str,
        assistant_response: str,
        evaluation: dict[str, Any],
        strategy_name: str,
        skills_used: list[str],
    ) -> dict[str, Any]:
        result = self.llm.json_chat(
            [
                {"role": "system", "content": EXPERIENCE_GATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户输入：\n{user_input}\n\n"
                        f"EvoMe 回复：\n{assistant_response}\n\n"
                        f"Strategy：{strategy_name}\n"
                        f"Skills：{skills_used}\n"
                        f"Evaluation：{evaluation}"
                    ),
                },
            ],
            temperature=0.1,
        )
        return {
            "should_save": bool(result.get("should_save", False)),
            "task_type": str(result.get("task_type", "general")).strip() or "general",
            "result_summary": str(result.get("result_summary", "")).strip(),
            "reason": str(result.get("reason", "")).strip(),
        }
