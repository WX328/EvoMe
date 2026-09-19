from typing import Any


REFLECTION_PROMPT = '''
你是 EvoMe 的 Reflection 模块。

你不负责回答用户，而负责从一次真实交互中提取“经验”。

只保存对未来行为有帮助的信息。
不要把 AI 自己的猜测当成用户事实。

你必须输出 JSON：
{
  "should_save": true,
  "memory_type": "experience",
  "importance": 0.0,
  "confidence": 0.0,
  "summary": "",
  "lesson": "",
  "strategy_signal": ""
}

规则：
- memory_type 当前固定使用 experience
- importance/confidence 范围 0~1
- summary 描述发生了什么
- lesson 描述从结果中能学到什么
- strategy_signal 只描述是否值得未来调整行为；没有则为空字符串
- 普通闲聊、低价值一次性信息 should_save=false
'''


class ReflectionService:
    def __init__(self, llm):
        self.llm = llm

    def reflect(
        self,
        *,
        user_input: str,
        response: str,
        evaluation: dict[str, Any],
        relevant_memories: list[str],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": REFLECTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户输入：\n{user_input}\n\n"
                    f"EvoMe 回复：\n{response}\n\n"
                    f"Evaluation：\n{evaluation}\n\n"
                    f"相关历史 Memory：\n{relevant_memories}\n"
                ),
            },
        ]

        result = self.llm.json_chat(messages)

        return {
            "should_save": bool(result.get("should_save", False)),
            "memory_type": "experience",
            "importance": float(result.get("importance", 0.5)),
            "confidence": float(result.get("confidence", 0.8)),
            "summary": str(result.get("summary", "")).strip(),
            "lesson": str(result.get("lesson", "")).strip(),
            "strategy_signal": str(
                result.get("strategy_signal", "")
            ).strip(),
        }
