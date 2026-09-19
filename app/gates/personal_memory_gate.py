from typing import Any


PERSONAL_MEMORY_GATE_PROMPT = """
你是 EvoMe 的 Personal Memory Gate。

只从“用户当前这一句话”中判断是否存在值得长期保存的用户信息。
禁止把 AI 回复、AI 推测或性格猜测当成用户事实。

允许 memory_type：fact / preference / goal / project / decision / knowledge

值得保存：长期目标、重要项目、明确决策、稳定偏好、明确完成/掌握的重要阶段、长期重要背景。
通常不保存：普通知识问题、一次性临时请求、当前情绪、没有新增信息的旧事实。

严格 JSON：
{
  "should_save": true,
  "memories": [
    {
      "memory_type": "project",
      "content": "",
      "importance": 0.8,
      "confidence": 0.95
    }
  ]
}

没有新长期信息时：
{"should_save": false, "memories": []}
"""


class PersonalMemoryGate:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, user_input: str) -> dict[str, Any]:
        result = self.llm.json_chat(
            [
                {"role": "system", "content": PERSONAL_MEMORY_GATE_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )

        memories = []
        for item in result.get("memories", []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            memories.append(
                {
                    "memory_type": str(item.get("memory_type", "fact")),
                    "content": content,
                    "importance": float(item.get("importance", 0.5)),
                    "confidence": float(item.get("confidence", 0.8)),
                }
            )

        return {
            "should_save": bool(result.get("should_save", False)) and bool(memories),
            "memories": memories,
        }
