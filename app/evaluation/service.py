from typing import Any


EVALUATION_PROMPT = '''
你是 EvoMe 的 Evaluator。

评估这次响应是否真正帮助了用户。
只输出 JSON：
{
  "score": 0.0,
  "success": false,
  "reason": ""
}

score 范围 0~1。
当前 V0.1 没有外部自动测试时，使用以下标准：
- 是否直接回应用户意图
- 是否利用了真正相关的历史 Memory
- 是否没有把历史推测成当前事实
- 是否具体、可执行
'''


class EvaluationService:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(
        self,
        *,
        user_input: str,
        response: str,
        relevant_memories: list[str],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": EVALUATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户输入：\n{user_input}\n\n"
                    f"EvoMe 回复：\n{response}\n\n"
                    f"检索到的 Memory：\n{relevant_memories}"
                ),
            },
        ]

        result = self.llm.json_chat(messages)

        score = max(0.0, min(1.0, float(result.get("score", 0.5))))
        return {
            "score": score,
            "success": bool(result.get("success", score >= 0.7)),
            "reason": str(result.get("reason", "")).strip(),
        }
