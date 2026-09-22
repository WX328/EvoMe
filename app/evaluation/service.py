from typing import Any


EVALUATION_PROMPT = '''
你是 EvoMe 的 Evaluator。只做紧凑评分，不复述用户问题或回答，不展开推理。

只输出一个 JSON 对象，只允许三个键：
{"score":0.0,"success":false,"reason":""}

规则：
- score 范围 0~1。
- reason 最多 60 个汉字，只写一条最关键依据。
- 禁止额外字段、Markdown、前后说明。
- 评估：是否直接回应意图；是否正确使用相关 Memory；是否把推测当事实；是否具体可执行。
'''


class EvaluationService:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def normalize_result(result: dict[str, Any] | None) -> dict[str, Any]:
        result = result if isinstance(result, dict) else {}
        try:
            score = max(0.0, min(1.0, float(result.get("score", 0.5))))
        except (TypeError, ValueError):
            score = 0.5
        return {
            "score": score,
            "success": bool(result.get("success", score >= 0.7)),
            "reason": str(result.get("reason", "")).strip()[:240],
        }

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
        return self.normalize_result(result)
