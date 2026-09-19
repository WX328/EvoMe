from __future__ import annotations

from typing import Any


RELATIONS = {"SUPPORTS", "REFINES", "CONFLICTS", "UNRELATED"}

RELATION_JUDGE_PROMPT = """
MODULE: HYPOTHESIS_RELATION_JUDGE
你是 EvoMe 的 Strategy Hypothesis Relation Judge。
判断“新的 Reflection”与“已有 Strategy Hypothesis”的语义关系。

只能选择：
- SUPPORTS：直接支持“当前 Hypothesis 的现有表述”；Context/Action/Expected Outcome 基本一致，不需要修改当前 WHEN/DO/EXPECT。
- REFINES：与核心规律相关，但新证据要求扩展、收窄或修正当前 WHEN/DO/EXPECT。REFINES 不是对当前表述的直接支持。
- CONFLICTS：在相近 Context 下给出相反证据，或表明原规律可能失效。
- UNRELATED：不是同一个行为规律。

不要用文本相似度代替语义判断。重点比较：
1. Context 是否相同/重叠；
2. Proposed Change 是否同向；
3. Expected Effect 是否同一目标；
4. 如果接纳新 Reflection 后必须改写当前 Hypothesis 的适用边界或行为描述，应判为 REFINES，而不是 SUPPORTS。

严格 JSON：
{
  "relation": "SUPPORTS",
  "confidence": 0.9,
  "reason": ""
}
"""


class HypothesisRelationJudge:
    def __init__(self, llm) -> None:
        self.llm = llm

    def judge(
        self,
        *,
        reflection_payload: dict[str, Any],
        hypothesis_payload: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.llm.json_chat(
            [
                {"role": "system", "content": RELATION_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"New Reflection：\n{reflection_payload}\n\n"
                        f"Existing Hypothesis：\n{hypothesis_payload}"
                    ),
                },
            ],
            temperature=0.0,
        )
        relation = str(result.get("relation", "UNRELATED")).strip().upper()
        if relation not in RELATIONS:
            relation = "UNRELATED"
        try:
            confidence = float(result.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "relation": relation,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(result.get("reason", "")).strip(),
        }
