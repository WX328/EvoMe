from __future__ import annotations

import json
from typing import Any

from app.prompt_utils import compact_for_prompt


POST_TURN_ASSESSOR_PROMPT = """
MODULE: POST_TURN_ASSESSOR
Analyze one completed EvoMe turn once, then return compact structured data for three downstream domains: evaluation, context, and a reflection draft.

Do not repeat the input, do not expose chain-of-thought, do not output Markdown, and do not add extra top-level keys.

Strict JSON shape:
{
  "evaluation": {"score": 0.0, "success": false, "reason": ""},
  "context": {
    "task_type": "other",
    "task_type_confidence": 0.8,
    "interaction_mode": "direct",
    "interaction_mode_confidence": 0.8,
    "explicit_instruction": "",
    "project_name": "unknown",
    "project_name_confidence": 0.5,
    "project_stage": "unknown",
    "project_stage_confidence": 0.5,
    "complexity_indicators": {
      "multi_step": false, "ambiguity": false, "cross_module": false,
      "external_tool": false, "long_term_state": false, "tradeoff": false
    },
    "risk_indicators": {
      "destructive_change": false, "external_side_effect": false,
      "security_sensitive": false, "irreversible": false
    }
  },
  "reflection": {
    "reflection_type": "success",
    "observation": "",
    "insight": "",
    "applicable_context": {},
    "strategy_signal": "",
    "skill_signal": "",
    "expected_effect": "",
    "confidence": 0.8
  }
}

Evaluation rules:
- score is 0..1; success reflects whether the answer helped the user.
- reason <= 60 Chinese characters or equivalent; one decisive reason only.

Context rules:
- task_type is one of qa/debugging/coding/architecture_design/project_planning/learning/research/decision_support/execution/other.
- interaction_mode is one of direct/collaborative/teaching/exploratory.
- unknown when uncertain; confidence is 0..1.
- explicit_instruction <= 80 Chinese characters; project fields <= 30.
- complexity/risk indicators are booleans only.

Reflection rules:
- This is only a draft. Downstream gates decide whether it is persisted.
- reflection_type is failure/success/pattern/conflict.
- observation <= 60 characters; insight <= 80; strategy_signal <= 80; skill_signal <= 60; expected_effect <= 60.
- Keep one local reusable lesson, not a universal rule.
- applicable_context only contains evidence-backed applicability fields; never include model/tools/skills/memory quality.
- Do not infer user satisfaction from evaluator score. Do not turn AI wording into a user fact.
- If there is no meaningful strategy/skill learning signal, leave the corresponding field empty.
"""


class PostTurnAssessor:
    """One-pass semantic assessment for the per-turn evolution path."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def assess(
        self,
        *,
        user_input: str,
        assistant_response: str,
        relevant_memories: list[str],
        strategy_name: str,
        skills_used: list[str],
    ) -> dict[str, Any]:
        payload = {
            "user_input": user_input,
            "assistant_response": compact_for_prompt(assistant_response),
            "relevant_memories": list(relevant_memories[:5]),
            "strategy": strategy_name,
            "skills_used": list(skills_used),
        }
        result = self.llm.json_chat(
            [
                {"role": "system", "content": POST_TURN_ASSESSOR_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            temperature=0.1,
        )
        if not isinstance(result, dict):
            return {"evaluation": {}, "context": {}, "reflection": {}}
        return {
            "evaluation": result.get("evaluation")
            if isinstance(result.get("evaluation"), dict)
            else {},
            "context": result.get("context")
            if isinstance(result.get("context"), dict)
            else {},
            "reflection": result.get("reflection")
            if isinstance(result.get("reflection"), dict)
            else {},
        }
