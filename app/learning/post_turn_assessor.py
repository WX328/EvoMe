from __future__ import annotations

import json
from typing import Any

from app.config import EVOME_PROMPT_MEMORY_CHAR_LIMIT
from app.prompt_utils import compact_for_prompt


POST_TURN_ASSESSOR_PROMPT = """
MODULE: POST_TURN_ASSESSOR
Return compact structured data for one completed turn. No repetition, no chain-of-thought, no Markdown, no extra keys.

JSON shape (every key required):
{"evaluation":{"score":0.0,"success":false,"reason":""},"context":{"task_type":"other","task_type_confidence":0.8,"interaction_mode":"direct","interaction_mode_confidence":0.8,"explicit_instruction":"","project_name":"unknown","project_name_confidence":0.5,"project_stage":"unknown","project_stage_confidence":0.5,"complexity_indicators":{"multi_step":false,"ambiguity":false,"cross_module":false,"external_tool":false,"long_term_state":false,"tradeoff":false},"risk_indicators":{"destructive_change":false,"external_side_effect":false,"security_sensitive":false,"irreversible":false}},"reflection":{"reflection_type":"success","observation":"","insight":"","applicable_context":{},"strategy_signal":"","skill_signal":"","expected_effect":"","confidence":0.8}}

Rules:
- evaluation: score 0..1; success = whether the answer helped the user. reason <=60 汉字, one decisive reason only.
- context: task_type is one of qa/debugging/coding/architecture_design/project_planning/learning/research/decision_support/execution/other. interaction_mode is one of direct/collaborative/teaching/exploratory. Use "unknown" when uncertain. explicit_instruction <=80 汉字; project fields <=30. complexity/risk indicators are booleans only.
- reflection: a DRAFT only; downstream gates decide whether it is persisted. reflection_type is one of failure/success/pattern/conflict. observation <=60, insight <=80, strategy_signal <=80, skill_signal <=60, expected_effect <=60. Keep one local reusable lesson, not a universal rule. applicable_context contains evidence-backed applicability fields only — never include model/tools/skills/memory quality. Do not infer user satisfaction from the evaluator score. Do not turn AI wording into a user fact. Leave strategy_signal/skill_signal empty when there is no real learning signal.
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
            "relevant_memories": [
                compact_for_prompt(memory, limit=EVOME_PROMPT_MEMORY_CHAR_LIMIT)
                for memory in list(relevant_memories[:5])
            ],
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
