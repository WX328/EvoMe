from __future__ import annotations

from typing import Any

from app.context.schema import (
    COMPLEXITY_LEVELS,
    INTERACTION_MODES,
    RISK_LEVELS,
    TASK_TYPES,
    clamp_confidence,
    normalize_enum,
    split_context,
)


CONTEXT_BUILDER_PROMPT = """
MODULE: CONTEXT_BUILDER
把当前任务环境压缩成结构化 Context。只输出 JSON，不解释、不复述输入、不输出 Markdown。

固定枚举：
- task_type: qa / debugging / coding / architecture_design / project_planning / learning / research / decision_support / execution / other
- interaction_mode: direct / collaborative / teaching / exploratory

规则：
- 当前明确输入优先于历史；不确定用 unknown。
- explicit_instruction 最多 80 个汉字；没有则空字符串。
- project_name / project_stage 最多 30 个汉字；不确定用 unknown。
- confidence 仅 0~1 数字。
- complexity_indicators / risk_indicators 只填布尔值。
- 禁止新增 reason / rationale / analysis 等字段。

严格输出：
{"task_type":"other","task_type_confidence":0.8,"interaction_mode":"direct","interaction_mode_confidence":0.8,"explicit_instruction":"","project_name":"unknown","project_name_confidence":0.5,"project_stage":"unknown","project_stage_confidence":0.5,"complexity_indicators":{"multi_step":false,"ambiguity":false,"cross_module":false,"external_tool":false,"long_term_state":false,"tradeoff":false},"risk_indicators":{"destructive_change":false,"external_side_effect":false,"security_sensitive":false,"irreversible":false}}
"""


class ContextBuilder:
    def __init__(self, llm, *, model_name: str = "unknown") -> None:
        self.llm = llm
        self.model_name = model_name

    @staticmethod
    def _explicit_mode(user_input: str) -> str | None:
        text = user_input.lower()
        collaborative_markers = ("一起讨论", "先讨论", "一起设计", "共同设计", "一起分析")
        direct_markers = ("直接给", "直接写", "不用解释", "只要结果", "直接实现")
        teaching_markers = ("教我", "讲解", "一步一步", "带我学", "怎么理解")
        exploratory_markers = ("你怎么看", "讨论一下", "探索", "脑暴", "有什么可能")
        if any(marker in text for marker in collaborative_markers):
            return "collaborative"
        if any(marker in text for marker in direct_markers):
            return "direct"
        if any(marker in text for marker in teaching_markers):
            return "teaching"
        if any(marker in text for marker in exploratory_markers):
            return "exploratory"
        return None

    @staticmethod
    def _level_from_count(count: int, *, low_max: int, medium_max: int) -> str:
        if count <= low_max:
            return "low"
        if count <= medium_max:
            return "medium"
        return "high"

    @staticmethod
    def _bool_count(data: dict[str, Any]) -> int:
        return sum(1 for value in data.values() if bool(value))

    @staticmethod
    def _field_meta(source: str, confidence: float) -> dict[str, Any]:
        return {"source": source, "confidence": clamp_confidence(confidence)}

    @staticmethod
    def _scope_bounded(user_input: str) -> bool:
        text = user_input.lower()
        markers = (
            "只想", "只需要", "只讨论", "只分析", "仅",
            "不涉及", "不要涉及", "不讨论", "限定", "范围",
            "only", "without", "limit to", "scope", "do not include",
        )
        return any(marker in text for marker in markers)

    def build(
        self,
        *,
        user_input: str,
        relevant_memories: list[str],
        available_skills: list[str],
        available_tools: list[str] | None = None,
        memory_quality: float = 0.0,
        task_type_hint: str | None = None,
    ) -> dict[str, Any]:
        memory_text = "\n".join(f"- {item}" for item in relevant_memories[:5]) or "无"
        result = self.llm.json_chat(
            [
                {"role": "system", "content": CONTEXT_BUILDER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"当前用户输入：\n{user_input}\n\n"
                        f"Experience Gate task_type hint：{task_type_hint or 'none'}\n\n"
                        f"相关 Personal Memory：\n{memory_text}"
                    ),
                },
            ],
            temperature=0.0,
        )

        explicit_mode = self._explicit_mode(user_input)
        inferred_mode = normalize_enum(
            result.get("interaction_mode"), INTERACTION_MODES, "direct"
        )
        interaction_mode = explicit_mode or inferred_mode
        interaction_source = "explicit" if explicit_mode else "inferred"
        interaction_confidence = 1.0 if explicit_mode else clamp_confidence(
            result.get("interaction_mode_confidence"), 0.7
        )

        task_type = normalize_enum(result.get("task_type"), TASK_TYPES, "other")
        if task_type == "other" and task_type_hint in TASK_TYPES:
            task_type = str(task_type_hint)
            task_source = "deterministic"
            task_confidence = 0.8
        else:
            task_source = "inferred"
            task_confidence = clamp_confidence(result.get("task_type_confidence"), 0.7)

        complexity_indicators = result.get("complexity_indicators") or {}
        if not isinstance(complexity_indicators, dict):
            complexity_indicators = {}
        complexity_count = self._bool_count(complexity_indicators)
        complexity = self._level_from_count(complexity_count, low_max=1, medium_max=3)
        complexity = normalize_enum(complexity, COMPLEXITY_LEVELS, "medium")

        risk_indicators = result.get("risk_indicators") or {}
        if not isinstance(risk_indicators, dict):
            risk_indicators = {}
        risk_count = self._bool_count(risk_indicators)
        risk_level = self._level_from_count(risk_count, low_max=0, medium_max=2)
        risk_level = normalize_enum(risk_level, RISK_LEVELS, "low")

        project_name = str(result.get("project_name", "unknown") or "unknown").strip()
        project_stage = str(result.get("project_stage", "unknown") or "unknown").strip()
        explicit_instruction = str(result.get("explicit_instruction", "") or "").strip()
        scope_bounded = self._scope_bounded(user_input)
        explicit_instruction_present = bool(explicit_instruction)
        cross_module = bool(complexity_indicators.get("cross_module", False))
        multi_step = bool(complexity_indicators.get("multi_step", False))
        tradeoff = bool(complexity_indicators.get("tradeoff", False))
        ambiguity = bool(complexity_indicators.get("ambiguity", False))
        long_term_state = bool(complexity_indicators.get("long_term_state", False))
        external_tool = bool(complexity_indicators.get("external_tool", False))

        field_meta = {
            "task_type": self._field_meta(task_source, task_confidence),
            "complexity": self._field_meta("deterministic", 0.85),
            "risk_level": self._field_meta("deterministic", 0.85),
            "interaction_mode": self._field_meta(
                interaction_source, interaction_confidence
            ),
            "explicit_instruction": self._field_meta(
                "explicit" if explicit_instruction else "inferred",
                1.0 if explicit_instruction else 0.4,
            ),
            "project_name": self._field_meta(
                "inferred", result.get("project_name_confidence", 0.6)
            ),
            "project_stage": self._field_meta(
                "inferred", result.get("project_stage_confidence", 0.6)
            ),
            "cross_module": self._field_meta("deterministic", 0.9),
            "multi_step": self._field_meta("deterministic", 0.9),
            "tradeoff": self._field_meta("deterministic", 0.9),
            "ambiguity": self._field_meta("deterministic", 0.85),
            "long_term_state": self._field_meta("deterministic", 0.85),
            "scope_bounded": self._field_meta("explicit", 1.0),
            "explicit_instruction_present": self._field_meta(
                "explicit" if explicit_instruction_present else "inferred",
                1.0 if explicit_instruction_present else 0.4,
            ),
            "available_skills": self._field_meta("deterministic", 1.0),
            "available_tools": self._field_meta("deterministic", 1.0),
            "model": self._field_meta("deterministic", 1.0),
            "relevant_memory_count": self._field_meta("deterministic", 1.0),
            "memory_quality": self._field_meta("deterministic", 0.9),
        }

        important_confidences = [
            field_meta["task_type"]["confidence"],
            field_meta["complexity"]["confidence"],
            field_meta["interaction_mode"]["confidence"],
            field_meta["project_name"]["confidence"],
        ]
        context_confidence = sum(important_confidences) / len(important_confidences)

        context = {
            "task_type": task_type,
            "complexity": complexity,
            "risk_level": risk_level,
            "interaction_mode": interaction_mode,
            "explicit_instruction": explicit_instruction,
            "explicit_instruction_present": explicit_instruction_present,
            "scope_bounded": scope_bounded,
            "project_name": project_name or "unknown",
            "project_stage": project_stage or "unknown",
            "cross_module": cross_module,
            "multi_step": multi_step,
            "tradeoff": tradeoff,
            "ambiguity": ambiguity,
            "long_term_state": long_term_state,
            "available_skills": list(available_skills),
            "available_tools": list(available_tools or []),
            "model": self.model_name,
            "relevant_memory_count": len(relevant_memories),
            "memory_quality": max(0.0, min(1.0, float(memory_quality))),
            "external_tool": external_tool,
            "complexity_indicators": complexity_indicators,
            "risk_indicators": risk_indicators,
            "field_meta": field_meta,
            "context_confidence": round(context_confidence, 4),
        }
        applicability_context, execution_context = split_context(context)
        context["applicability_context"] = applicability_context
        context["execution_context"] = execution_context
        return context
