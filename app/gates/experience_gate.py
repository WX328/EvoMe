from __future__ import annotations

import re
from typing import Any

from app.prompt_utils import compact_for_prompt


EXPERIENCE_GATE_PROMPT = """
你是 EvoMe 的 Experience Gate。

你的职责是判断“EvoMe 这次执行本身”是否具有长期学习价值，是否值得保存为 Experience。
这里判断的是 Learning Value，不是“用户的问题重不重要”，也不是“这次回答是否成功”。

值得保存：
- 失败/部分失败、用户纠正、明显低分结果；
- 可复用的成功执行模式；
- 多步骤、跨模块、有明显权衡的设计/调试/规划任务；
- Strategy/Skill 有意义地参与并产生了可评价结果；
- 与历史经验冲突，或可能改变后续 Strategy/Skill 的执行经验。

通常不保存：
- 普通定义问答；
- 低价值闲聊；
- 简单、重复、没有新学习信号的普通成功；
- 只是在陈述用户事实、但没有形成 EvoMe 执行经验的交互。

注意：
1. Experience 可以保存，即使后续 Reflection 最终没有产生新 Insight。
2. “成功”不等于“值得保存”；“失败”也可能具有很高学习价值。
3. 不要因为回答看起来完整，就自动 should_save=true。

严格 JSON：
{
  "should_save": true,
  "task_type": "general",
  "result_summary": "",
  "reason": ""
}
"""


class ExperienceGate:
    """Hybrid Experience Gate.

    Clear high-value/low-value cases are handled deterministically so that
    evolution is stable and cheap. Only ambiguous cases are delegated to LLM.
    """

    _CORRECTION_MARKERS = (
        "不对",
        "错了",
        "不是这样",
        "你理解错",
        "有问题",
        "不行",
        "重新来",
        "重新分析",
        "需要修正",
    )

    _ARCHITECTURE_TERMS = (
        "架构",
        "模块",
        "组件",
        "职责",
        "边界",
        "数据流",
        "调用流程",
        "读写流程",
        "反模式",
        "分层",
        "存储层",
        "memory",
        "strategy",
        "skill",
        "context builder",
        "reflection",
        "hypothesis",
        "aggregator",
    )

    _DESIGN_ACTION_MARKERS = (
        "设计",
        "分析",
        "检查",
        "划分",
        "梳理",
        "讨论",
        "比较",
        "明确",
        "怎么流动",
        "怎么调用",
        "如何流转",
    )

    _DEBUG_PROBLEM_MARKERS = (
        "报错",
        "错误",
        "异常",
        "冲突",
        "失败",
        "bug",
        "崩溃",
        "traceback",
    )

    _DEBUG_ACTION_MARKERS = (
        "排查",
        "定位",
        "诊断",
        "修复",
        "解决",
        "为什么",
        "原因",
    )

    _PLANNING_MARKERS = (
        "项目规划",
        "路线图",
        "里程碑",
        "阶段规划",
        "实施计划",
        "落地计划",
    )

    _LOW_VALUE_MARKERS = (
        "是什么",
        "什么意思",
        "怎么翻译",
        "翻译成",
        "英文怎么说",
        "怎么读",
        "你好",
        "hello",
        "hi",
    )

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _contains_any(cls, text: str, markers: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in markers)

    @classmethod
    def _count_terms(cls, text: str, markers: tuple[str, ...]) -> int:
        lowered = text.lower()
        return sum(1 for marker in markers if marker.lower() in lowered)

    @classmethod
    def _infer_structural_task_type(cls, user_input: str) -> str | None:
        """Conservative deterministic task recognition for obvious cases only."""
        text = cls._normalize_text(user_input)

        architecture_term_count = cls._count_terms(text, cls._ARCHITECTURE_TERMS)
        has_design_action = cls._contains_any(text, cls._DESIGN_ACTION_MARKERS)
        if architecture_term_count >= 2 and has_design_action:
            return "architecture_design"

        has_debug_problem = cls._contains_any(text, cls._DEBUG_PROBLEM_MARKERS)
        has_debug_action = cls._contains_any(text, cls._DEBUG_ACTION_MARKERS)
        if has_debug_problem and has_debug_action:
            return "debugging"

        if cls._contains_any(text, cls._PLANNING_MARKERS):
            return "project_planning"

        return None

    @classmethod
    def _is_obvious_low_value(cls, user_input: str) -> bool:
        text = cls._normalize_text(user_input)
        if not text:
            return True

        # Only hard-skip very short, clearly definitional/chitchat requests.
        # Anything structurally rich remains eligible for LLM judgment.
        if len(text) > 80:
            return False
        if cls._infer_structural_task_type(text) is not None:
            return False
        return cls._contains_any(text, cls._LOW_VALUE_MARKERS)

    @staticmethod
    def _compact_summary(
        *,
        user_input: str,
        assistant_response: str,
        task_type: str,
        evaluation: dict[str, Any],
    ) -> str:
        user_text = ExperienceGate._normalize_text(user_input)
        response_text = ExperienceGate._normalize_text(assistant_response)
        reason = ExperienceGate._normalize_text(evaluation.get("reason", ""))

        user_preview = user_text[:140]
        response_preview = response_text[:300]
        parts = [f"[{task_type}] 用户任务：{user_preview}"]
        if response_preview:
            parts.append(f"执行结果：{response_preview}")
        if reason:
            parts.append(f"评估：{reason[:120]}")
        return "；".join(parts)

    def _deterministic_decision(
        self,
        *,
        user_input: str,
        assistant_response: str,
        evaluation: dict[str, Any],
    ) -> dict[str, Any] | None:
        score = max(0.0, min(1.0, float(evaluation.get("score", 0.5))))
        success = bool(evaluation.get("success", score >= 0.7))
        structural_task_type = self._infer_structural_task_type(user_input)

        # Meaningful failures are learning evidence even when the original task
        # itself was simple. Keep this rule before hard-skip.
        if (not success) or score < 0.65:
            task_type = structural_task_type or "general"
            return {
                "should_save": True,
                "task_type": task_type,
                "result_summary": self._compact_summary(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    task_type=task_type,
                    evaluation=evaluation,
                ),
                "reason": "deterministic: meaningful failure/low-score outcome has learning value",
                "decision_source": "deterministic_hard_pass",
            }

        # Explicit user correction is a strong learning signal.
        if self._contains_any(user_input, self._CORRECTION_MARKERS):
            task_type = structural_task_type or "general"
            return {
                "should_save": True,
                "task_type": task_type,
                "result_summary": self._compact_summary(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    task_type=task_type,
                    evaluation=evaluation,
                ),
                "reason": "deterministic: explicit correction/revision signal from user",
                "decision_source": "deterministic_hard_pass",
            }

        # Clear reusable structural work should not depend on a stochastic gate.
        if structural_task_type is not None:
            return {
                "should_save": True,
                "task_type": structural_task_type,
                "result_summary": self._compact_summary(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    task_type=structural_task_type,
                    evaluation=evaluation,
                ),
                "reason": (
                    "deterministic: structurally significant design/debug/planning "
                    "task has reusable execution value"
                ),
                "decision_source": "deterministic_hard_pass",
            }

        # Save tokens on only the clearest low-value cases. Everything else
        # remains ambiguous and is sent to the LLM gate.
        if self._is_obvious_low_value(user_input):
            return {
                "should_save": False,
                "task_type": "general",
                "result_summary": "",
                "reason": "deterministic: short definitional/chitchat interaction",
                "decision_source": "deterministic_hard_skip",
            }

        return None

    def evaluate(
        self,
        *,
        user_input: str,
        assistant_response: str,
        evaluation: dict[str, Any],
        strategy_name: str,
        skills_used: list[str],
    ) -> dict[str, Any]:
        deterministic = self._deterministic_decision(
            user_input=user_input,
            assistant_response=assistant_response,
            evaluation=evaluation,
        )
        if deterministic is not None:
            return deterministic

        result = self.llm.json_chat(
            [
                {"role": "system", "content": EXPERIENCE_GATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户输入：\n{user_input}\n\n"
                        f"EvoMe 回复：\n{compact_for_prompt(assistant_response)}\n\n"
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
            "decision_source": "llm",
        }
