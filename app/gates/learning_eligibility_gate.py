from __future__ import annotations

import re
from typing import Any

from app.cli_commands import looks_like_operational_input


LEARNING_ELIGIBILITY_PROMPT = """
MODULE: LEARNING_ELIGIBILITY_GATE
你是 EvoMe 的学习资格门。你的任务不是评价回答质量，而是决定这次交互是否有资格进入长期学习系统。

分类只能是：natural / operational / noise / testing / meta。

核心原则：
1. 只有与“用户/用户世界的真实长期信息”或“EvoMe 实际执行结果、明确反馈、可复用任务经验”有关的真实交互，才有学习资格。
2. 普通知识问答即使内容正确、完整，也通常没有长期学习价值。
3. operational：命令、状态查看、运维操作。默认只审计，不学习。
4. noise：乱码、误按、无意义输入。只审计，不学习。
5. testing：假设数据、模拟用户事实、仅用于测试的输入。默认只审计；若用户明确要求“把测试结果作为系统经验记录”，只允许 evolution，不允许把模拟事实写入 PersonalMemory。
6. meta：只是询问 EvoMe 当前记忆、Hypothesis、状态、内部概念或观察系统本身。默认只审计。真正的设计/修复/实现任务不属于 meta。
7. 用户明确纠正 EvoMe、描述真实偏好/目标/项目/决策，或执行真实复杂任务，可以进入相应学习通道。
8. “以后/今后给我……”“这样我更容易理解”“以后默认……”这类明确的未来交互偏好，同时属于用户长期偏好和 EvoMe 执行反馈：allow_personal_memory=true 且 allow_evolution=true。

两个通道必须分开：
- allow_personal_memory：是否允许 PersonalMemory Gate 从当前用户输入提取真实长期用户信息。
- allow_evolution：是否允许 Evaluation → Experience → Reflection → Hypothesis/Revision 链。

严格 JSON：
{
  "category": "natural",
  "allow_personal_memory": false,
  "allow_evolution": false,
  "reason": ""
}
"""


class LearningEligibilityGate:
    """Hybrid, fail-closed learning eligibility boundary.

    Deterministic rules handle obvious operational/noise/testing/meta/user-fact
    cases. Ambiguous natural language is delegated to the LLM. Invalid LLM
    output fails closed so an uncertain turn cannot silently contaminate
    long-term memory/evolution state.
    """

    CATEGORIES = {"natural", "operational", "noise", "testing", "meta"}

    _TESTING_MARKERS = (
        "这只是测试",
        "只是测试",
        "不是真的",
        "测试数据",
        "测试记忆",
        "测试一下记忆",
        "模拟一个用户",
        "模拟用户",
        "模拟一下",
        "假设我",
        "dummy data",
        "mock data",
        "synthetic data",
        "for testing only",
        "test only",
    )
    _TESTING_OPT_IN_MARKERS = (
        "把这次测试结果作为经验",
        "把测试结果作为经验",
        "记录这次测试",
        "把这次测试记录下来",
        "把测试记录下来",
        "作为学习样本",
        "纳入学习",
        "作为系统经验",
    )

    _CORRECTION_OR_FEEDBACK_MARKERS = (
        "你刚才错",
        "你刚才讲错",
        "你理解错",
        "你刚才有问题",
        "刚才那个方案有问题",
        "不对",
        "不是这样",
        "需要修正",
        "以后回答",
        "以后先",
        "我更喜欢",
        "这样更容易",
        "这样更清楚",
        "这样不好",
        "太啰嗦",
        "太简略",
    )

    _DURABLE_PREFERENCE_MARKERS = (
        "以后给我",
        "你以后给我",
        "今后给我",
        "以后帮我",
        "今后帮我",
        "以后默认",
        "今后默认",
        "默认按",
        "默认先",
        "我更容易理解",
        "这样我更容易",
        "这样更适合我",
        "对我更有帮助",
        "以后都",
        "今后都",
    )

    _PERSONAL_WORLD_MARKERS = (
        "我喜欢",
        "我不喜欢",
        "我更喜欢",
        "我的目标",
        "我的计划",
        "我计划",
        "我打算",
        "我决定",
        "我正在",
        "我已经",
        "我目前",
        "我的项目",
        "我希望以后",
        "以后请",
        "对我来说",
        "记住我",
    )

    _DESIGN_EXECUTION_MARKERS = (
        "设计",
        "实现",
        "修复",
        "排查",
        "定位",
        "重构",
        "规划",
        "分析",
        "比较",
        "检查",
        "划分",
        "梳理",
        "落地",
        "做一下",
        "继续做",
        "帮我改",
        "复杂任务",
        "报错",
        "错误",
        "异常",
        "bug",
        "traceback",
        "运行失败",
    )

    _META_SUBJECT_MARKERS = (
        "evome",
        "memory",
        "personal memory",
        "experience",
        "reflection",
        "hypothesis",
        "revision signal",
        "revision-signal",
        "revision_signal",
        "candidate revision",
        "revision pressure",
        "你记住",
        "你的记忆",
    )
    _META_OBSERVE_MARKERS = (
        "你现在",
        "当前",
        "有哪些",
        "查看",
        "显示",
        "状态",
        "是什么",
        "什么意思",
        "为什么",
        "记住了什么",
    )

    _ORDINARY_QA_MARKERS = (
        "是什么",
        "什么意思",
        "怎么用",
        "怎么理解",
        "解释一下",
        "讲一下",
        "介绍一下",
        "为什么",
        "有什么区别",
        "英文怎么说",
        "怎么翻译",
    )

    _CHITCHAT_MARKERS = (
        "你好",
        "hello",
        "hi",
        "在吗",
        "谢谢",
        "晚安",
        "早上好",
    )

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _contains_any(cls, text: str, markers: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in markers)

    @classmethod
    def _is_noise(cls, text: str) -> bool:
        normalized = cls._normalize(text)
        if not normalized:
            return True
        compact = re.sub(r"\s+", "", normalized)
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", compact):
            return True
        if len(compact) >= 4 and len(set(compact)) == 1:
            return True
        # Common accidental keyboard mashing. Real words/sentences are not
        # classified here; low-value natural chat is handled separately.
        if re.fullmatch(r"[A-Za-z0-9]{5,}", compact):
            vowels = sum(ch.lower() in "aeiou" for ch in compact if ch.isalpha())
            if vowels == 0:
                return True
        return False

    @classmethod
    def _looks_structural(cls, text: str) -> bool:
        return cls._contains_any(text, cls._DESIGN_EXECUTION_MARKERS)

    @classmethod
    def _looks_meta_observation(cls, text: str) -> bool:
        if cls._looks_structural(text):
            return False
        return cls._contains_any(text, cls._META_SUBJECT_MARKERS) and cls._contains_any(
            text, cls._META_OBSERVE_MARKERS
        )

    @classmethod
    def _looks_ordinary_qa(cls, text: str) -> bool:
        if cls._looks_structural(text):
            return False
        if cls._contains_any(text, cls._PERSONAL_WORLD_MARKERS):
            return False
        return cls._contains_any(text, cls._ORDINARY_QA_MARKERS)

    @classmethod
    def _looks_durable_preference(cls, text: str) -> bool:
        if cls._contains_any(text, cls._DURABLE_PREFERENCE_MARKERS):
            return True
        lowered = text.lower()
        # Conservative future-default pattern: require both a future marker and
        # a user-facing reference so generic statements about the future do not
        # become PersonalMemory.
        has_future = ("以后" in lowered) or ("今后" in lowered)
        has_user_reference = any(marker in lowered for marker in ("给我", "帮我", "我更", "对我", "我希望"))
        return has_future and has_user_reference

    def _deterministic_decision(self, user_input: str) -> dict[str, Any] | None:
        text = self._normalize(user_input)

        if looks_like_operational_input(text):
            return self._result(
                category="operational",
                allow_personal_memory=False,
                allow_evolution=False,
                reason="deterministic: operational/CLI input",
                decision_source="deterministic_hard_skip",
            )

        if self._is_noise(text):
            return self._result(
                category="noise",
                allow_personal_memory=False,
                allow_evolution=False,
                reason="deterministic: noise or accidental input",
                decision_source="deterministic_hard_skip",
            )

        # Explicit opt-in applies only to execution/test experience. Synthetic
        # user facts must still never enter PersonalMemory.
        if self._contains_any(text, self._TESTING_MARKERS):
            opt_in = self._contains_any(text, self._TESTING_OPT_IN_MARKERS)
            return self._result(
                category="testing",
                allow_personal_memory=False,
                allow_evolution=opt_in,
                reason=(
                    "deterministic: testing input explicitly opted into evolution"
                    if opt_in
                    else "deterministic: testing/simulated input is non-learning by default"
                ),
                decision_source=(
                    "deterministic_hard_pass" if opt_in else "deterministic_hard_skip"
                ),
            )

        has_personal = self._contains_any(text, self._PERSONAL_WORLD_MARKERS)
        has_feedback = self._contains_any(text, self._CORRECTION_OR_FEEDBACK_MARKERS)
        durable_preference = self._looks_durable_preference(text)
        if durable_preference:
            return self._result(
                category="natural",
                allow_personal_memory=True,
                allow_evolution=True,
                reason="deterministic: explicit durable preference + execution feedback; route to both lanes",
                decision_source="deterministic_hard_pass",
            )
        if has_feedback:
            return self._result(
                category="natural",
                allow_personal_memory=bool(has_personal or durable_preference),
                allow_evolution=True,
                reason=(
                    "deterministic: explicit durable preference + execution feedback; route to both lanes"
                    if durable_preference
                    else "deterministic: explicit real feedback/correction has learning value"
                ),
                decision_source="deterministic_hard_pass",
            )

        # Pure user/world facts need the PersonalMemory lane, but do not by
        # themselves justify Experience/Reflection.
        if has_personal and not self._looks_structural(text):
            return self._result(
                category="natural",
                allow_personal_memory=True,
                allow_evolution=False,
                reason="deterministic: real user/world information; memory lane only",
                decision_source="deterministic_hard_pass",
            )

        if self._looks_meta_observation(text):
            return self._result(
                category="meta",
                allow_personal_memory=False,
                allow_evolution=False,
                reason="deterministic: self-observation/meta query is audit-only",
                decision_source="deterministic_hard_skip",
            )

        if self._contains_any(text, self._CHITCHAT_MARKERS) and len(text) <= 30:
            return self._result(
                category="natural",
                allow_personal_memory=False,
                allow_evolution=False,
                reason="deterministic: chitchat has no durable learning signal",
                decision_source="deterministic_hard_skip",
            )

        if self._looks_ordinary_qa(text):
            return self._result(
                category="natural",
                allow_personal_memory=False,
                allow_evolution=False,
                reason="deterministic: ordinary knowledge Q&A is audit-only",
                decision_source="deterministic_hard_skip",
            )

        if self._looks_structural(text):
            return self._result(
                category="natural",
                allow_personal_memory=has_personal,
                allow_evolution=True,
                reason="deterministic: real design/debug/planning execution may yield reusable experience",
                decision_source="deterministic_hard_pass",
            )

        return None

    @staticmethod
    def _result(
        *,
        category: str,
        allow_personal_memory: bool,
        allow_evolution: bool,
        reason: str,
        decision_source: str,
    ) -> dict[str, Any]:
        return {
            "category": category,
            "learning_eligible": bool(allow_personal_memory or allow_evolution),
            "allow_personal_memory": bool(allow_personal_memory),
            "allow_evolution": bool(allow_evolution),
            "reason": reason,
            "decision_source": decision_source,
        }

    def classify(self, user_input: str) -> dict[str, Any]:
        deterministic = self._deterministic_decision(user_input)
        if deterministic is not None:
            return deterministic

        result = self.llm.json_chat(
            [
                {"role": "system", "content": LEARNING_ELIGIBILITY_PROMPT},
                {"role": "user", "content": str(user_input or "")},
            ],
            temperature=0.1,
        )
        if not isinstance(result, dict):
            result = {}

        category = str(result.get("category", "")).strip().lower()
        if category not in self.CATEGORIES:
            return self._result(
                category="natural",
                allow_personal_memory=False,
                allow_evolution=False,
                reason="llm classifier returned invalid output; fail-closed",
                decision_source="llm_fail_closed",
            )

        allow_memory = bool(result.get("allow_personal_memory", False))
        allow_evolution = bool(result.get("allow_evolution", False))

        # Category invariants override permissive/malformed model output.
        if category in {"operational", "noise", "meta"}:
            allow_memory = False
            allow_evolution = False
        elif category == "testing":
            allow_memory = False
            if not self._contains_any(user_input, self._TESTING_OPT_IN_MARKERS):
                allow_evolution = False

        return self._result(
            category=category,
            allow_personal_memory=allow_memory,
            allow_evolution=allow_evolution,
            reason=str(result.get("reason", "")).strip() or "LLM eligibility decision",
            decision_source="llm",
        )
