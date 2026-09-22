from typing import Any
from app.context.schema import sanitize_applicable_context, split_context

REFLECTION_PROMPT = """
MODULE: REFLECTION
你是 EvoMe 的 Reflection 模块。只对已通过 Experience Gate 的 Experience 做局部、紧凑反思。

只输出一个 JSON 对象，禁止解释、Markdown、复述输入：
{"reflection_type":"success","observation":"","insight":"","applicable_context":{},"strategy_signal":"","skill_signal":"","expected_effect":"","confidence":0.8}

约束：
- reflection_type 只能是 failure / success / pattern / conflict。
- observation 最多 60 个汉字；只写发生了什么。
- insight 最多 80 个汉字；只写一条局部规律，不写永恒真理。
- strategy_signal 最多 80 个汉字；没有明确可执行调整时为空。
- skill_signal 最多 60 个汉字；没有明确能力调整时为空。
- expected_effect 最多 60 个汉字；没有 strategy_signal 时为空。
- applicable_context 只能从提供的 Applicability Context 选择必要字段；不要复制整个 Context。
- available_skills / available_tools / model / memory_quality 等 Execution Context 禁止进入 applicable_context。
- 不要把 AI 推测当成用户事实；不确定字段宁可省略。
"""

class ReflectionService:
    def __init__(self, llm): self.llm = llm
    def reflect(self, *, task_type: str, result_summary: str, evaluation: dict[str, Any], strategy_name: str, skills_used: list[str], context_data: dict[str, Any] | None = None) -> dict[str, Any]:
        applicability_context, execution_context = split_context(context_data or {})
        result = self.llm.json_chat([
            {"role":"system","content":REFLECTION_PROMPT},
            {"role":"user","content":f"Task Type：{task_type}\nStrategy：{strategy_name}\nSkills：{skills_used}\nApplicability Context：{applicability_context}\nExecution Context：{execution_context}\nResult：{result_summary}\nEvaluation：{evaluation}"},
        ], temperature=0.1)
        reflection_type = str(result.get("reflection_type", "success")).strip()
        if reflection_type not in {"failure","success","pattern","conflict"}: reflection_type="success"
        applicable_context = sanitize_applicable_context(result.get("applicable_context") or {})
        insight = str(result.get("insight", result.get("lesson", ""))).strip()
        try: confidence=float(result.get("confidence",0.8))
        except (TypeError,ValueError): confidence=0.8
        return {
            "reflection_type":reflection_type,"observation":str(result.get("observation","")).strip(),
            "insight":insight,"lesson":insight,"applicable_context":applicable_context,
            "strategy_signal":str(result.get("strategy_signal","")).strip(),"skill_signal":str(result.get("skill_signal","")).strip(),
            "expected_effect":str(result.get("expected_effect","")).strip(),"confidence":max(0.0,min(1.0,confidence)),
        }
