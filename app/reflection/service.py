from typing import Any
from app.context.schema import sanitize_applicable_context, split_context

REFLECTION_PROMPT = """
MODULE: REFLECTION
你是 EvoMe 的 Reflection 模块。
只对已经被 Experience Gate 判定为“有学习价值”的 Experience 进行反思。

目标：
1. observation：客观描述这次 Experience 发生了什么。
2. insight：从这次 Experience 中得到的局部规律，不写成永恒真理。
3. applicable_context：指出这条规律可能在哪些环境下成立，只能从给出的 Applicability Context 中选择字段。
4. strategy_signal：如果未来行为可能需要调整，写出可执行的行为信号；没有则为空。
5. skill_signal：如果能力层可能需要调整，写出能力信号；没有则为空。
6. expected_effect：如果 strategy_signal 被采用，预期改善什么；没有则为空。

严格区分：
- Applicability Context：什么时候策略适用，可以进入 applicable_context。
- Execution Context：available_skills / available_tools / model / memory_quality 等，仅描述运行条件，禁止写入 applicable_context。

允许 reflection_type：failure / success / pattern / conflict
不要把 AI 的猜测当成用户事实。不确定时只保留高置信字段。
严格 JSON：
{"reflection_type":"success","observation":"","insight":"","applicable_context":{},"strategy_signal":"","skill_signal":"","expected_effect":"","confidence":0.8}
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
