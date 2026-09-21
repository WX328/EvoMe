from __future__ import annotations
from typing import Any
from app.hypothesis.signal_quality import normalize_revision_signal

RELATIONS = {"SUPPORTS", "REFINES", "CONFLICTS", "UNRELATED"}

RELATION_JUDGE_PROMPT = """
MODULE: HYPOTHESIS_RELATION_JUDGE
你是 EvoMe 的 Strategy Hypothesis Relation Judge。
判断“新的 Reflection”与“已有 Strategy Hypothesis 当前 Revision”的语义关系。

只能选择：SUPPORTS / REFINES / CONFLICTS / UNRELATED。
- SUPPORTS：当前 WHEN/DO/EXPECT 已覆盖新证据，不需要修改。
- REFINES：核心规律相关，但新证据要求扩展、收窄或修正当前 WHEN/DO/EXPECT。
- CONFLICTS：相近 Context 下出现反证。
- UNRELATED：不是同一行为规律。

当且仅当 relation=REFINES 时输出 revision_signals。每个 Signal 只表达一个局部变化，禁止把多个动作或多个 Outcome 塞进同一 Signal。

固定 target_field taxonomy：
CONTEXT: task_type / complexity / risk_level / interaction_mode / project_stage / cross_module / multi_step / tradeoff / ambiguity / long_term_state / explicit_instruction_present / scope_bounded
ACTION: workflow_structure / response_structure / scope_alignment / verification_step / tool_usage / memory_usage / boundary_handling / data_flow_handling / error_handling
OUTCOME: clarity / executability / correctness / efficiency / boundary_consistency / scope_drift / responsibility_confusion / evaluation_performance / user_satisfaction
ABSTRACTION: context_scope / action_scope / outcome_scope / hypothesis_scope

禁止使用 applicable_context、do_data.text、expect_data.text 等宽泛字段。
禁止把 available_skills / available_tools / model / memory_quality / relevant_memory_count 当作 CONTEXT Revision，它们属于 Execution Context。

operation 必须同时符合 dimension 的兼容矩阵：
- CONTEXT: EXPAND / NARROW / ADD_CONDITION / REMOVE_CONDITION / CHANGE
- ACTION: EXPAND / CHANGE / GENERALIZE / SPECIALIZE
- OUTCOME: EXPAND / NARROW / CHANGE
- ABSTRACTION: GENERALIZE / SPECIALIZE
例如“给执行流程增加一步”属于 ACTION/EXPAND，不要写 ACTION/ADD_CONDITION。
before_data 写当前 Hypothesis 里的现有值；observed_data 只写这条 Reflection 真正观察到的新信息；不得提前生成最终 after。
user_satisfaction 只有存在明确用户反馈时才允许输出；不得从 evaluator score 推断。
最多输出 5 条原子 Signal。relation 不是 REFINES 时 revision_signals 必须为空数组。

严格 JSON：
{"relation":"REFINES","confidence":0.9,"reason":"","revision_signals":[{"dimension":"CONTEXT","operation":"EXPAND","target_field":"complexity","before_data":["high"],"observed_data":["medium"],"canonical_change":"complexity: high -> include observed medium","confidence":0.88}]}
"""

def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try: parsed=float(value)
    except (TypeError,ValueError): parsed=default
    return max(0.0,min(1.0,parsed))

class HypothesisRelationJudge:
    def __init__(self,llm)->None: self.llm=llm
    def judge(self,*,reflection_payload:dict[str,Any],hypothesis_payload:dict[str,Any])->dict[str,Any]:
        result=self.llm.json_chat([
            {"role":"system","content":RELATION_JUDGE_PROMPT},
            {"role":"user","content":f"New Reflection：\n{reflection_payload}\n\nExisting Hypothesis：\n{hypothesis_payload}"},
        ],temperature=0.0)
        relation=str(result.get("relation","UNRELATED")).strip().upper()
        if relation not in RELATIONS: relation="UNRELATED"
        raw_signals=result.get("revision_signals")
        if raw_signals is None and result.get("revision_signal") is not None: raw_signals=[result.get("revision_signal")]
        if not isinstance(raw_signals,list): raw_signals=[]
        revision_signals=[]; rejected=[]
        if relation=="REFINES":
            for item in raw_signals[:5]:
                normalized,reason=normalize_revision_signal(item,reflection_payload=reflection_payload,hypothesis_payload=hypothesis_payload)
                if normalized is not None: revision_signals.append(normalized)
                else: rejected.append(reason)
        return {"relation":relation,"confidence":_clamp_confidence(result.get("confidence",0.5)),"reason":str(result.get("reason","")).strip(),"revision_signals":revision_signals,"rejected_revision_signal_reasons":rejected}
