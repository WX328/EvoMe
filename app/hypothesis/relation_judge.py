from __future__ import annotations

import json
from typing import Any

from app.hypothesis.signal_quality import normalize_revision_signal

RELATIONS = {"SUPPORTS", "REFINES", "CONFLICTS", "UNRELATED"}

BATCH_RELATION_JUDGE_PROMPT = """
MODULE: HYPOTHESIS_RELATION_JUDGE
Compare one new Reflection with the current revisions of several Strategy Hypotheses.
Return one compact decision per hypothesis. Do not explain outside JSON.

Relations:
- SUPPORTS: current WHEN/DO/EXPECT already covers the evidence.
- REFINES: same core rule, but evidence requires a local WHEN/DO/EXPECT change.
- CONFLICTS: counter-evidence in a comparable context.
- UNRELATED: different behavioral rule.

A hypothesis may include pending_signals. They are validated active-open evidence
attached to the current revision, but they are NOT applied WHEN/DO/EXPECT truth.
Use them only as semantic-continuity evidence. If the Reflection repeats or
strengthens a pending change, prefer the same hypothesis over UNRELATED and
normally emit a new evidence-grounded REFINES signal. Never return SUPPORTS only
because a pending signal already describes the behavior.

If relation=REFINES, revision_signals MUST contain 1-5 atomic, evidence-grounded local changes.
If you cannot formulate at least one valid local change, choose SUPPORTS or UNRELATED instead; never emit REFINES with an empty revision_signals array.
Otherwise revision_signals must be []. Maximum 5 signals per hypothesis.

Stable target_field taxonomy:
CONTEXT: task_type, complexity, risk_level, interaction_mode, project_stage, cross_module, multi_step, tradeoff, ambiguity, long_term_state, explicit_instruction_present, scope_bounded
ACTION: workflow_structure, response_structure, scope_alignment, verification_step, tool_usage, memory_usage, boundary_handling, data_flow_handling, error_handling
OUTCOME: clarity, executability, correctness, efficiency, boundary_consistency, scope_drift, responsibility_confusion, evaluation_performance, user_satisfaction
ABSTRACTION: context_scope, action_scope, outcome_scope, hypothesis_scope

Operation compatibility:
CONTEXT=EXPAND/NARROW/ADD_CONDITION/REMOVE_CONDITION/CHANGE;
ACTION=EXPAND/CHANGE/GENERALIZE/SPECIALIZE;
OUTCOME=EXPAND/NARROW/CHANGE;
ABSTRACTION=GENERALIZE/SPECIALIZE.

Do not turn execution facts (skills/tools/model/memory quality) into applicability context.
Do not expand complexity/risk/interaction_mode merely because one turn happened under that condition.
An explicit user preference about answer/order/format is normally an ACTION change; use response_structure when the evidence is specifically about presentation order or answer structure.
user_satisfaction requires explicit user feedback. before_data must reflect the current hypothesis;
observed_data must contain only newly observed evidence; never invent final after-data.
Keep reason under 60 Chinese characters or equivalent.

Strict JSON:
{"decisions":[{"hypothesis_id":1,"relation":"SUPPORTS","confidence":0.9,"reason":"","revision_signals":[]}]}
"""


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Remove provenance/telemetry noise before sending hypothesis semantics to the LLM."""
    if depth >= 4:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= 400 else value[:397] + "..."
    if isinstance(value, list):
        return [_compact_value(item, depth=depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        ignored = {
            "source_signal_ids",
            "provenance",
            "provenance_data",
            "field_meta",
            "metadata",
            "raw_evaluator_output",
        }
        result: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in ignored:
                continue
            result[str(key)] = _compact_value(item, depth=depth + 1)
            if len(result) >= 12:
                break
        return result
    return value


def _compact_reflection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "insight": _compact_value(payload.get("insight", "")),
        "context": _compact_value(payload.get("applicable_context") or {}),
        "change": _compact_value(payload.get("proposed_change", "")),
        "effect": _compact_value(payload.get("expected_effect", "")),
        "user_feedback": _compact_value(payload.get("user_feedback", "")),
    }


def _compact_hypothesis(payload: dict[str, Any]) -> dict[str, Any]:
    pending = payload.get("pending_signals") or []
    if not isinstance(pending, list):
        pending = []
    return {
        "id": payload.get("id"),
        "revision": payload.get("revision"),
        "when": _compact_value(
            payload.get("when_data") or payload.get("applicable_context") or {}
        ),
        "do": _compact_value(
            payload.get("do_data")
            if payload.get("do_data") is not None
            else payload.get("proposed_change", "")
        ),
        "expect": _compact_value(
            payload.get("expect_data")
            if payload.get("expect_data") is not None
            else payload.get("expected_effect", "")
        ),
        "pending_signals": [
            _compact_value(item)
            for item in pending[:3]
            if isinstance(item, dict)
        ],
    }


class HypothesisRelationJudge:
    def __init__(self, llm) -> None:
        self.llm = llm

    @staticmethod
    def _normalize_decision(
        raw: dict[str, Any],
        *,
        reflection_payload: dict[str, Any],
        hypothesis_payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_relation = str(raw.get("relation", "")).strip().upper()
        relation_label_valid = raw_relation in RELATIONS
        relation = raw_relation
        if not relation_label_valid:
            relation = "UNRELATED"

        raw_signals = raw.get("revision_signals")
        if raw_signals is None and raw.get("revision_signal") is not None:
            raw_signals = [raw.get("revision_signal")]
        if not isinstance(raw_signals, list):
            raw_signals = []

        revision_signals: list[dict[str, Any]] = []
        rejected: list[str] = []
        if relation == "REFINES":
            for item in raw_signals[:5]:
                normalized, reason = normalize_revision_signal(
                    item,
                    reflection_payload=reflection_payload,
                    hypothesis_payload=hypothesis_payload,
                )
                if normalized is not None:
                    revision_signals.append(normalized)
                else:
                    rejected.append(reason)

        decision_valid = relation_label_valid and not (
            relation == "REFINES" and not revision_signals
        )
        invalid_reason = ""
        if not decision_valid:
            if not relation_label_valid:
                invalid_reason = f"invalid relation label: {raw_relation or '<empty>'}"
            else:
                invalid_reason = (
                    "REFINES requires at least one valid atomic revision_signal"
                )
                if rejected:
                    invalid_reason += ": " + "; ".join(rejected[:3])

        return {
            "hypothesis_id": hypothesis_payload.get("id"),
            "relation": relation,
            "confidence": _clamp_confidence(raw.get("confidence", 0.5)),
            "reason": str(raw.get("reason", "")).strip(),
            "revision_signals": revision_signals,
            "rejected_revision_signal_reasons": rejected,
            "decision_valid": decision_valid,
            "invalid_reason": invalid_reason,
        }

    def judge_batch(
        self,
        *,
        reflection_payload: dict[str, Any],
        hypothesis_payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not hypothesis_payloads:
            return []

        payload_by_id = {
            int(item["id"]): item
            for item in hypothesis_payloads
            if item.get("id") is not None
        }
        prompt_payload = {
            "reflection": _compact_reflection(reflection_payload),
            "hypotheses": [
                _compact_hypothesis(item)
                for item in hypothesis_payloads
                if item.get("id") is not None
            ],
        }
        result = self.llm.json_chat(
            [
                {"role": "system", "content": BATCH_RELATION_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        prompt_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            temperature=0.0,
        )
        if not isinstance(result, dict):
            result = {}

        raw_decisions = result.get("decisions")
        # Backward-compatible adapter for tests/custom LLMs that still return
        # the old single-candidate shape.
        if not isinstance(raw_decisions, list):
            if len(payload_by_id) == 1 and "relation" in result:
                only_id = next(iter(payload_by_id))
                raw_decisions = [{**result, "hypothesis_id": only_id}]
            else:
                raw_decisions = []

        decisions: list[dict[str, Any]] = []
        seen: set[int] = set()
        for raw in raw_decisions:
            if not isinstance(raw, dict):
                continue
            try:
                hypothesis_id = int(raw.get("hypothesis_id"))
            except (TypeError, ValueError):
                continue
            if hypothesis_id not in payload_by_id or hypothesis_id in seen:
                continue
            seen.add(hypothesis_id)
            decisions.append(
                self._normalize_decision(
                    raw,
                    reflection_payload=reflection_payload,
                    hypothesis_payload=payload_by_id[hypothesis_id],
                )
            )
        return decisions

    def judge(
        self,
        *,
        reflection_payload: dict[str, Any],
        hypothesis_payload: dict[str, Any],
    ) -> dict[str, Any]:
        # The legacy single-candidate API historically accepted semantic
        # payloads without a persisted hypothesis id. Give only that adapter a
        # private sentinel so the strict batch API can still correlate results.
        batch_payload = hypothesis_payload
        if hypothesis_payload.get("id") is None:
            batch_payload = {**hypothesis_payload, "id": 0}
        decisions = self.judge_batch(
            reflection_payload=reflection_payload,
            hypothesis_payloads=[batch_payload],
        )
        if decisions:
            result = dict(decisions[0])
            result.pop("hypothesis_id", None)
            return result
        return {
            "relation": "UNRELATED",
            "confidence": 0.0,
            "reason": "relation judge returned no valid decision",
            "revision_signals": [],
            "rejected_revision_signal_reasons": [],
            "decision_valid": False,
            "invalid_reason": "relation judge returned no valid decision",
        }
