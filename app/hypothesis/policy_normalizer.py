from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context.schema import sanitize_applicable_context
from app.models import (
    Experience,
    HypothesisEvidence,
    HypothesisRevision,
    ProposalNormalizationRecord,
    ReflectionRecord,
    RevisionAnalysisRecord,
    RevisionProposal,
    RevisionSignal,
    StrategyHypothesis,
)


NORMALIZATION_VERSION = 2
CONDITIONAL_POLICY_KIND = "conditional_policy"
RAW_PROPOSAL_KIND = "raw"
SUPERSEDED_STATUS = "superseded_by_normalized_proposal"

EVIDENCE_PRECEDENCE = {
    "current_explicit_instruction": 1,
    "durable_explicit_preference": 2,
    "repeated_independent_experience": 3,
    "single_experience_pattern": 4,
    "model_inference": 5,
}


@dataclass(frozen=True)
class PolicyNormalizationResult:
    record: ProposalNormalizationRecord
    proposal: RevisionProposal | None
    llm_called: bool


class UnsupportedPolicyCondition(ValueError):
    """Raised when a generated condition cannot be traced to source evidence."""


class ProposalPolicyNormalizer:
    """Normalize ACTION/response_structure unions into evidence-backed policy.

    The normalizer deliberately receives compact signal/reflection context and
    provenance only.  It never loads InteractionRecord or chat history.  One
    A source proposal can produce at most one audit record per normalization
    version and therefore at most one normalization LLM call per version.
    """

    def __init__(self, db: Session, llm) -> None:
        self.db = db
        self.llm = llm

    @staticmethod
    def _compact_text(value: Any, limit: int = 360) -> str:
        text = str(value or "").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _parse_ids(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        result: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                continue
            if parsed not in result:
                result.append(parsed)
        return result

    @staticmethod
    def _is_target(proposal: RevisionProposal) -> bool:
        return (
            str(proposal.dimension).upper() == "ACTION"
            and str(proposal.target_field) == "response_structure"
            and str(proposal.proposal_kind or RAW_PROPOSAL_KIND) != CONDITIONAL_POLICY_KIND
        )

    def _source_signals(self, proposal: RevisionProposal) -> list[RevisionSignal]:
        ids = self._parse_ids(proposal.source_signal_ids or [])
        if not ids:
            return []
        rows = list(
            self.db.scalars(
                select(RevisionSignal).where(RevisionSignal.id.in_(ids))
            ).all()
        )
        by_id = {int(row.id): row for row in rows}
        return [by_id[signal_id] for signal_id in ids if signal_id in by_id]

    def _compact_signal(self, signal: RevisionSignal) -> dict[str, Any]:
        reflection = self.db.get(ReflectionRecord, int(signal.reflection_id))
        experience = None
        context: dict[str, Any] = {}
        reflection_signal = ""
        if reflection is not None:
            context = sanitize_applicable_context(reflection.applicable_context or {})
            reflection_signal = self._compact_text(
                reflection.canonical_signal
                or reflection.strategy_signal
                or reflection.insight
                or reflection.lesson
            )
            if reflection.experience_id is not None:
                experience = self.db.get(Experience, int(reflection.experience_id))
                if not context and experience is not None:
                    context = sanitize_applicable_context(experience.context_data or {})

        evidence = (
            self.db.get(HypothesisEvidence, int(signal.evidence_id))
            if signal.evidence_id is not None
            else None
        )
        provenance = dict(signal.provenance_data or {})
        explicit_feedback = bool(provenance.get("explicit_user_feedback"))
        if experience is not None and str(experience.user_feedback or "").strip():
            explicit_feedback = True
        semantic_text = str(
            f"{signal.observed_data or ''} {signal.canonical_change or ''} {reflection_signal}"
        ).lower()
        preference_signal = any(
            token in semantic_text
            for token in ("preference", "order", "sequence", "偏好", "顺序")
        )
        explicit_instruction = bool(
            provenance.get("explicit_user_instruction")
            or provenance.get("current_user_instruction")
            or context.get("explicit_instruction_present")
        )
        durable_preference = bool(
            provenance.get("durable_preference")
            or provenance.get("explicit_durable_preference")
            or provenance.get("preference_scope") == "durable"
            or (
                explicit_instruction
                and context.get("long_term_state")
                and preference_signal
            )
        )

        compact_provenance = dict(provenance)
        compact_provenance.update(
            {
                "explicit_user_feedback": explicit_feedback,
                "explicit_user_instruction": explicit_instruction,
                "durable_preference": durable_preference,
            }
        )

        return {
            "signal_id": int(signal.id),
            "dimension": str(signal.dimension).upper(),
            "operation": str(signal.operation).upper(),
            "target_field": str(signal.target_field),
            "reflection_id": int(signal.reflection_id),
            "evidence_id": int(signal.evidence_id) if signal.evidence_id is not None else None,
            "independent": bool(evidence.is_independent) if evidence is not None else False,
            "before_data": signal.before_data,
            "observed_data": signal.observed_data,
            "canonical_change": self._compact_text(signal.canonical_change),
            "confidence": float(signal.confidence),
            "compact_applicable_context": context,
            "reflection_signal": reflection_signal,
            "provenance": compact_provenance,
        }

    @staticmethod
    def _source_problem(
        proposal: RevisionProposal,
        requested_ids: list[int],
        signals: list[RevisionSignal],
    ) -> str:
        if not requested_ids:
            return "proposal_has_no_source_signals"
        if [int(item.id) for item in signals] != requested_ids:
            return "source_signal_missing"
        for signal in signals:
            if str(signal.resolution_status or "open").lower() != "open":
                return f"source_signal_{signal.id}_not_open"
            if int(signal.hypothesis_id) != int(proposal.hypothesis_id):
                return f"source_signal_{signal.id}_hypothesis_mismatch"
            if int(signal.base_revision) != int(proposal.base_revision):
                return f"source_signal_{signal.id}_revision_mismatch"
            if str(signal.dimension).upper() != "ACTION":
                return f"source_signal_{signal.id}_dimension_mismatch"
            if str(signal.target_field) != "response_structure":
                return f"source_signal_{signal.id}_target_mismatch"
        return ""

    @staticmethod
    def _has_current_explicit(items: list[dict[str, Any]]) -> bool:
        return any(
            bool(item["provenance"].get("explicit_user_instruction"))
            and not bool(item["provenance"].get("durable_preference"))
            for item in items
        )

    @staticmethod
    def _has_durable_explicit(items: list[dict[str, Any]]) -> bool:
        return any(
            bool(item["provenance"].get("durable_preference"))
            or bool(item["provenance"].get("explicit_user_feedback"))
            for item in items
        )

    @classmethod
    def _evidence_level(cls, items: list[dict[str, Any]]) -> tuple[str, int]:
        if cls._has_current_explicit(items):
            label = "current_explicit_instruction"
        elif cls._has_durable_explicit(items):
            label = "durable_explicit_preference"
        else:
            independent_sources = {
                (item.get("evidence_id"), item.get("reflection_id"))
                for item in items
                if item.get("independent")
            }
            label = (
                "repeated_independent_experience"
                if len(independent_sources) >= 2
                else "single_experience_pattern"
            )
        return label, EVIDENCE_PRECEDENCE[label]

    @classmethod
    def _condition_predicates(cls, condition: Any) -> list[tuple[str, Any, str]]:
        if not isinstance(condition, dict) or not condition:
            raise UnsupportedPolicyCondition("condition_must_be_non_empty_object")

        if "field" in condition:
            field = str(condition.get("field") or "").strip()
            operator = str(condition.get("operator") or "equals").strip().lower()
            if not field or operator not in {"equals", "in", "contains"}:
                raise UnsupportedPolicyCondition("unsupported_condition_predicate")
            return [(field, condition.get("value"), operator)]

        predicates: list[tuple[str, Any, str]] = []
        for key, value in condition.items():
            lowered = str(key).lower()
            if lowered in {"all", "any"}:
                if not isinstance(value, list) or not value:
                    raise UnsupportedPolicyCondition("empty_logical_condition")
                for child in value:
                    predicates.extend(cls._condition_predicates(child))
            elif lowered == "kind":
                predicates.append(("$kind", str(value), "equals"))
            else:
                predicates.append((str(key), value, "equals"))
        if not predicates:
            raise UnsupportedPolicyCondition("condition_has_no_predicates")
        return predicates

    @staticmethod
    def _value_supported(actual: Any, expected: Any, operator: str) -> bool:
        if operator == "equals":
            return actual == expected
        if operator == "in":
            return isinstance(expected, list) and actual in expected
        if operator == "contains":
            if isinstance(actual, (list, tuple, set)):
                return expected in actual
            if isinstance(actual, str) and isinstance(expected, str):
                return expected in actual
        return False

    @classmethod
    def _validate_condition(
        cls,
        condition: dict[str, Any],
        source_items: list[dict[str, Any]],
    ) -> int:
        predicates = cls._condition_predicates(condition)
        for field, expected, operator in predicates:
            if field == "$kind":
                kind = str(expected).strip().lower()
                source_texts = [
                    str(
                        f"{item.get('observed_data', '')} "
                        f"{item.get('canonical_change', '')} "
                        f"{item.get('reflection_signal', '')}"
                    ).lower()
                    for item in source_items
                ]
                if kind == "explicit_user_instruction" and cls._has_current_explicit(source_items):
                    continue
                if kind in {"explicit_user_preference", "durable_user_preference"} and (
                    cls._has_current_explicit(source_items)
                    or cls._has_durable_explicit(source_items)
                ) and any(
                    any(token in text for token in ("preference", "order", "sequence", "偏好", "顺序"))
                    for text in source_texts
                ):
                    continue
                raise UnsupportedPolicyCondition(f"unsupported_condition_kind:{kind}")

            if not any(
                field in (item.get("compact_applicable_context") or {})
                and cls._value_supported(
                    (item.get("compact_applicable_context") or {}).get(field), expected, operator
                )
                for item in source_items
            ):
                raise UnsupportedPolicyCondition(f"unsupported_condition:{field}")
        return len(predicates)

    @staticmethod
    def _normalize_effect(effect: Any) -> dict[str, list[Any]]:
        if not isinstance(effect, dict):
            raise UnsupportedPolicyCondition("effect_must_be_object")
        normalized: dict[str, list[Any]] = {}
        for key in ("replace_order", "append_sections", "add_checks"):
            value = effect.get(key, [])
            if value is None:
                value = []
            if not isinstance(value, list):
                raise UnsupportedPolicyCondition(f"effect_{key}_must_be_list")
            normalized[key] = value
        if not any(normalized.values()):
            raise UnsupportedPolicyCondition("rule_has_no_behavior_effect")
        return normalized

    def _normalize_policy(
        self,
        raw: dict[str, Any],
        compact_signals: list[dict[str, Any]],
        proposal_signal_ids: list[int],
    ) -> dict[str, Any]:
        if raw.get("cannot_normalize") is True:
            reason = str(raw.get("reason") or "llm_cannot_normalize").strip()
            raise UnsupportedPolicyCondition(f"llm_cannot_normalize:{reason}")
        base_order = raw.get("base_order")
        if not isinstance(base_order, dict):
            raise UnsupportedPolicyCondition("base_order_must_be_object")
        base_steps = base_order.get("steps")
        if not isinstance(base_steps, list) or not base_steps:
            raise UnsupportedPolicyCondition("base_order_steps_must_be_non_empty_list")

        by_id = {int(item["signal_id"]): item for item in compact_signals}
        allowed_ids = set(proposal_signal_ids)
        base_source_ids = self._parse_ids(base_order.get("source_signal_ids"))
        if not base_source_ids or not set(base_source_ids).issubset(allowed_ids):
            raise UnsupportedPolicyCondition("base_order_requires_source_signals")

        raw_rules = raw.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise UnsupportedPolicyCondition("conditional_policy_requires_rules")

        rules: list[dict[str, Any]] = []
        covered_ids = set(base_source_ids)
        seen_rule_keys: set[str] = set()
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise UnsupportedPolicyCondition(f"rule_{index}_must_be_object")
            source_ids = self._parse_ids(raw_rule.get("source_signal_ids"))
            if not source_ids or not set(source_ids).issubset(allowed_ids):
                raise UnsupportedPolicyCondition(f"rule_{index}_invalid_source_signal_ids")
            source_items = [by_id[item] for item in source_ids]
            condition = raw_rule.get("condition")
            predicate_count = self._validate_condition(condition, source_items)
            effect = self._normalize_effect(raw_rule.get("effect"))
            level, evidence_rank = self._evidence_level(source_items)
            # Higher priority wins. Evidence class dominates first, then a more
            # specific condition, then newer source evidence breaks exact ties.
            priority = (
                (6 - evidence_rank) * 100_000
                + min(predicate_count, 99) * 1_000
                + min(max(source_ids), 999)
            )
            normalized_rule = {
                "rule_type": str(raw_rule.get("rule_type") or "conditional_structure"),
                "condition": condition,
                "effect": effect,
                "priority": priority,
                "evidence_level": level,
                "evidence_rank": evidence_rank,
                "source_signal_ids": source_ids,
            }
            key = json.dumps(
                {"condition": condition, "effect": effect},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if key in seen_rule_keys:
                continue
            seen_rule_keys.add(key)
            rules.append(normalized_rule)
            covered_ids.update(source_ids)

        annotations: list[dict[str, Any]] = []
        raw_annotations = raw.get("annotations", [])
        if not isinstance(raw_annotations, list):
            raise UnsupportedPolicyCondition("annotations_must_be_list")
        for index, item in enumerate(raw_annotations):
            if not isinstance(item, dict):
                raise UnsupportedPolicyCondition(f"annotation_{index}_must_be_object")
            source_ids = self._parse_ids(item.get("source_signal_ids"))
            text = str(item.get("text") or "").strip()
            if not text or not source_ids or not set(source_ids).issubset(allowed_ids):
                raise UnsupportedPolicyCondition(f"annotation_{index}_is_not_evidence_backed")
            annotations.append({"text": text, "source_signal_ids": source_ids})
            covered_ids.update(source_ids)

        if covered_ids != allowed_ids:
            missing = sorted(allowed_ids - covered_ids)
            raise UnsupportedPolicyCondition(f"source_signals_not_represented:{missing}")

        rules.sort(key=lambda item: (-int(item["priority"]), item["rule_type"]))
        return {
            "policy_version": NORMALIZATION_VERSION,
            "base_order": {
                "steps": base_steps,
                "source_signal_ids": base_source_ids,
            },
            "rules": rules,
            "annotations": annotations,
            "source_signal_ids": proposal_signal_ids,
            "precedence": [
                "current_explicit_instruction",
                "durable_explicit_preference",
                "repeated_independent_experience",
                "single_experience_pattern",
                "model_inference",
            ],
        }

    def _judge(
        self,
        proposal: RevisionProposal,
        revision: HypothesisRevision,
        compact_signals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "current_revision": {
                "hypothesis_id": int(revision.hypothesis_id),
                "revision": int(revision.revision_number),
                "response_structure": (revision.do_data or {}).get("response_structure"),
            },
            "raw_proposal": {
                "proposal_id": int(proposal.id),
                "before": proposal.before_data,
                "proposed_after": proposal.proposed_after_data,
                "reason": self._compact_text(proposal.reason),
                "source_signal_ids": self._parse_ids(proposal.source_signal_ids or []),
            },
            "signals": compact_signals,
            "evidence_precedence": [
                "current explicit user instruction",
                "durable explicit user preference",
                "repeated independent Experience pattern",
                "single Experience pattern",
                "model inference",
            ],
        }
        output_example = {
            "base_order": {
                "steps": ["summary", "details"],
                "source_signal_ids": [101],
            },
            "rules": [
                {
                    "rule_type": "contextual_checks",
                    "condition": {
                        "field": "task_type",
                        "operator": "equals",
                        "value": "architecture_design",
                    },
                    "effect": {
                        "replace_order": [],
                        "append_sections": [],
                        "add_checks": ["boundaries"],
                    },
                    "priority": 100,
                    "source_signal_ids": [102],
                }
            ],
            "annotations": [
                {
                    "text": "mark historical assumptions",
                    "source_signal_ids": [103],
                }
            ],
            "reason": "each component cites only its local supporting signal",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Normalize a merged ACTION/response_structure proposal into a conditional "
                    "response policy. Do not union every observed section into an always-on list. "
                    "Return JSON only. base_order MUST be an object with non-empty steps and "
                    "non-empty source_signal_ids. Every conditional rule and every behavior "
                    "annotation MUST also have non-empty source_signal_ids. Every cited id must be "
                    "a subset of raw_proposal.source_signal_ids. Map evidence locally: do not attach "
                    "all proposal signal ids to every component merely because all signals target "
                    "response_structure. The input signals explicitly include signal_id, dimension, "
                    "operation, target_field, observed_data, canonical_change, provenance, and "
                    "compact_applicable_context. Put a historical default in base_order, explicit "
                    "order preferences in replace_order, and scenario-specific sections/checks in "
                    "conditional rules only when their cited evidence supports the condition. "
                    "Conditions may use only exact compact_applicable_context facts from cited "
                    "signals, or kind=explicit_user_instruction / explicit_user_preference when "
                    "cited provenance supports it. Never invent labels such as evidence_oriented "
                    "or boundary_analysis. If a conditional component lacks a supported condition, "
                    "omit it only when the remaining policy is semantically complete; otherwise "
                    "return {\"cannot_normalize\":true,\"reason\":\"...\"}. Every proposal "
                    "source signal must be represented by base_order, a rule, or a sourced "
                    "annotation. The ids in this example are illustrative; copy actual ids from "
                    "the input: "
                    + json.dumps(output_example, ensure_ascii=False, separators=(",", ":"))
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]
        result = self.llm.json_chat(messages, temperature=0.1)
        return result if isinstance(result, dict) else {}

    def _record_failure(
        self,
        proposal: RevisionProposal,
        source_ids: list[int],
        reason: str,
        *,
        raw_output: dict[str, Any] | None = None,
        llm_called: bool,
    ) -> PolicyNormalizationResult:
        record = ProposalNormalizationRecord(
            source_proposal_id=int(proposal.id),
            normalized_proposal_id=None,
            normalization_version=NORMALIZATION_VERSION,
            status="cannot_normalize",
            rule_count=0,
            source_signal_ids=source_ids,
            reason=reason,
            normalization_data={"raw_output": raw_output or {}, "error": reason},
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return PolicyNormalizationResult(record, None, llm_called=llm_called)

    def normalize_proposal(self, proposal: RevisionProposal) -> PolicyNormalizationResult:
        existing = self.db.scalar(
            select(ProposalNormalizationRecord).where(
                ProposalNormalizationRecord.source_proposal_id == proposal.id,
                ProposalNormalizationRecord.normalization_version
                == NORMALIZATION_VERSION,
            )
        )
        if existing is not None:
            child = (
                self.db.get(RevisionProposal, int(existing.normalized_proposal_id))
                if existing.normalized_proposal_id is not None
                else None
            )
            return PolicyNormalizationResult(existing, child, llm_called=False)

        source_ids = self._parse_ids(proposal.source_signal_ids or [])
        if not self._is_target(proposal):
            return self._record_failure(
                proposal, source_ids, "proposal_not_action_response_structure", llm_called=False
            )
        if str(proposal.status) == "invalidated":
            return self._record_failure(
                proposal, source_ids, "proposal_invalidated", llm_called=False
            )

        hypothesis = self.db.get(StrategyHypothesis, int(proposal.hypothesis_id))
        if hypothesis is None:
            return self._record_failure(
                proposal, source_ids, "hypothesis_missing", llm_called=False
            )
        if int(hypothesis.current_revision or 1) != int(proposal.base_revision):
            return self._record_failure(
                proposal, source_ids, "proposal_base_is_not_current_revision", llm_called=False
            )
        revision = self.db.scalar(
            select(HypothesisRevision).where(
                HypothesisRevision.hypothesis_id == proposal.hypothesis_id,
                HypothesisRevision.revision_number == proposal.base_revision,
            )
        )
        if revision is None:
            return self._record_failure(
                proposal, source_ids, "current_revision_missing", llm_called=False
            )

        signals = self._source_signals(proposal)
        source_problem = self._source_problem(proposal, source_ids, signals)
        if source_problem:
            return self._record_failure(
                proposal, source_ids, source_problem, llm_called=False
            )
        compact_signals = [self._compact_signal(signal) for signal in signals]

        raw = self._judge(proposal, revision, compact_signals)
        try:
            policy = self._normalize_policy(raw, compact_signals, source_ids)
        except UnsupportedPolicyCondition as exc:
            return self._record_failure(
                proposal,
                source_ids,
                str(exc),
                raw_output=raw,
                llm_called=True,
            )

        fingerprint_payload = json.dumps(
            {
                "parent": int(proposal.id),
                "normalization_version": NORMALIZATION_VERSION,
                "policy": policy,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        analysis = RevisionAnalysisRecord(
            hypothesis_id=int(proposal.hypothesis_id),
            base_revision=int(proposal.base_revision),
            dimension="ACTION",
            operation=str(proposal.operation),
            target_field="response_structure",
            signal_ids=source_ids,
            candidate_fingerprint=fingerprint,
            consistent=True,
            direction_confidence=float(proposal.direction_confidence),
            reason=f"policy_normalization_v{NORMALIZATION_VERSION}: {raw.get('reason', '')}",
        )
        self.db.add(analysis)
        self.db.flush()

        child = RevisionProposal(
            analysis_id=int(analysis.id),
            hypothesis_id=int(proposal.hypothesis_id),
            base_revision=int(proposal.base_revision),
            dimension="ACTION",
            operation=str(proposal.operation),
            target_field="response_structure",
            before_data=(revision.do_data or {}).get("response_structure"),
            proposed_after_data={"response_structure": policy},
            reason=f"Conditional response policy normalized from RP#{proposal.id}",
            source_signal_ids=source_ids,
            independent_signal_count=int(proposal.independent_signal_count or 0),
            direction_confidence=float(proposal.direction_confidence),
            status="proposed",
            parent_proposal_id=int(proposal.id),
            proposal_kind=CONDITIONAL_POLICY_KIND,
            normalization_version=NORMALIZATION_VERSION,
        )
        self.db.add(child)
        self.db.flush()

        record = ProposalNormalizationRecord(
            source_proposal_id=int(proposal.id),
            normalized_proposal_id=int(child.id),
            normalization_version=NORMALIZATION_VERSION,
            status="normalized",
            rule_count=len(policy["rules"]),
            source_signal_ids=source_ids,
            reason=str(raw.get("reason") or "conditional policy normalized"),
            normalization_data={
                "policy": policy,
                "evidence_precedence": EVIDENCE_PRECEDENCE,
            },
        )
        self.db.add(record)
        proposal.status = SUPERSEDED_STATUS
        self.db.commit()
        self.db.refresh(record)
        self.db.refresh(child)
        return PolicyNormalizationResult(record, child, llm_called=True)

    def normalize_by_id(self, proposal_id: int) -> PolicyNormalizationResult:
        proposal = self.db.get(RevisionProposal, int(proposal_id))
        if proposal is None:
            raise ValueError(f"RP#{proposal_id} does not exist")
        return self.normalize_proposal(proposal)

    def normalize_pending(self, *, max_normalizations: int = 2) -> list[PolicyNormalizationResult]:
        limit = max(0, int(max_normalizations))
        if limit == 0:
            return []
        rows = list(
            self.db.scalars(
                select(RevisionProposal)
                .where(
                    RevisionProposal.status == "proposed",
                    RevisionProposal.dimension == "ACTION",
                    RevisionProposal.target_field == "response_structure",
                    RevisionProposal.proposal_kind != CONDITIONAL_POLICY_KIND,
                )
                .order_by(RevisionProposal.id.asc())
            ).all()
        )
        results: list[PolicyNormalizationResult] = []
        for proposal in rows:
            exists = self.db.scalar(
                select(ProposalNormalizationRecord.id).where(
                    ProposalNormalizationRecord.source_proposal_id == proposal.id,
                    ProposalNormalizationRecord.normalization_version
                    == NORMALIZATION_VERSION,
                )
            )
            if exists is not None:
                continue
            results.append(self.normalize_proposal(proposal))
            if len(results) >= limit:
                break
        return results

    def list_records(self) -> list[ProposalNormalizationRecord]:
        return list(
            self.db.scalars(
                select(ProposalNormalizationRecord).order_by(
                    ProposalNormalizationRecord.id.desc()
                )
            ).all()
        )
