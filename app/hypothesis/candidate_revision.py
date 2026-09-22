from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context.schema import sanitize_applicable_context
from app.models import (
    CandidateRevision,
    Experience,
    HypothesisEvidence,
    HypothesisRevision,
    ReflectionRecord,
    RevisionProposal,
    RevisionSignal,
)


CURRENT_EVALUATION_VERSION = 3

# Candidate-local signal resolution. A source RevisionSignal exists precisely
# because rN did not fully explain that local observation. Candidate validation
# therefore measures whether the proposed delta resolves those signals, rather
# than expecting an entire Reflection with multiple independent refinements to
# magically become SUPPORTS.
SIGNAL_RESOLUTION_WEIGHTS = {
    "RESOLVED": 1.0,
    "PARTIAL": 0.20,
    "UNRESOLVED": -0.35,
    "CONFLICTS": -1.0,
}
BASE_SIGNAL_WEIGHT = -0.35  # source signals are REFINES against the base revision

OPERATION_RISK = {
    "NARROW": 0.05,
    "SPECIALIZE": 0.08,
    "ADD_CONDITION": 0.08,
    "CHANGE": 0.20,
    "EXPAND": 0.18,
    "REMOVE_CONDITION": 0.25,
    "GENERALIZE": 0.30,
}

DIMENSION_RISK = {
    "CONTEXT": 0.10,
    "ACTION": 0.15,
    "OUTCOME": 0.03,
    "ABSTRACTION": 0.25,
}

DIMENSION_TARGET = {
    "CONTEXT": "when_data",
    "ACTION": "do_data",
    "OUTCOME": "expect_data",
}

DIMENSION_ALIAS = {
    "CONTEXT": "when",
    "ACTION": "do",
    "OUTCOME": "expect",
}


class SemanticComplexityV2:
    """Score behaviour, not JSON bookkeeping.

    Provenance and ordering metadata are intentionally free.  A response policy
    pays for its base behaviour, predicates and actual effects; nesting has a
    small capped cost and repeated concepts are de-duplicated.
    """

    METADATA_KEYS = {
        "source_signal_ids",
        "base_order_source_signal_ids",
        "priority",
        "requested_priority",
        "evidence_level",
        "evidence_rank",
        "precedence",
        "policy_version",
        "normalization_version",
        "provenance",
        "metadata",
    }

    @staticmethod
    def _identity(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def _unique(cls, values: list[Any]) -> tuple[list[Any], int]:
        seen: set[str] = set()
        result: list[Any] = []
        duplicates = 0
        for value in values:
            key = cls._identity(value)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            result.append(value)
        return result, duplicates

    @classmethod
    def _predicate_count(cls, condition: Any) -> int:
        if not isinstance(condition, dict):
            return 0
        if "field" in condition or "kind" in condition:
            return 1
        total = 0
        for key, value in condition.items():
            if key in {"all", "any"} and isinstance(value, list):
                total += sum(cls._predicate_count(item) for item in value)
            elif key not in cls.METADATA_KEYS:
                total += 1
        return total

    @classmethod
    def _max_depth(cls, value: Any, depth: int = 0) -> int:
        if isinstance(value, dict):
            children = [
                cls._max_depth(item, depth + 1)
                for key, item in value.items()
                if key not in cls.METADATA_KEYS
            ]
            return max(children, default=depth)
        if isinstance(value, (list, tuple)):
            return max((cls._max_depth(item, depth + 1) for item in value), default=depth)
        return depth

    @classmethod
    def _policy_score(cls, policy: dict[str, Any]) -> int:
        base_order_spec = policy.get("base_order")
        if isinstance(base_order_spec, dict):
            base_steps = base_order_spec.get("steps")
            base_order = base_steps if isinstance(base_steps, list) else []
        else:
            # Read-only compatibility for historical normalization v1 policies.
            base_order = base_order_spec if isinstance(base_order_spec, list) else []
        unique_base, base_duplicates = cls._unique(base_order)
        score = 1 + len(unique_base)  # one base behaviour plus its ordered steps
        duplicate_count = base_duplicates

        rules = policy.get("rules") if isinstance(policy.get("rules"), list) else []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            score += 1  # one conditional branch
            score += cls._predicate_count(rule.get("condition"))
            effect = rule.get("effect") if isinstance(rule.get("effect"), dict) else {}
            for key in ("replace_order", "append_sections", "add_checks"):
                values = effect.get(key) if isinstance(effect.get(key), list) else []
                if not values:
                    continue
                _, duplicates = cls._unique(values)
                # A structured list is one behaviour delta. Its strings are
                # parameters of that delta, not N independent behaviours.
                score += 1
                duplicate_count += duplicates

        annotations = policy.get("annotations") if isinstance(policy.get("annotations"), list) else []
        annotation_texts = [
            item.get("text") if isinstance(item, dict) else item
            for item in annotations
            if (item.get("text") if isinstance(item, dict) else item)
        ]
        unique_annotations, annotation_duplicates = cls._unique(annotation_texts)
        score += len(unique_annotations)
        duplicate_count += annotation_duplicates

        # Deep encoding is a small readability concern, not a proxy for all
        # leaves. Duplicate concepts carry only a capped penalty.
        score += min(2, max(0, cls._max_depth(policy) - 4))
        score += min(3, duplicate_count)
        return score

    @classmethod
    def _generic_score(cls, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, dict):
            if "base_order" in value and "rules" in value:
                return cls._policy_score(value)
            return sum(
                cls._generic_score(item)
                for key, item in value.items()
                if key not in cls.METADATA_KEYS
            )
        if isinstance(value, (list, tuple)):
            unique, duplicates = cls._unique(list(value))
            return len(unique) + sum(cls._generic_score(item) for item in unique) + min(2, duplicates)
        if isinstance(value, str):
            return 1 if value.strip() else 0
        return 1

    @classmethod
    def revision_score(
        cls,
        when_data: dict[str, Any],
        do_data: dict[str, Any],
        expect_data: dict[str, Any],
    ) -> int:
        return (
            cls._generic_score(when_data)
            + cls._generic_score(do_data)
            + cls._generic_score(expect_data)
        )


@dataclass(frozen=True)
class CandidateEvaluationResult:
    candidate: CandidateRevision
    llm_called: bool


class CandidateRevisionService:
    """Phase 5.1: signal-aware candidate revision validation.

    Important distinction:
    - RevisionSignal is the *local* evidence that motivated a proposal.
    - Reflection/HypothesisEvidence may contain several unrelated refinements.

    A local patch should therefore be judged on whether it RESOLVES its source
    signals, while a small guard set checks that it does not regress previously
    supported behaviour. This avoids the Phase-5 failure mode where every
    single-field candidate stayed REFINES because the same Reflection still
    contained other unresolved signals.

    CandidateRevision remains experimental: this service never activates a new
    HypothesisRevision and never changes StrategyHypothesis.current_revision.
    """

    def __init__(
        self,
        db: Session,
        llm,
        *,
        max_evidence: int = 8,
        max_guard_evidence: int = 4,
    ) -> None:
        self.db = db
        self.llm = llm
        self.max_evidence = max(1, int(max_evidence))
        self.max_guard_evidence = max(0, int(max_guard_evidence))

    @staticmethod
    def _compact_text(value: Any, limit: int = 420) -> str:
        text = str(value or "").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _canonical_signal(
        when_data: dict[str, Any],
        do_data: dict[str, Any],
        expect_data: dict[str, Any],
    ) -> str:
        return json.dumps(
            {"when": when_data, "do": do_data, "expect": expect_data},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _complexity_of(cls, value: Any, *, depth: int = 0) -> int:
        if value is None:
            return 0
        if isinstance(value, dict):
            if not value:
                return 0
            child_score = sum(cls._complexity_of(item, depth=depth + 1) for item in value.values())
            return len(value) + child_score + (1 if depth >= 2 else 0)
        if isinstance(value, (list, tuple, set)):
            if not value:
                return 0
            return len(value) + sum(cls._complexity_of(item, depth=depth + 1) for item in value)
        if isinstance(value, str):
            if not value.strip():
                return 0
            return 1 + min(2, len(value) // 180)
        return 1

    @classmethod
    def _revision_complexity(
        cls,
        when_data: dict[str, Any],
        do_data: dict[str, Any],
        expect_data: dict[str, Any],
    ) -> int:
        return (
            cls._complexity_of(when_data)
            + cls._complexity_of(do_data)
            + cls._complexity_of(expect_data)
        )

    @staticmethod
    def _is_conditional_policy(proposal: RevisionProposal) -> bool:
        return str(proposal.proposal_kind or "raw") == "conditional_policy"

    @staticmethod
    def _extract_patch_value(proposal: RevisionProposal) -> tuple[str, Any]:
        dimension = str(proposal.dimension).upper()
        alias = DIMENSION_ALIAS.get(dimension)
        data = proposal.proposed_after_data

        if isinstance(data, dict):
            if alias and alias in data and isinstance(data[alias], dict):
                return "merge", copy.deepcopy(data[alias])
            if proposal.target_field in data:
                return "field", copy.deepcopy(data[proposal.target_field])
            non_meta = {
                key: value
                for key, value in data.items()
                if key not in {"target_field", "operation", "dimension"}
            }
            if len(non_meta) == 1:
                key, value = next(iter(non_meta.items()))
                if key == proposal.target_field:
                    return "field", copy.deepcopy(value)

        return "field", copy.deepcopy(data)

    def _build_candidate_data(
        self,
        base: HypothesisRevision,
        proposal: RevisionProposal,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        when_data = copy.deepcopy(base.when_data or {})
        do_data = copy.deepcopy(base.do_data or {})
        expect_data = copy.deepcopy(base.expect_data or {})

        dimension = str(proposal.dimension).upper()
        if dimension == "ABSTRACTION":
            data = proposal.proposed_after_data
            if not isinstance(data, dict):
                raise ValueError("ABSTRACTION Proposal 必须提供结构化 when/do/expect patch")
            if isinstance(data.get("when"), dict):
                when_data.update(copy.deepcopy(data["when"]))
            if isinstance(data.get("do"), dict):
                do_data.update(copy.deepcopy(data["do"]))
            if isinstance(data.get("expect"), dict):
                expect_data.update(copy.deepcopy(data["expect"]))
            if not any(key in data for key in ("when", "do", "expect")):
                raise ValueError("ABSTRACTION Proposal 缺少 when/do/expect patch")
            return when_data, do_data, expect_data

        target_name = DIMENSION_TARGET.get(dimension)
        if target_name is None:
            raise ValueError(f"不支持的 Revision dimension: {dimension}")

        target = {
            "when_data": when_data,
            "do_data": do_data,
            "expect_data": expect_data,
        }[target_name]
        patch_mode, patch_value = self._extract_patch_value(proposal)
        if patch_mode == "merge" and isinstance(patch_value, dict):
            target.update(patch_value)
        else:
            target[proposal.target_field] = patch_value

        return when_data, do_data, expect_data

    def _source_signals(self, proposal: RevisionProposal) -> list[RevisionSignal]:
        ids = [int(item) for item in (proposal.source_signal_ids or []) if item is not None]
        if not ids:
            return []
        rows = list(
            self.db.scalars(
                select(RevisionSignal).where(RevisionSignal.id.in_(ids))
            ).all()
        )
        by_id = {int(item.id): item for item in rows}
        return [by_id[item] for item in ids if item in by_id]

    def _compact_source_signals(self, signals: list[RevisionSignal]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for signal in signals:
            context: dict[str, Any] = {}
            reflection = self.db.get(ReflectionRecord, signal.reflection_id)
            if reflection is not None:
                context = sanitize_applicable_context(reflection.applicable_context or {})
                if not context and reflection.experience_id is not None:
                    experience = self.db.get(Experience, reflection.experience_id)
                    if experience is not None:
                        context = sanitize_applicable_context(experience.context_data or {})
            items.append(
                {
                    "signal_id": int(signal.id),
                    "reflection_id": int(signal.reflection_id),
                    "dimension": str(signal.dimension).upper(),
                    "operation": str(signal.operation).upper(),
                    "target_field": str(signal.target_field),
                    "before": signal.before_data,
                    "observed": signal.observed_data,
                    "canonical_change": self._compact_text(signal.canonical_change),
                    "signal_confidence": float(signal.confidence),
                    "context": context,
                }
            )
        return items

    def _source_evidence_ids(self, signals: list[RevisionSignal]) -> set[int]:
        return {
            int(signal.evidence_id)
            for signal in signals
            if signal.evidence_id is not None
        }

    def _guard_evidence_rows(
        self,
        proposal: RevisionProposal,
        source_evidence_ids: set[int],
    ) -> list[tuple[HypothesisEvidence, ReflectionRecord, Experience | None]]:
        if self.max_guard_evidence <= 0:
            return []
        stmt = (
            select(HypothesisEvidence, ReflectionRecord, Experience)
            .join(ReflectionRecord, ReflectionRecord.id == HypothesisEvidence.reflection_id)
            .outerjoin(Experience, Experience.id == ReflectionRecord.experience_id)
            .where(
                HypothesisEvidence.hypothesis_id == proposal.hypothesis_id,
                HypothesisEvidence.hypothesis_revision == proposal.base_revision,
            )
        )
        rows = [
            row for row in self.db.execute(stmt).all()
            if int(row[0].id) not in source_evidence_ids
        ]
        # Regression guard prioritizes evidence that currently SUPPORTS the base,
        # then conflicts, then a small number of remaining refinements.
        priority = {"SUPPORTS": 0, "CONFLICTS": 1, "REFINES": 2, "UNRELATED": 3}
        rows.sort(
            key=lambda row: (
                priority.get(str(row[0].relation_type).upper(), 9),
                int(row[0].id),
            )
        )
        return rows[: self.max_guard_evidence]

    def _compact_guard_evidence(
        self,
        rows: list[tuple[HypothesisEvidence, ReflectionRecord, Experience | None]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for evidence, reflection, experience in rows:
            context = sanitize_applicable_context(reflection.applicable_context or {})
            if not context and experience is not None:
                context = sanitize_applicable_context(experience.context_data or {})
            items.append(
                {
                    "evidence_id": int(evidence.id),
                    "relation_to_base": str(evidence.relation_type).upper(),
                    "context": context,
                    "signal": self._compact_text(
                        reflection.canonical_signal
                        or reflection.strategy_signal
                        or reflection.insight
                        or reflection.lesson
                    ),
                }
            )
        return items

    def _reevaluate(
        self,
        candidate: dict[str, Any],
        source_signals: list[dict[str, Any]],
        guard_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Evaluate a local Candidate Revision. Source RevisionSignals are the exact local "
                    "refinements that motivated this proposal. For EACH source signal decide whether "
                    "the candidate RESOLVES it, PARTIALLY resolves it, leaves it UNRESOLVED, or "
                    "CONFLICTS with it. Do not require the candidate to solve other unrelated "
                    "refinements from the same Reflection. Also inspect guard evidence only for "
                    "regressions: PASS if the candidate preserves it, REGRESSION if it breaks it, "
                    "UNCERTAIN if unclear. Do not invent context or benefits. Return a JSON object only. "
                    "Use the exact keys `signal_assessments` and `guard_assessments`. For each signal "
                    "assessment, `signal_id` MUST be the numeric integer copied exactly from the input, "
                    "and `resolution` MUST be one of RESOLVED, PARTIAL, UNRESOLVED, CONFLICTS. For each "
                    "guard assessment, `evidence_id` MUST be the numeric integer copied exactly from "
                    "the input, and `verdict` MUST be one of PASS, REGRESSION, UNCERTAIN. Example: "
                    "{\"signal_assessments\":[{\"signal_id\":13,\"resolution\":\"RESOLVED\","
                    "\"confidence\":0.9,\"reason\":\"...\"}],\"guard_assessments\":[]}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidate": candidate,
                        "source_signals": source_signals,
                        "guard_evidence": guard_evidence,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        result = self.llm.json_chat(messages, temperature=0.1)
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _parse_numeric_id(value: Any) -> int | None:
        """Accept strict ints plus common LLM renderings such as `RS#13` or `E#4`."""
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        matches = re.findall(r"\d+", text)
        if len(matches) == 1:
            return int(matches[0])
        return None

    @staticmethod
    def _normalize_resolution(value: Any) -> str | None:
        text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "RESOLVED": "RESOLVED",
            "FULLY_RESOLVED": "RESOLVED",
            "SUPPORTED": "RESOLVED",
            "SUPPORTS": "RESOLVED",
            "已解决": "RESOLVED",
            "解决": "RESOLVED",
            "PARTIAL": "PARTIAL",
            "PARTIALLY_RESOLVED": "PARTIAL",
            "PARTIAL_RESOLUTION": "PARTIAL",
            "部分解决": "PARTIAL",
            "UNRESOLVED": "UNRESOLVED",
            "NOT_RESOLVED": "UNRESOLVED",
            "NOTRESOLVED": "UNRESOLVED",
            "未解决": "UNRESOLVED",
            "CONFLICTS": "CONFLICTS",
            "CONFLICT": "CONFLICTS",
            "CONTRADICTS": "CONFLICTS",
            "冲突": "CONFLICTS",
        }
        return aliases.get(text)

    @staticmethod
    def _normalize_guard_verdict(value: Any) -> str | None:
        text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "PASS": "PASS",
            "PRESERVED": "PASS",
            "OK": "PASS",
            "通过": "PASS",
            "REGRESSION": "REGRESSION",
            "REGRESSED": "REGRESSION",
            "BREAKS": "REGRESSION",
            "回归": "REGRESSION",
            "UNCERTAIN": "UNCERTAIN",
            "UNKNOWN": "UNCERTAIN",
            "不确定": "UNCERTAIN",
        }
        return aliases.get(text)

    @classmethod
    def _normalize_signal_assessments(
        cls,
        raw: dict[str, Any],
        signal_ids: set[int],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[int] = set()
        rows = raw.get("signal_assessments")
        if rows is None:
            rows = raw.get("source_signal_assessments")
        if rows is None:
            rows = raw.get("signals")
        if not isinstance(rows, list):
            return result
        for item in rows:
            if not isinstance(item, dict):
                continue
            signal_id = cls._parse_numeric_id(
                item.get("signal_id", item.get("revision_signal_id", item.get("id")))
            )
            if signal_id is None or signal_id not in signal_ids or signal_id in seen:
                continue
            resolution = cls._normalize_resolution(
                item.get("resolution", item.get("status", item.get("verdict")))
            )
            if resolution is None:
                continue
            seen.add(signal_id)
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            result.append(
                {
                    "signal_id": signal_id,
                    "resolution": resolution,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": str(item.get("reason", item.get("explanation", ""))).strip(),
                }
            )
        return result

    @classmethod
    def _normalize_guard_assessments(
        cls,
        raw: dict[str, Any],
        evidence_ids: set[int],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[int] = set()
        rows = raw.get("guard_assessments")
        if rows is None:
            rows = raw.get("guard_evidence_assessments")
        if rows is None:
            rows = raw.get("guards")
        if not isinstance(rows, list):
            return result
        for item in rows:
            if not isinstance(item, dict):
                continue
            evidence_id = cls._parse_numeric_id(
                item.get("evidence_id", item.get("hypothesis_evidence_id", item.get("id")))
            )
            if evidence_id is None or evidence_id not in evidence_ids or evidence_id in seen:
                continue
            verdict = cls._normalize_guard_verdict(
                item.get("verdict", item.get("status", item.get("result")))
            )
            if verdict is None:
                continue
            seen.add(evidence_id)
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            result.append(
                {
                    "evidence_id": evidence_id,
                    "verdict": verdict,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": str(item.get("reason", item.get("explanation", ""))).strip(),
                }
            )
        return result

    @staticmethod
    def _normalized_weight_score(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        weighted = 0.0
        total_confidence = 0.0
        for item in items:
            confidence = max(0.05, float(item.get("confidence", 0.0)))
            weighted += SIGNAL_RESOLUTION_WEIGHTS[str(item[key]).upper()] * confidence
            total_confidence += confidence
        raw = weighted / total_confidence if total_confidence else -1.0
        return round(max(0.0, min(1.0, (raw + 1.0) / 2.0)), 4)

    @staticmethod
    def _base_local_fit() -> float:
        return round((BASE_SIGNAL_WEIGHT + 1.0) / 2.0, 4)

    @staticmethod
    def _risk(proposal: RevisionProposal, complexity_delta: int) -> tuple[float, str]:
        score = OPERATION_RISK.get(str(proposal.operation).upper(), 0.25)
        score += DIMENSION_RISK.get(str(proposal.dimension).upper(), 0.20)
        if complexity_delta > 3:
            score += 0.08
        if complexity_delta > 8:
            score += 0.10
        score = max(0.0, min(1.0, score))
        if score <= 0.30:
            return round(score, 4), "LOW"
        if score <= 0.60:
            return round(score, 4), "MEDIUM"
        return round(score, 4), "HIGH"

    @staticmethod
    def _status(
        *,
        fit_gain: float,
        utility: float,
        risk_level: str,
        source_coverage: float,
        regression_count: int,
        uncertain_guard_count: int,
    ) -> str:
        if source_coverage < 0.80:
            return "needs_review"
        if regression_count > 0:
            return "rejected"
        if fit_gain <= 0.05:
            return "rejected"
        if uncertain_guard_count > 0:
            return "needs_review"
        if risk_level == "LOW" and utility >= 0.08:
            return "ready_to_apply"
        return "needs_review"

    def _evaluate_data(
        self,
        *,
        proposal: RevisionProposal,
        base: HypothesisRevision,
        when_data: dict[str, Any],
        do_data: dict[str, Any],
        expect_data: dict[str, Any],
    ) -> dict[str, Any]:
        complexity_version = 2 if self._is_conditional_policy(proposal) else 1
        scorer = (
            SemanticComplexityV2.revision_score
            if complexity_version == 2
            else self._revision_complexity
        )
        base_complexity = scorer(
            base.when_data or {}, base.do_data or {}, base.expect_data or {}
        )
        candidate_complexity = scorer(when_data, do_data, expect_data)
        complexity_delta = candidate_complexity - base_complexity

        signals = self._source_signals(proposal)
        compact_signals = self._compact_source_signals(signals)
        signal_ids = {int(item["signal_id"]) for item in compact_signals}
        source_evidence_ids = self._source_evidence_ids(signals)
        guard_rows = self._guard_evidence_rows(proposal, source_evidence_ids)
        compact_guard = self._compact_guard_evidence(guard_rows)
        guard_ids = {int(item["evidence_id"]) for item in compact_guard}

        raw = self._reevaluate(
            {
                "hypothesis_id": proposal.hypothesis_id,
                "base_revision": proposal.base_revision,
                "source_proposal_id": proposal.id,
                "when": when_data,
                "do": do_data,
                "expect": expect_data,
            },
            compact_signals,
            compact_guard,
        )
        assessments = self._normalize_signal_assessments(raw, signal_ids)
        guard_assessments = self._normalize_guard_assessments(raw, guard_ids)

        source_coverage = round(len(assessments) / len(compact_signals), 4) if compact_signals else 0.0
        guard_coverage = round(len(guard_assessments) / len(compact_guard), 4) if compact_guard else 1.0

        base_fit = self._base_local_fit()
        evaluation_valid = bool(compact_signals) and bool(assessments)
        evaluation_error = ""
        if not compact_signals:
            evaluation_error = "proposal_has_no_source_signals"
        elif not assessments:
            evaluation_error = "no_valid_signal_assessments"

        if evaluation_valid:
            candidate_fit = self._normalized_weight_score(assessments, "resolution")
            fit_gain = round(candidate_fit - base_fit, 4)
        else:
            # Missing/invalid evaluator output is uncertainty, not evidence that
            # the candidate is worse than the base revision.
            candidate_fit = base_fit
            fit_gain = 0.0

        regression_count = sum(1 for item in guard_assessments if item["verdict"] == "REGRESSION")
        uncertain_guard_count = sum(1 for item in guard_assessments if item["verdict"] == "UNCERTAIN")

        if complexity_version == 2:
            complexity_penalty = round(min(0.08, max(0, complexity_delta) * 0.005), 4)
        else:
            complexity_penalty = round(min(0.15, max(0, complexity_delta) * 0.01), 4)
        risk_score, risk_level = self._risk(proposal, complexity_delta)
        risk_penalty = round(risk_score * 0.20, 4)
        regression_penalty = round(regression_count * 0.25 + uncertain_guard_count * 0.05, 4)
        if evaluation_valid:
            utility = round(fit_gain - complexity_penalty - risk_penalty - regression_penalty, 4)
            status = self._status(
                fit_gain=fit_gain,
                utility=utility,
                risk_level=risk_level,
                source_coverage=source_coverage,
                regression_count=regression_count,
                uncertain_guard_count=uncertain_guard_count,
            )
            if self._is_conditional_policy(proposal) and status == "ready_to_apply":
                status = "needs_review"
        else:
            utility = 0.0
            status = "needs_review"

        return {
            "base_complexity": base_complexity,
            "candidate_complexity": candidate_complexity,
            "complexity_delta": complexity_delta,
            "base_fit": base_fit,
            "candidate_fit": candidate_fit,
            "fit_gain": fit_gain,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "utility": utility,
            "status": status,
            "complexity_version": complexity_version,
            "signal_ids": sorted(signal_ids),
            "source_coverage": source_coverage,
            "guard_coverage": guard_coverage,
            "regression_count": regression_count,
            "uncertain_guard_count": uncertain_guard_count,
            "evaluation_data": {
                "method": "signal_resolution_v3",
                "evaluation_version": CURRENT_EVALUATION_VERSION,
                "complexity_version": complexity_version,
                "manual_review_required": self._is_conditional_policy(proposal),
                "evaluation_valid": evaluation_valid,
                "evaluation_error": evaluation_error,
                "raw_evaluator_output": raw,
                "source_signal_ids_expected": sorted(signal_ids),
                "source_signal_ids_accepted": sorted(int(item["signal_id"]) for item in assessments),
                "source_coverage": source_coverage,
                "guard_coverage": guard_coverage,
                "source_signal_assessments": assessments,
                "guard_assessments": guard_assessments,
                "regression_count": regression_count,
                "uncertain_guard_count": uncertain_guard_count,
                "complexity_penalty": complexity_penalty,
                "risk_penalty": risk_penalty,
                "regression_penalty": regression_penalty,
            },
        }

    def evaluate_proposal(self, proposal: RevisionProposal) -> CandidateEvaluationResult:
        existing = self.db.scalar(
            select(CandidateRevision).where(CandidateRevision.source_proposal_id == proposal.id)
        )
        if existing is not None and int(existing.evaluation_version or 1) >= CURRENT_EVALUATION_VERSION:
            return CandidateEvaluationResult(existing, llm_called=False)

        base = self.db.scalar(
            select(HypothesisRevision).where(
                HypothesisRevision.hypothesis_id == proposal.hypothesis_id,
                HypothesisRevision.revision_number == proposal.base_revision,
            )
        )
        if base is None:
            raise ValueError(f"H#{proposal.hypothesis_id}:r{proposal.base_revision} 不存在")

        when_data, do_data, expect_data = self._build_candidate_data(base, proposal)
        evaluation = self._evaluate_data(
            proposal=proposal,
            base=base,
            when_data=when_data,
            do_data=do_data,
            expect_data=expect_data,
        )

        previous = copy.deepcopy(existing.evaluation_data) if existing is not None else None
        candidate = existing or CandidateRevision(
            hypothesis_id=proposal.hypothesis_id,
            base_revision=proposal.base_revision,
            source_proposal_id=proposal.id,
        )
        candidate.title = base.title
        candidate.insight = base.insight
        candidate.when_data = when_data
        candidate.do_data = do_data
        candidate.expect_data = expect_data
        candidate.canonical_signal = self._canonical_signal(when_data, do_data, expect_data)
        candidate.status = evaluation["status"]
        candidate.base_fit_score = evaluation["base_fit"]
        candidate.candidate_fit_score = evaluation["candidate_fit"]
        candidate.fit_gain = evaluation["fit_gain"]
        candidate.base_complexity_score = evaluation["base_complexity"]
        candidate.complexity_score = evaluation["candidate_complexity"]
        candidate.complexity_delta = evaluation["complexity_delta"]
        candidate.risk_level = evaluation["risk_level"]
        candidate.risk_score = evaluation["risk_score"]
        candidate.utility_score = evaluation["utility"]
        candidate.evaluation_version = CURRENT_EVALUATION_VERSION
        candidate.complexity_version = evaluation["complexity_version"]
        candidate.evidence_ids = evaluation["signal_ids"]
        data = evaluation["evaluation_data"]
        if previous:
            data["previous_evaluation"] = previous
        candidate.evaluation_data = data

        if existing is None:
            self.db.add(candidate)
        self.db.commit()
        self.db.refresh(candidate)
        return CandidateEvaluationResult(candidate, llm_called=True)

    def evaluate_pending(self, *, max_candidates: int = 2) -> list[CandidateEvaluationResult]:
        limit = max(0, int(max_candidates))
        if limit == 0:
            return []
        stmt = (
            select(RevisionProposal)
            .where(RevisionProposal.status == "proposed")
            .order_by(RevisionProposal.id.asc())
        )
        results: list[CandidateEvaluationResult] = []
        for proposal in self.db.scalars(stmt).all():
            if (
                str(proposal.dimension).upper() == "ACTION"
                and str(proposal.target_field) == "response_structure"
                and str(proposal.proposal_kind or "raw") != "conditional_policy"
            ):
                # Phase 5.6 routes these through ProposalPolicyNormalizer first.
                # Direct evaluate_proposal remains available for historical test
                # fixtures and audit reproduction of legacy candidates.
                continue
            existing = self.db.scalar(
                select(CandidateRevision).where(CandidateRevision.source_proposal_id == proposal.id)
            )
            if existing is not None and int(existing.evaluation_version or 1) >= CURRENT_EVALUATION_VERSION:
                continue
            results.append(self.evaluate_proposal(proposal))
            if len(results) >= limit:
                break
        return results

    def list_candidates(self) -> list[CandidateRevision]:
        stmt = select(CandidateRevision).order_by(CandidateRevision.id.desc())
        return list(self.db.scalars(stmt).all())
