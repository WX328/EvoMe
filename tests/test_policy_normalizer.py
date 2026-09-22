import copy
import json

from app.hypothesis.candidate_revision import CandidateRevisionService, SemanticComplexityV2
from app.hypothesis.policy_normalizer import ProposalPolicyNormalizer
from app.hypothesis.revision_commit import RevisionCommitService
from app.models import (
    CandidateRevision,
    Experience,
    ProposalNormalizationRecord,
    ReflectionRecord,
    RevisionAnalysisRecord,
    RevisionProposal,
)
from tests.test_candidate_revision import FakeCandidateLLM
from tests.test_revision_aggregation import _add_refine_signal, _seed_hypothesis


class PolicyLLM:
    def __init__(self, *, unsupported=False):
        self.calls = 0
        self.unsupported = unsupported

    def json_chat(self, messages, temperature=0.1):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        signal_ids = [item["signal_id"] for item in payload["signals"]]
        explicit_condition = (
            {"field": "audience", "operator": "equals", "value": "invented"}
            if self.unsupported
            else {"kind": "explicit_user_instruction"}
        )
        return {
            "base_order": {
                "steps": [
                    "responsibility matrix",
                    "data flow",
                    "interface draft",
                    "pending decisions",
                ],
                "source_signal_ids": [signal_ids[3]],
            },
            "rules": [
                {
                    "rule_type": "explicit_order_override",
                    "condition": explicit_condition,
                    "effect": {
                        "replace_order": ["data flow", "module interfaces", "implementation"],
                        "append_sections": [],
                        "add_checks": [],
                    },
                    "priority": 1,
                    "source_signal_ids": [signal_ids[0]],
                },
                {
                    "rule_type": "evidence_reporting",
                    "condition": {"task_type": "evidence_analysis"},
                    "effect": {
                        "replace_order": [],
                        "append_sections": [
                            "evidence",
                            "candidate conclusions",
                            "unsupported claims",
                            "confidence",
                            "next steps",
                        ],
                        "add_checks": [],
                    },
                    "priority": 99999,
                    "source_signal_ids": [signal_ids[1]],
                },
                {
                    "rule_type": "architecture_boundary_checks",
                    "condition": {"task_type": "architecture_design"},
                    "effect": {
                        "replace_order": [],
                        "append_sections": [],
                        "add_checks": [
                            "responsibility boundary",
                            "call relationship",
                            "confusion risk",
                            "anti-pattern",
                        ],
                    },
                    "priority": 99999,
                    "source_signal_ids": [signal_ids[2]],
                },
            ],
            "annotations": [],
            "reason": "separate default, overrides, evidence sections, and architecture checks",
        }


def _raw_response_structure_proposal(db):
    hypothesis = _seed_hypothesis(db)
    signals = [
        _add_refine_signal(
            db,
            hypothesis,
            interaction_id=5600 + index,
            reflection_seed=seed,
            target_field="response_structure",
        )
        for index, seed in enumerate(
            [
                "explicit output order preference",
                "evidence confidence unsupported next step",
                "architecture boundaries and anti-pattern checks",
                "historical default response order",
            ]
        )
    ]

    signals[0].provenance_data = {
        "explicit_user_instruction": True,
        "source_interaction_id": signals[0].interaction_id,
    }
    contexts = [
        {"task_type": "architecture_design", "complexity": "medium"},
        {"task_type": "evidence_analysis", "complexity": "high"},
        {"task_type": "architecture_design", "complexity": "high"},
        {"task_type": "architecture_design", "complexity": "medium"},
    ]
    for signal, context in zip(signals, contexts):
        reflection = db.get(ReflectionRecord, signal.reflection_id)
        reflection.applicable_context = context
        experience = db.get(Experience, reflection.experience_id)
        experience.context_data = context

    analysis = RevisionAnalysisRecord(
        hypothesis_id=hypothesis.id,
        base_revision=hypothesis.current_revision,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        signal_ids=[item.id for item in signals],
        candidate_fingerprint=f"phase-5-6-raw-{hypothesis.id}",
        consistent=True,
        direction_confidence=0.84,
        reason="legacy union proposal",
    )
    db.add(analysis)
    db.flush()
    raw_after = {
        "response_structure": {
            "default_order": [
                "responsibility matrix",
                "data flow",
                "interface draft",
                "pending decisions",
            ],
            "user_preference_override": "data flow -> module interfaces -> implementation",
            "evidence_and_confidence_sections": [
                "evidence",
                "candidate conclusions",
                "unsupported claims",
                "confidence",
                "next steps",
            ],
            "boundary_and_checks": [
                "responsibility boundary",
                "call relationship",
                "anti-pattern",
            ],
        }
    }
    proposal = RevisionProposal(
        analysis_id=analysis.id,
        hypothesis_id=hypothesis.id,
        base_revision=hypothesis.current_revision,
        dimension="ACTION",
        operation="EXPAND",
        target_field="response_structure",
        before_data=None,
        proposed_after_data=raw_after,
        reason="legacy union proposal",
        source_signal_ids=[item.id for item in signals],
        independent_signal_count=4,
        direction_confidence=0.84,
        status="proposed",
    )
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return hypothesis, proposal, signals


def test_phase5_6_normalizes_union_into_evidence_backed_conditional_policy(db):
    hypothesis, raw_proposal, signals = _raw_response_structure_proposal(db)
    original_after = copy.deepcopy(raw_proposal.proposed_after_data)
    llm = PolicyLLM()
    normalizer = ProposalPolicyNormalizer(db, llm)

    first = normalizer.normalize_proposal(raw_proposal)
    second = normalizer.normalize_proposal(raw_proposal)

    assert llm.calls == 1
    assert first.record.id == second.record.id
    assert first.proposal.id == second.proposal.id
    assert first.record.status == "normalized"
    assert first.record.rule_count == 3
    assert first.proposal.parent_proposal_id == raw_proposal.id
    assert first.proposal.proposal_kind == "conditional_policy"
    assert first.record.normalization_version == 2
    assert first.proposal.normalization_version == 2
    assert first.proposal.base_revision == hypothesis.current_revision

    policy = first.proposal.proposed_after_data["response_structure"]
    assert policy["base_order"]["steps"] == [
        "responsibility matrix",
        "data flow",
        "interface draft",
        "pending decisions",
    ]
    assert policy["base_order"]["source_signal_ids"] == [signals[3].id]
    assert "evidence" not in policy["base_order"]["steps"]
    assert len(policy["rules"]) == 3
    assert all(rule["source_signal_ids"] for rule in policy["rules"])
    assert set(policy["source_signal_ids"]) == {item.id for item in signals}

    explicit = next(rule for rule in policy["rules"] if rule["rule_type"] == "explicit_order_override")
    evidence = next(rule for rule in policy["rules"] if rule["rule_type"] == "evidence_reporting")
    assert explicit["effect"]["replace_order"]
    assert explicit["evidence_rank"] == 1
    assert explicit["priority"] > evidence["priority"]
    # The model's arbitrary requested priorities are ignored in favour of the
    # deterministic evidence/specificity ordering.
    assert evidence["priority"] != 99999

    db.refresh(raw_proposal)
    assert raw_proposal.proposed_after_data == original_after
    assert raw_proposal.status == "superseded_by_normalized_proposal"


def test_phase5_6_v2_preserves_v1_failure_and_retries_same_proposal(db):
    _hypothesis, raw_proposal, _signals = _raw_response_structure_proposal(db)
    v1 = ProposalNormalizationRecord(
        source_proposal_id=raw_proposal.id,
        normalized_proposal_id=None,
        normalization_version=1,
        status="cannot_normalize",
        rule_count=0,
        source_signal_ids=raw_proposal.source_signal_ids,
        reason="base_order_requires_source_signals",
        normalization_data={"error": "base_order_requires_source_signals"},
    )
    db.add(v1)
    db.commit()
    v1_id = v1.id

    llm = PolicyLLM()
    result = ProposalPolicyNormalizer(db, llm).normalize_proposal(raw_proposal)

    assert llm.calls == 1
    assert result.record.id != v1_id
    assert result.record.normalization_version == 2
    assert result.record.status == "normalized"
    db.refresh(v1)
    assert v1.status == "cannot_normalize"
    assert v1.reason == "base_order_requires_source_signals"


def test_phase5_6_unsupported_condition_is_rejected_without_child_proposal(db):
    _hypothesis, raw_proposal, _signals = _raw_response_structure_proposal(db)
    llm = PolicyLLM(unsupported=True)

    result = ProposalPolicyNormalizer(db, llm).normalize_proposal(raw_proposal)

    assert llm.calls == 1
    assert result.record.status == "cannot_normalize"
    assert result.record.reason == "unsupported_condition:audience"
    assert result.proposal is None
    db.refresh(raw_proposal)
    assert raw_proposal.status == "proposed"


def test_phase5_6_invalidated_source_never_enters_policy_and_spends_no_llm(db):
    _hypothesis, raw_proposal, signals = _raw_response_structure_proposal(db)
    signals[1].resolution_status = "invalid"
    db.commit()
    llm = PolicyLLM()

    result = ProposalPolicyNormalizer(db, llm).normalize_proposal(raw_proposal)

    assert llm.calls == 0
    assert result.record.status == "cannot_normalize"
    assert result.record.reason == f"source_signal_{signals[1].id}_not_open"
    assert result.proposal is None


def test_phase5_6_new_candidate_rebuilds_from_head_and_does_not_inherit_old_evaluation(db):
    hypothesis, raw_proposal, _signals = _raw_response_structure_proposal(db)
    old_candidate = CandidateRevision(
        hypothesis_id=hypothesis.id,
        base_revision=hypothesis.current_revision,
        source_proposal_id=raw_proposal.id,
        title="legacy candidate",
        status="needs_review",
        base_fit_score=0.325,
        candidate_fit_score=1.0,
        fit_gain=0.675,
        risk_level="MEDIUM",
        risk_score=0.51,
        utility_score=0.423,
        evaluation_version=3,
        complexity_version=1,
        evaluation_data={"legacy_marker": "CR#4"},
    )
    db.add(old_candidate)
    db.commit()

    normalized = ProposalPolicyNormalizer(db, PolicyLLM()).normalize_proposal(raw_proposal)
    candidate = CandidateRevisionService(db, FakeCandidateLLM()).evaluate_proposal(
        normalized.proposal
    ).candidate

    assert candidate.id != old_candidate.id
    assert candidate.source_proposal_id == normalized.proposal.id
    assert candidate.base_revision == hypothesis.current_revision
    assert candidate.complexity_version == 2
    assert candidate.evaluation_data["complexity_version"] == 2
    assert "legacy_marker" not in candidate.evaluation_data
    assert "previous_evaluation" not in candidate.evaluation_data
    assert candidate.status == "needs_review"

    db.refresh(old_candidate)
    assert old_candidate.status == "needs_review"
    assert old_candidate.evaluation_data == {"legacy_marker": "CR#4"}


def test_phase5_6_semantic_complexity_ignores_policy_metadata(db):
    _hypothesis, raw_proposal, _signals = _raw_response_structure_proposal(db)
    normalized = ProposalPolicyNormalizer(db, PolicyLLM()).normalize_proposal(raw_proposal)
    policy = normalized.proposal.proposed_after_data["response_structure"]
    score = SemanticComplexityV2.revision_score({}, {"response_structure": policy}, {})

    metadata_heavy = copy.deepcopy(policy)
    metadata_heavy["source_signal_ids"] = list(range(5000))
    metadata_heavy["precedence"] = ["metadata"] * 1000
    for rule in metadata_heavy["rules"]:
        rule["source_signal_ids"] = list(range(1000))
        rule["priority"] = 999999
        rule["provenance"] = {"deep": {"metadata": ["x"] * 500}}
    metadata_score = SemanticComplexityV2.revision_score(
        {}, {"response_structure": metadata_heavy}, {}
    )

    assert metadata_score == score
    assert score < 40


def test_phase5_6_conditional_policy_cannot_auto_apply_even_if_status_is_forced_ready(db):
    hypothesis, raw_proposal, _signals = _raw_response_structure_proposal(db)
    normalized = ProposalPolicyNormalizer(db, PolicyLLM()).normalize_proposal(raw_proposal)
    candidate = CandidateRevisionService(db, FakeCandidateLLM()).evaluate_proposal(
        normalized.proposal
    ).candidate
    assert candidate.status == "needs_review"

    candidate.status = "ready_to_apply"
    candidate.risk_level = "LOW"
    candidate.risk_score = 0.1
    candidate.utility_score = 1.0
    db.commit()

    results = RevisionCommitService(db, auto_min_utility=0.0).auto_apply_ready(
        max_candidates=1
    )
    assert results == []
    db.refresh(hypothesis)
    assert hypothesis.current_revision == 1
