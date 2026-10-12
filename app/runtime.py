from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.audit.service import AuditService
from app.background import BackgroundEvolutionQueue
from app.config import (
    DEEPSEEK_MODEL,
    EVOME_AGENT_LOOP_ENABLED,
    EVOME_AGENT_MAX_STEPS,
    EVOME_HOOKS_ENABLED,
    EVOME_HOOK_HISTORY_LIMIT,
    EVOME_HOOK_PROTECT_SECRETS,
    EVOME_BASH_TOOL_ENABLED,
    EVOME_BASH_TIMEOUT_SECONDS,
    EVOME_BASH_PERMISSION,
    EVOME_TOOL_MAX_OUTPUT_CHARS,
    EVOME_TOOL_WORKSPACE,
    EVOME_QDRANT_PATH,
    EVOME_SLOW_LOOP_EVERY_N_TURNS,
    HYPOTHESIS_RELATION_CONFIDENCE,
    HYPOTHESIS_SCORE_THRESHOLD,
    HYPOTHESIS_TOP_K,
    REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS,
    REVISION_ANALYSIS_MAX_PER_RUN,
    REVISION_CANDIDATE_MAX_EVIDENCE,
    REVISION_CANDIDATE_MAX_PER_RUN,
    REVISION_AUTO_APPLY_MIN_UTILITY,
    REVISION_AUTO_APPLY_BUDGET_HOURS,
    REVISION_AUTO_APPLY_PER_HYPOTHESIS,
    REVISION_AUTO_APPLY_MAX_PER_RUN,
    REVISION_REVALIDATION_MAX_SIGNALS_PER_BATCH,
    REVISION_REVALIDATION_MAX_HYPOTHESES_PER_RUN,
)
from app.context.builder import ContextBuilder
from app.evaluation.service import EvaluationService
from app.evolution.service import EvolutionMemoryService
from app.gates.experience_gate import ExperienceGate
from app.gates.learning_eligibility_gate import LearningEligibilityGate
from app.gates.personal_memory_gate import PersonalMemoryGate
from app.hooks import HookManager, build_sensitive_path_guard
from app.hypothesis.aggregator import HypothesisAggregator
from app.hypothesis.ancestor_revalidation import AncestorSignalRevalidationService
from app.hypothesis.candidate_rebase import CandidateRebaseService
from app.hypothesis.candidate_revision import CandidateRevisionService
from app.hypothesis.policy_normalizer import ProposalPolicyNormalizer
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.revision_aggregation import RevisionAggregationService
from app.hypothesis.revision_analyzer import RevisionAnalysisService
from app.hypothesis.revision_commit import RevisionCommitService
from app.hypothesis.revision_reconciliation import RevisionReconciliationService
from app.hypothesis.retriever import HybridHypothesisRetriever
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService
from app.learning.invalidation import LearningInvalidationService
from app.learning.post_turn_assessor import PostTurnAssessor
from app.llm import DeepSeekClient
from app.telemetry import GLOBAL_TOKEN_USAGE
from app.memory.service import PersonalMemoryService
from app.models import CandidateRevision, RevisionProposal, StrategyHypothesis, StrategyRecord
from app.reflection.service import ReflectionService
from app.skills.builtin import build_default_skill_registry
from app.tools import ApprovalStore
from app.strategy.default_workflow import DefaultConversationStrategy


class EvoMeRuntime:
    def __init__(
        self,
        db: Session,
        llm=None,
        session_id: str | None = None,
        *,
        enable_background: bool = True,
        enable_agent_loop: bool | None = None,
    ) -> None:
        self.db = db
        injected_llm = llm is not None
        self.llm = llm or DeepSeekClient()
        self.session_id = session_id or uuid4().hex
        self.enable_background = enable_background
        self.enable_agent_loop = (
            bool(enable_agent_loop)
            if enable_agent_loop is not None
            else (EVOME_AGENT_LOOP_ENABLED if not injected_llm else False)
        )

        # Background jobs must use a fresh SQLAlchemy Session. Reusing the
        # foreground Session across threads is unsafe.
        self._background_session_factory = sessionmaker(
            bind=db.get_bind(),
            autoflush=False,
            expire_on_commit=False,
        )
        self.background_queue = (
            BackgroundEvolutionQueue(max_workers=1) if enable_background else None
        )
        self._last_background_job_id: str | None = None
        # Phase 5.7C: counts user turns so the slow evolution job can run on a
        # reduced cadence (see EVOME_SLOW_LOOP_EVERY_N_TURNS).
        self._slow_loop_turn_counter = 0

        self.audit_service = AuditService(db)
        self.personal_memory_service = PersonalMemoryService(db)
        self.evolution_memory_service = EvolutionMemoryService(db)
        self.hypothesis_service = HypothesisService(db)
        self.revision_signal_service = RevisionSignalService(db)
        self.revision_reconciliation_service = RevisionReconciliationService(db)
        self.learning_invalidation_service = LearningInvalidationService(db)
        self.revision_aggregation_service = RevisionAggregationService(
            db, min_independent_signals=REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS
        )
        self.revision_analysis_service = RevisionAnalysisService(
            db,
            self.llm,
            min_independent_signals=REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS,
        )
        self.policy_normalizer_service = ProposalPolicyNormalizer(db, self.llm)
        self.candidate_revision_service = CandidateRevisionService(
            db,
            self.llm,
            max_evidence=REVISION_CANDIDATE_MAX_EVIDENCE,
        )
        self.ancestor_revalidation_service = AncestorSignalRevalidationService(
            db,
            self.llm,
            max_signals_per_batch=REVISION_REVALIDATION_MAX_SIGNALS_PER_BATCH,
        )
        self.candidate_rebase_service = CandidateRebaseService(
            db,
            self.llm,
            max_evidence=REVISION_CANDIDATE_MAX_EVIDENCE,
        )
        self.revision_commit_service = RevisionCommitService(
            db,
            auto_min_utility=REVISION_AUTO_APPLY_MIN_UTILITY,
            auto_budget_hours=REVISION_AUTO_APPLY_BUDGET_HOURS,
            auto_per_hypothesis=REVISION_AUTO_APPLY_PER_HYPOTHESIS,
        )
        # Phase 5.4 is deterministic and idempotent. This also upgrades an
        # already-running Phase 5.3 database where r2/events existed before
        # RevisionSignal resolution metadata was introduced.
        self.revision_reconciliation_service.reconcile_existing_events()
        self.hypothesis_service.recalculate_all()

        self.personal_memory_gate = PersonalMemoryGate(self.llm)
        self.learning_eligibility_gate = LearningEligibilityGate(self.llm)
        self.experience_gate = ExperienceGate(self.llm)
        self.context_builder = ContextBuilder(self.llm, model_name=DEEPSEEK_MODEL)
        self.reflection_service = ReflectionService(self.llm)
        self.evaluation_service = EvaluationService(self.llm)
        self.post_turn_assessor = PostTurnAssessor(self.llm)
        self.skill_registry = build_default_skill_registry(self.personal_memory_service)
        self.tool_approval_store = ApprovalStore()
        self.hook_manager = (
            HookManager(history_limit=EVOME_HOOK_HISTORY_LIMIT)
            if EVOME_HOOKS_ENABLED
            else None
        )
        if self.hook_manager is not None and EVOME_HOOK_PROTECT_SECRETS:
            self.hook_manager.register(
                "before_tool",
                build_sensitive_path_guard(EVOME_TOOL_WORKSPACE),
                name="protect_sensitive_paths",
                priority=100,
            )

        self.hypothesis_retriever = HybridHypothesisRetriever(
            db, qdrant_path=EVOME_QDRANT_PATH
        )
        self.hypothesis_relation_judge = HypothesisRelationJudge(self.llm)
        self.hypothesis_aggregator = HypothesisAggregator(
            db,
            relation_judge=self.hypothesis_relation_judge,
            retriever=self.hypothesis_retriever,
            top_k=HYPOTHESIS_TOP_K,
            score_threshold=HYPOTHESIS_SCORE_THRESHOLD,
            min_relation_confidence=HYPOTHESIS_RELATION_CONFIDENCE,
        )
        self.hypothesis_aggregator.sync_index()

        self.strategy = self._build_strategy(
            db=db,
            llm=self.llm,
            personal_memory_service=self.personal_memory_service,
            audit_service=self.audit_service,
            evolution_memory_service=self.evolution_memory_service,
            hypothesis_retriever=self.hypothesis_retriever,
        )
        self._ensure_strategy_record()

    def _build_strategy(
        self,
        *,
        db: Session,
        llm,
        personal_memory_service: PersonalMemoryService,
        audit_service: AuditService,
        evolution_memory_service: EvolutionMemoryService,
        hypothesis_retriever: HybridHypothesisRetriever,
    ) -> DefaultConversationStrategy:
        personal_memory_gate = PersonalMemoryGate(llm)
        learning_eligibility_gate = LearningEligibilityGate(llm)
        experience_gate = ExperienceGate(llm)
        context_builder = ContextBuilder(llm, model_name=DEEPSEEK_MODEL)
        reflection_service = ReflectionService(llm)
        evaluation_service = EvaluationService(llm)
        post_turn_assessor = PostTurnAssessor(llm)
        skill_registry = build_default_skill_registry(personal_memory_service)
        hypothesis_relation_judge = HypothesisRelationJudge(llm)
        hypothesis_aggregator = HypothesisAggregator(
            db,
            relation_judge=hypothesis_relation_judge,
            retriever=hypothesis_retriever,
            top_k=HYPOTHESIS_TOP_K,
            score_threshold=HYPOTHESIS_SCORE_THRESHOLD,
            min_relation_confidence=HYPOTHESIS_RELATION_CONFIDENCE,
        )
        return DefaultConversationStrategy(
            llm=llm,
            audit_service=audit_service,
            personal_memory_service=personal_memory_service,
            personal_memory_gate=personal_memory_gate,
            learning_eligibility_gate=learning_eligibility_gate,
            experience_gate=experience_gate,
            context_builder=context_builder,
            evolution_memory_service=evolution_memory_service,
            reflection_service=reflection_service,
            hypothesis_aggregator=hypothesis_aggregator,
            evaluation_service=evaluation_service,
            post_turn_assessor=post_turn_assessor,
            skill_registry=skill_registry,
            enable_agent_loop=self.enable_agent_loop,
            agent_max_steps=EVOME_AGENT_MAX_STEPS,
            tool_workspace=EVOME_TOOL_WORKSPACE,
            enable_bash_tool=EVOME_BASH_TOOL_ENABLED,
            bash_timeout_seconds=EVOME_BASH_TIMEOUT_SECONDS,
            tool_max_output_chars=EVOME_TOOL_MAX_OUTPUT_CHARS,
            bash_permission=EVOME_BASH_PERMISSION,
            approval_store=self.tool_approval_store,
            hook_manager=self.hook_manager,
        )

    @property
    def strategy_full_name(self) -> str:
        return f"{self.strategy.name}_v{self.strategy.version}"

    @property
    def last_background_job_id(self) -> str | None:
        return self._last_background_job_id

    def _ensure_strategy_record(self) -> StrategyRecord:
        stmt = select(StrategyRecord).where(
            StrategyRecord.name == self.strategy.name,
            StrategyRecord.version == self.strategy.version,
        )
        record = self.db.scalar(stmt)
        if record is None:
            record = StrategyRecord(
                name=self.strategy.name,
                version=self.strategy.version,
                status="active",
                description=(
                    "Fast Path: retrieve Personal Memory -> stream response; "
                    "Slow Path: post-turn assess -> audit -> memory/experience/reflection -> "
                    "hypothesis/revision-signal processing"
                ),
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
        return record

    @staticmethod
    def _record_strategy_result_for_session(
        db: Session,
        *,
        strategy_name: str,
        strategy_version: int,
        score: float,
        success: bool,
    ) -> None:
        stmt = select(StrategyRecord).where(
            StrategyRecord.name == strategy_name,
            StrategyRecord.version == strategy_version,
        )
        record = db.scalar(stmt)
        if record is None:
            return
        record.run_count += 1
        record.total_score += score
        if success:
            record.success_count += 1
        db.commit()

    def _record_strategy_result(self, score: float, success: bool) -> None:
        self._record_strategy_result_for_session(
            self.db,
            strategy_name=self.strategy.name,
            strategy_version=self.strategy.version,
            score=score,
            success=success,
        )

    def _new_background_llm(self):
        # Production uses a distinct client so foreground streaming and slow-loop
        # JSON calls do not share mutable client state. Tests/custom adapters are
        # lightweight and may safely reuse the supplied fake object.
        if isinstance(self.llm, DeepSeekClient):
            return DeepSeekClient()
        return self.llm

    @staticmethod
    def _result_from_state(state: dict[str, Any], *, strategy_name: str) -> dict[str, Any]:
        evaluation = state.get("evaluation", {})
        eligibility = state.get("learning_eligibility", {})
        return {
            "response": state.get("response", ""),
            "strategy_name": strategy_name,
            "evaluation_score": float(evaluation.get("score", 0.0)),
            "evaluation": evaluation,
            "learning_category": str(eligibility.get("category", "unknown")),
            "learning_eligible": bool(eligibility.get("learning_eligible", False)),
            "allow_personal_memory": bool(eligibility.get("allow_personal_memory", False)),
            "allow_evolution": bool(eligibility.get("allow_evolution", False)),
            "learning_gate_source": str(eligibility.get("decision_source", "unknown")),
            "learning_gate_reason": str(eligibility.get("reason", "")),
            "interaction_id": state.get("interaction_id"),
            "personal_memories_saved": state.get("personal_memories_saved", 0),
            "experience_saved": bool(state.get("experience_saved", False)),
            "experience_gate_source": str(
                state.get("experience_gate", {}).get("decision_source", "unknown")
            ),
            "reflection_saved": bool(state.get("reflection_saved", False)),
            "context_data": state.get("context_data", {}),
            "hypothesis_id": state.get("hypothesis_id"),
            "hypothesis_action": state.get("hypothesis_action", "skipped"),
            "revision_signal_count": int(state.get("revision_signal_count", 0)),
            "revision_analysis_count": int(state.get("revision_analysis_count", 0)),
            "revision_proposal_count": int(state.get("revision_proposal_count", 0)),
            "proposal_normalization_count": int(state.get("proposal_normalization_count", 0)),
            "candidate_revision_count": int(state.get("candidate_revision_count", 0)),
            "revision_auto_apply_count": int(state.get("revision_auto_apply_count", 0)),
            "agent_steps": state.get("agent_steps", []),
            "agent_termination_reason": state.get("agent_termination_reason", ""),
            "tool_error_count": int(state.get("tool_error_count", 0)),
        }

    def run(self, user_input: str) -> dict:
        """Legacy synchronous path retained for tests and batch callers.

        Phase 5.7A stops at RevisionSignal creation. Revision analysis, policy
        normalization, candidate validation and auto-apply are explicit/batch jobs.
        """
        state = self.strategy.invoke(user_input=user_input, session_id=self.session_id)
        eligibility = state.get("learning_eligibility", {})
        state["revision_analysis_count"] = 0
        state["revision_proposal_count"] = 0
        state["proposal_normalization_count"] = 0
        state["candidate_revision_count"] = 0
        state["revision_auto_apply_count"] = 0
        if eligibility.get("allow_evolution"):
            evaluation = state.get("evaluation", {})
            self._record_strategy_result(
                float(evaluation.get("score", 0.0)),
                bool(evaluation.get("success", False)),
            )
        return self._result_from_state(state, strategy_name=self.strategy_full_name)

    def _process_revision_analysis_job(self) -> dict[str, Any]:
        """Run pending Phase 4 analyses in the background without a user turn."""
        with self._background_session_factory() as db:
            llm = self._new_background_llm()
            service = RevisionAnalysisService(
                db,
                llm,
                min_independent_signals=REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS,
            )
            results = service.analyze_pending(
                max_analyses=REVISION_ANALYSIS_MAX_PER_RUN
            )
            return {
                "revision_analysis_count": len(results),
                "revision_proposal_count": sum(
                    1 for item in results if item.proposal is not None
                ),
                "analysis_ids": [item.analysis.id for item in results],
                "proposal_ids": [
                    item.proposal.id for item in results if item.proposal is not None
                ],
            }

    def _process_candidate_validation_job(self) -> dict[str, Any]:
        """Run pending Phase 5 candidate validation in the background."""
        with self._background_session_factory() as db:
            llm = self._new_background_llm()
            normalizer = ProposalPolicyNormalizer(db, llm)
            normalization_results = normalizer.normalize_pending(
                max_normalizations=REVISION_CANDIDATE_MAX_PER_RUN
            )
            service = CandidateRevisionService(
                db,
                llm,
                max_evidence=REVISION_CANDIDATE_MAX_EVIDENCE,
            )
            results = service.evaluate_pending(
                max_candidates=REVISION_CANDIDATE_MAX_PER_RUN
            )
            commit_service = RevisionCommitService(
                db,
                auto_min_utility=REVISION_AUTO_APPLY_MIN_UTILITY,
                auto_budget_hours=REVISION_AUTO_APPLY_BUDGET_HOURS,
                auto_per_hypothesis=REVISION_AUTO_APPLY_PER_HYPOTHESIS,
            )
            commit_results = commit_service.auto_apply_ready(
                max_candidates=REVISION_AUTO_APPLY_MAX_PER_RUN
            )
            retriever = self.hypothesis_retriever.for_session(db)
            for item in commit_results:
                if item.event is None:
                    continue
                hypothesis = db.get(StrategyHypothesis, item.event.hypothesis_id)
                if hypothesis is not None:
                    retriever.index(hypothesis)
            return {
                "proposal_normalization_count": len(normalization_results),
                "normalized_proposal_ids": [
                    item.proposal.id
                    for item in normalization_results
                    if item.proposal is not None
                ],
                "candidate_revision_count": len(results),
                "candidate_ids": [item.candidate.id for item in results],
                "ready_to_apply": sum(
                    1 for item in results if item.candidate.status == "ready_to_apply"
                ),
                "needs_review": sum(
                    1 for item in results if item.candidate.status == "needs_review"
                ),
                "rejected": sum(
                    1 for item in results if item.candidate.status == "rejected"
                ),
                "auto_applied": len(commit_results),
                "revision_event_ids": [
                    item.event.id for item in commit_results if item.event is not None
                ],
                "new_revisions": [
                    item.event.to_revision for item in commit_results if item.event is not None
                ],
            }

    def _process_ancestor_revalidation_job(self) -> dict[str, Any]:
        """Revalidate ancestor-open signals against current heads in the background."""
        with self._background_session_factory() as db:
            llm = self._new_background_llm()
            service = AncestorSignalRevalidationService(
                db,
                llm,
                max_signals_per_batch=REVISION_REVALIDATION_MAX_SIGNALS_PER_BATCH,
            )
            results = service.revalidate_pending(
                max_hypotheses=REVISION_REVALIDATION_MAX_HYPOTHESES_PER_RUN
            )
            hypothesis_service = HypothesisService(db)
            for item in results:
                hypothesis_service.recalculate(item.hypothesis_id, commit=False)
            if results:
                db.commit()
            return {
                "revalidation_batches": len(results),
                "hypothesis_ids": [item.hypothesis_id for item in results],
                "target_revisions": [item.target_revision for item in results],
                "record_ids": [rid for item in results for rid in item.record_ids],
                "rebased_signal_ids": [sid for item in results for sid in item.rebased_signal_ids],
                "resolved_signal_ids": [sid for item in results for sid in item.resolved_signal_ids],
                "superseded_signal_ids": [sid for item in results for sid in item.superseded_signal_ids],
                "uncertain_signal_ids": [sid for item in results for sid in item.uncertain_signal_ids],
            }

    def _process_candidate_rebase_job(self, candidate_id: int) -> dict[str, Any]:
        """Three-way rebase one stale Candidate onto the current revision."""
        with self._background_session_factory() as db:
            llm = self._new_background_llm()
            service = CandidateRebaseService(
                db,
                llm,
                max_evidence=REVISION_CANDIDATE_MAX_EVIDENCE,
            )
            result = service.rebase_candidate(int(candidate_id))
            return {
                "source_candidate_id": int(candidate_id),
                "candidate_rebase_record_id": int(result.record.id),
                "rebase_status": str(result.record.status),
                "target_revision": int(result.record.target_revision),
                "rebased_proposal_id": result.record.rebased_proposal_id,
                "rebased_candidate_id": result.record.rebased_candidate_id,
                "candidate_status": (result.candidate.status if result.candidate is not None else None),
                "llm_called": bool(result.llm_called),
                "reason": str(result.record.reason),
            }

    def _process_revision_auto_apply_job(self) -> dict[str, Any]:
        """Apply already-validated LOW-risk Candidates without another LLM call."""
        with self._background_session_factory() as db:
            service = RevisionCommitService(
                db,
                auto_min_utility=REVISION_AUTO_APPLY_MIN_UTILITY,
                auto_budget_hours=REVISION_AUTO_APPLY_BUDGET_HOURS,
                auto_per_hypothesis=REVISION_AUTO_APPLY_PER_HYPOTHESIS,
            )
            results = service.auto_apply_ready(
                max_candidates=REVISION_AUTO_APPLY_MAX_PER_RUN
            )
            retriever = self.hypothesis_retriever.for_session(db)
            for item in results:
                if item.event is None:
                    continue
                hypothesis = db.get(StrategyHypothesis, item.event.hypothesis_id)
                if hypothesis is not None:
                    retriever.index(hypothesis)
            return {
                "auto_applied": len(results),
                "candidate_ids": [
                    item.candidate.id for item in results if item.candidate is not None
                ],
                "revision_event_ids": [
                    item.event.id for item in results if item.event is not None
                ],
                "new_revisions": [
                    item.event.to_revision for item in results if item.event is not None
                ],
            }

    def _process_slow_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_evolution = bool(payload.pop("_run_evolution", True))
        with self._background_session_factory() as db:
            llm = self._new_background_llm()
            personal_memory_service = PersonalMemoryService(db)
            audit_service = AuditService(db)
            evolution_memory_service = EvolutionMemoryService(db)
            retriever = self.hypothesis_retriever.for_session(db)
            strategy = self._build_strategy(
                db=db,
                llm=llm,
                personal_memory_service=personal_memory_service,
                audit_service=audit_service,
                evolution_memory_service=evolution_memory_service,
                hypothesis_retriever=retriever,
            )
            state = strategy.process_slow_state(
                dict(payload),
                run_evolution=run_evolution,
            )
            eligibility = state.get("learning_eligibility", {})

            # Phase 5.7A: a user turn may create Experience/Reflection/Hypothesis/
            # RevisionSignal, but it never drains the global revision backlog.
            # Heavy work remains available through explicit queue commands.
            state["revision_analysis_count"] = 0
            state["revision_proposal_count"] = 0
            state["proposal_normalization_count"] = 0
            state["candidate_revision_count"] = 0
            state["revision_auto_apply_count"] = 0

            if run_evolution and eligibility.get("allow_evolution"):
                evaluation = state.get("evaluation", {})
                self._record_strategy_result_for_session(
                    db,
                    strategy_name=strategy.name,
                    strategy_version=strategy.version,
                    score=float(evaluation.get("score", 0.0)),
                    success=bool(evaluation.get("success", False)),
                )
            return self._result_from_state(
                state,
                strategy_name=f"{strategy.name}_v{strategy.version}",
            )

    def stream_run(
        self,
        user_input: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Stream the user-facing answer, then enqueue learning in the background.

        The synchronous critical path is intentionally small: memory recall and
        answer generation. Evaluation/reflection/hypothesis work starts only
        after the complete assistant response exists.
        """
        if self.background_queue is None:
            # Useful for embedders/tests that explicitly disable background mode.
            result = self.run(user_input)
            if on_token and result["response"]:
                on_token(result["response"])
            return {**result, "background_job_id": None, "background_status": "disabled"}

        state = self.strategy.prepare_fast_state(
            user_input=user_input,
            session_id=self.session_id,
        )
        chunks: list[str] = []
        for token in self.strategy.stream_response(state):
            chunks.append(token)
            if on_token is not None:
                on_token(token)

        response = "".join(chunks)
        state["response"] = response

        # Copy only serializable/plain state into the queued closure. ORM objects
        # are never sent across threads.
        payload = dict(state)

        # Phase 5.7C: optionally run the slow evolution path only every Nth turn.
        self._slow_loop_turn_counter += 1
        cadence = EVOME_SLOW_LOOP_EVERY_N_TURNS
        run_slow = cadence <= 1 or (self._slow_loop_turn_counter % cadence) == 0

        payload["_run_evolution"] = run_slow

        job_id = self.background_queue.submit(
            kind="slow_evolution" if run_slow else "post_turn_audit",
            fn=lambda: self._process_slow_payload(payload),
        )
        self._last_background_job_id = job_id

        return {
            "response": response,
            "strategy_name": self.strategy_full_name,
            "background_job_id": job_id,
            "background_status": "queued" if run_slow else "queued_audit_only",
            "relevant_memory_count": len(state.get("relevant_memories", [])),
            "agent_step_count": len(state.get("agent_steps", [])),
            "agent_termination_reason": state.get("agent_termination_reason", ""),
            "tool_error_count": int(state.get("tool_error_count", 0)),
        }

    def list_hook_events(self, limit: int = 20):
        return self.hook_manager.recent(limit=limit) if self.hook_manager else []

    def list_pending_tool_approvals(self):
        return self.tool_approval_store.list_pending()

    def approve_tool_call(self, approval_id: str):
        return self.strategy.agent_loop.executor.approve(approval_id) if self.strategy.agent_loop else None

    def deny_tool_call(self, approval_id: str) -> bool:
        if self.strategy.agent_loop is None:
            return False
        return self.strategy.agent_loop.executor.deny(approval_id)

    def token_usage_summary(self) -> dict[str, Any]:
        return GLOBAL_TOKEN_USAGE.summary()

    def enqueue_revision_analysis(self) -> str | None:
        """Queue Phase 4 analysis explicitly; useful for operator/testing workflows."""
        if self.background_queue is None:
            self._process_revision_analysis_job()
            return None
        job_id = self.background_queue.submit(
            kind="revision_analysis",
            fn=self._process_revision_analysis_job,
        )
        self._last_background_job_id = job_id
        return job_id

    def enqueue_candidate_validation(self) -> str | None:
        """Queue Phase 5 candidate evaluation explicitly."""
        if self.background_queue is None:
            self._process_candidate_validation_job()
            return None
        job_id = self.background_queue.submit(
            kind="candidate_validation",
            fn=self._process_candidate_validation_job,
        )
        self._last_background_job_id = job_id
        return job_id

    def enqueue_revision_auto_apply(self) -> str | None:
        """Queue deterministic Phase 5.3 auto-apply for existing ready Candidates."""
        if self.background_queue is None:
            self._process_revision_auto_apply_job()
            return None
        job_id = self.background_queue.submit(
            kind="revision_auto_apply",
            fn=self._process_revision_auto_apply_job,
        )
        self._last_background_job_id = job_id
        return job_id

    def enqueue_ancestor_revalidation(self) -> str | None:
        """Queue Phase 5.5 ancestor signal revalidation."""
        if self.background_queue is None:
            self._process_ancestor_revalidation_job()
            return None
        job_id = self.background_queue.submit(
            kind="ancestor_revalidation",
            fn=self._process_ancestor_revalidation_job,
        )
        self._last_background_job_id = job_id
        return job_id

    def enqueue_candidate_rebase(self, candidate_id: int) -> str | None:
        """Queue Phase 5.5 stale Candidate rebase and re-evaluation."""
        if self.background_queue is None:
            self._process_candidate_rebase_job(int(candidate_id))
            return None
        job_id = self.background_queue.submit(
            kind="candidate_rebase",
            fn=lambda: self._process_candidate_rebase_job(int(candidate_id)),
        )
        self._last_background_job_id = job_id
        return job_id

    def list_background_jobs(self, limit: int = 10):
        if self.background_queue is None:
            return []
        return self.background_queue.list_recent(limit=limit)

    def wait_for_background(
        self,
        job_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        if self.background_queue is None:
            return None
        target = job_id or self._last_background_job_id
        if target is None:
            return None
        return self.background_queue.wait(target, timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        if self.background_queue is not None:
            self.background_queue.shutdown(wait=wait)

    def list_personal_memories(self):
        # Expire foreground ORM cache so background commits are visible promptly.
        self.db.expire_all()
        return self.personal_memory_service.list_all()

    def list_experiences(self):
        self.db.expire_all()
        return self.evolution_memory_service.list_experiences()

    def list_reflections(self):
        self.db.expire_all()
        return self.evolution_memory_service.list_reflections()

    def list_hypotheses(self):
        self.db.expire_all()
        return self.hypothesis_service.list_all()

    def list_hypothesis_revisions(self, hypothesis_id: int):
        self.db.expire_all()
        return self.hypothesis_service.list_revisions(hypothesis_id)

    def list_revision_signals(self):
        self.db.expire_all()
        return self.revision_signal_service.list_all()

    def count_invalid_revision_signals(self) -> int:
        self.db.expire_all()
        return self.revision_signal_service.count_invalid()

    def get_revision_pressure(self, hypothesis_id: int, revision_number: int | None = None):
        self.db.expire_all()
        return self.revision_reconciliation_service.pressure_summary(
            hypothesis_id, revision_number
        )

    def list_revision_signal_groups(self, hypothesis_id: int | None = None):
        self.db.expire_all()
        return self.revision_aggregation_service.group_signals(hypothesis_id=hypothesis_id)

    def get_evidence_summary(self, hypothesis_id: int, revision_number: int | None = None):
        self.db.expire_all()
        return self.revision_aggregation_service.build_evidence_summary(
            hypothesis_id=hypothesis_id,
            revision_number=revision_number,
        )

    def list_revision_analyses(self):
        self.db.expire_all()
        return self.revision_analysis_service.list_analyses()

    def list_revision_proposals(self):
        self.db.expire_all()
        return self.revision_analysis_service.list_proposals()

    def normalize_proposal(self, proposal_id: int):
        self.db.expire_all()
        return self.policy_normalizer_service.normalize_by_id(int(proposal_id))

    def list_proposal_normalizations(self):
        self.db.expire_all()
        return self.policy_normalizer_service.list_records()

    def get_response_policy(self, candidate_id: int) -> dict[str, Any]:
        self.db.expire_all()
        candidate = self.db.get(CandidateRevision, int(candidate_id))
        if candidate is None:
            raise ValueError(f"CR#{candidate_id} does not exist")
        proposal = self.db.get(RevisionProposal, int(candidate.source_proposal_id))
        if proposal is None or str(proposal.proposal_kind or "raw") != "conditional_policy":
            raise ValueError(f"CR#{candidate_id} is not a conditional-policy candidate")
        policy = (candidate.do_data or {}).get("response_structure")
        if not isinstance(policy, dict):
            raise ValueError(f"CR#{candidate_id} has no response policy")
        return policy

    def list_candidate_revisions(self):
        self.db.expire_all()
        return self.candidate_revision_service.list_candidates()

    def list_signal_revalidations(self, limit: int = 100):
        self.db.expire_all()
        return self.ancestor_revalidation_service.list_records(limit=limit)

    def list_candidate_rebases(self, limit: int = 100):
        self.db.expire_all()
        return self.candidate_rebase_service.list_records(limit=limit)

    def approve_candidate(self, candidate_id: int, *, reason: str = "human_approved"):
        self.db.expire_all()
        result = self.revision_commit_service.approve_candidate(candidate_id, reason=reason)
        if result.event is not None:
            hypothesis = self.hypothesis_service.get(result.event.hypothesis_id)
            if hypothesis is not None:
                self.hypothesis_retriever.index(hypothesis)
        return result

    def reject_candidate(self, candidate_id: int, *, reason: str = "human_rejected"):
        self.db.expire_all()
        return self.revision_commit_service.reject_candidate(candidate_id, reason=reason)

    def rollback_revision(
        self,
        hypothesis_id: int,
        target_revision: int,
        *,
        reason: str = "human_rollback",
    ):
        self.db.expire_all()
        result = self.revision_commit_service.rollback(
            hypothesis_id,
            target_revision,
            reason=reason,
        )
        if result.event is not None:
            hypothesis = self.hypothesis_service.get(result.event.hypothesis_id)
            if hypothesis is not None:
                self.hypothesis_retriever.index(hypothesis)
        return result

    def list_revision_events(self, limit: int = 50):
        self.db.expire_all()
        return self.revision_commit_service.list_events(limit=limit)

    def invalidate_learning_from_interaction(
        self,
        interaction_id: int,
        *,
        reason: str = "operator_marked_non_learning",
    ):
        self.db.expire_all()
        event = self.learning_invalidation_service.invalidate_interaction(
            interaction_id, reason=reason
        )
        # Invalidation may alter current Hypothesis statistics/status. Refresh
        # the semantic index so retrieval sees the recalculated projection.
        for hypothesis_id in (event.event_data or {}).get("recalculated_hypothesis_ids", []):
            hypothesis = self.hypothesis_service.get(int(hypothesis_id))
            if hypothesis is not None:
                self.hypothesis_retriever.index(hypothesis)
        return event

    def list_learning_invalidations(self, limit: int = 50):
        self.db.expire_all()
        return self.learning_invalidation_service.list_events(limit=limit)

    def list_interactions(self, limit: int = 20):
        self.db.expire_all()
        return self.audit_service.list_recent(limit=limit)
