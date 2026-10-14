from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent import PlanStore
from app.agent.task_system import TaskStore
from app.agent.background_tasks import BackgroundTaskService
from app.agent.cron_store import ScheduleStore
from app.agent.cron_scheduler import CronScheduler
from app.agent.team_runtime import TeamRuntime
from app.agent.integrated_harness import IntegratedHarness
from app.agent.workflow_store import WorkflowStore
from app.agent.workflow_runtime import WorkflowRuntime
from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.mcp_plugin import MCPPluginManager
from app.agent.memory import AgentMemoryStore
from app.agent.memory_pipeline import AgentMemoryPipeline
from app.audit.service import AuditService
from app.background import BackgroundEvolutionQueue
from app.config import (
    DEEPSEEK_MODEL,
    EVOME_AGENT_LOOP_ENABLED,
    EVOME_TASK_SYSTEM_ENABLED,
    EVOME_TASK_DIRECTORY,
    EVOME_S11_BACKGROUND_ENABLED,
    EVOME_S11_MAX_STEPS,
    EVOME_S12_CRON_ENABLED,
    EVOME_S12_CRON_DIR,
    EVOME_S12_POLL_SECONDS,
    EVOME_S13_TEAMS_ENABLED,
    EVOME_S13_TEAM_DIR,
    EVOME_S13_MAX_STEPS,
    EVOME_S14_MCP_ENABLED,
    EVOME_S14_MCP_CONFIG,
    EVOME_S14_MCP_TIMEOUT,
    EVOME_S15_HARNESS_ENABLED,
    EVOME_S15_MAX_EVENTS,
    EVOME_S15_EVENT_CONTEXT_CHARS,
    EVOME_S16_WORKFLOW_ENABLED,
    EVOME_S16_WORKFLOW_DIR,
    EVOME_S16_MAX_AGENT_CALLS,
    EVOME_S16_MAX_PARALLEL,
    EVOME_AGENT_MEMORY_ENABLED,
    EVOME_AGENT_MEMORY_DIR,
    EVOME_AGENT_MEMORY_AUTO_STAGE,
    EVOME_AGENT_MEMORY_RECALL_CHARS,
    EVOME_AGENT_MAX_STEPS,
    EVOME_PLANNING_ENABLED,
    EVOME_PLAN_MAX_TASKS,
    EVOME_PLAN_STEP_MAX_STEPS,
    EVOME_PLAN_STEP_MAX_CONTINUATIONS,
    EVOME_PLAN_MAX_REPLANS,
    EVOME_PLAN_AUTO_REPLAN,
    EVOME_PLAN_REQUIRE_FILE_EVIDENCE,
    EVOME_SUBAGENT_ENABLED,
    EVOME_SUBAGENT_MAX_STEPS,
    EVOME_SUBAGENT_MAX_CALLS,
    EVOME_CONTEXT_COMPACT_ENABLED,
    EVOME_CONTEXT_MAX_CHARS,
    EVOME_CONTEXT_KEEP_RECENT_MESSAGES,
    EVOME_SKILL_LOADING_ENABLED,
    EVOME_SKILL_DIRECTORY,
    EVOME_SKILL_MAX_CONTENT_CHARS,
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
        enable_scheduler: bool = False,
        enable_teams: bool = False,
        enable_mcp: bool = False,
        enable_harness: bool = False,
        enable_workflows: bool = False,
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
        self.plan_store = PlanStore()
        # Tests / embedded clients never auto-start a team from .env alone.
        self.team_runtime = None
        self._pending_team_events: list[dict[str, Any]] = []
        team_enabled = (
            EVOME_S13_TEAMS_ENABLED and enable_teams and enable_background
            and self.enable_agent_loop and EVOME_TASK_SYSTEM_ENABLED
        )
        self._team_root = (
            (Path(EVOME_TOOL_WORKSPACE).resolve() / EVOME_S13_TEAM_DIR).resolve()
            if team_enabled else None
        )
        # Runtime tests and explicit offline/injected-LLM configurations can
        # disable the Agent Loop or background worker independently of .env.
        # Never start a scheduler in those modes merely because the env flag
        # is set; normal CLI access reports that S12 is unavailable.
        s12_active = (
            EVOME_S12_CRON_ENABLED and enable_scheduler and enable_background
            and self.enable_agent_loop and EVOME_TASK_SYSTEM_ENABLED
            and EVOME_S11_BACKGROUND_ENABLED
        )
        self.schedule_store = (
            ScheduleStore(
                workspace_root=EVOME_TOOL_WORKSPACE,
                directory=EVOME_S12_CRON_DIR,
            )
            if s12_active else None
        )
        self.workflow_store: WorkflowStore | None = (
            WorkflowStore(
                workspace_root=EVOME_TOOL_WORKSPACE,
                directory=EVOME_S16_WORKFLOW_DIR,
            )
            if EVOME_S16_WORKFLOW_ENABLED and enable_workflows
            and enable_background and self.enable_agent_loop else None
        )
        self.task_store = (
            TaskStore(
                workspace_root=EVOME_TOOL_WORKSPACE,
                directory=EVOME_TASK_DIRECTORY,
            )
            if EVOME_TASK_SYSTEM_ENABLED and self.enable_agent_loop else None
        )
        self.agent_memory_store = (
            AgentMemoryStore(
                workspace_root=EVOME_TOOL_WORKSPACE,
                directory=EVOME_AGENT_MEMORY_DIR,
                max_recall_chars=EVOME_AGENT_MEMORY_RECALL_CHARS,
            )
            if EVOME_AGENT_MEMORY_ENABLED and self.enable_agent_loop else None
        )
        self.agent_memory_pipeline = (
            AgentMemoryPipeline(store=self.agent_memory_store, llm=self.llm)
            if self.agent_memory_store is not None else None
        )
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
        self.workflow_runtime: WorkflowRuntime | None = None
        if self.workflow_store is not None and self.strategy.agent_loop is not None:
            self.workflow_runtime = WorkflowRuntime(
                store=self.workflow_store,
                llm_factory=self._new_background_llm,
                workspace_root=EVOME_TOOL_WORKSPACE,
                hook_manager=self.hook_manager,
                max_calls=EVOME_S16_MAX_AGENT_CALLS,
                max_parallel=EVOME_S16_MAX_PARALLEL,
            )
            self.strategy.tool_registry.register_tool(ToolDefinition(
                name="Workflow",
                description=(
                    "运行已由宿主预先注册的只读、多阶段 Workflow。"
                    "需要 S03 明确人工批准。name=review-files，args 的 paths "
                    "只接受工作区相对文件路径数组；续跑需 resume_from_run_id。"
                    "不能传入代码、动态脚本或 shell 命令。"
                ),
                input_schema={
                    "name": {"type": "string"},
                    "args": {"type": "object"},
                    "resume_from_run_id": {"type": "string"},
                },
            ))
            self.strategy.tool_registry.register_implementation(ToolImplementation(
                name="s16_workflow_v1", tool_name="Workflow",
                handler=self.workflow_runtime.run_tool,
            ))
        self.mcp_manager: MCPPluginManager | None = None
        if EVOME_S14_MCP_ENABLED and enable_mcp and self.strategy.agent_loop is not None:
            self.mcp_manager = MCPPluginManager(
                workspace_root=EVOME_TOOL_WORKSPACE,
                registry=self.strategy.tool_registry,
                config_file=EVOME_S14_MCP_CONFIG,
                approval_revoke=self._revoke_mcp_approvals,
                timeout=EVOME_S14_MCP_TIMEOUT,
            )
        self.task_background_service = (
            BackgroundTaskService(
                store=self.task_store,
                llm_factory=self._new_background_llm,
                workspace_root=EVOME_TOOL_WORKSPACE,
                hook_manager=self.hook_manager,
                memory_root=(
                    self.agent_memory_store.root
                    if self.agent_memory_store is not None else None
                ),
                schedule_root=(
                    self.schedule_store.root if self.schedule_store is not None else None
                ),
                team_root=self._team_root,
                workflow_root=(
                    self.workflow_store.root if self.workflow_store is not None else None
                ),
                max_steps=EVOME_S11_MAX_STEPS,
                max_output_chars=EVOME_TOOL_MAX_OUTPUT_CHARS,
            )
            if (enable_background and EVOME_S11_BACKGROUND_ENABLED
                and self.task_store is not None)
            else None
        )
        if team_enabled and self.task_store is not None:
            self.team_runtime = TeamRuntime(
                store=self.task_store,
                llm_factory=self._new_background_llm,
                workspace_root=EVOME_TOOL_WORKSPACE,
                directory=EVOME_S13_TEAM_DIR,
                hook_manager=self.hook_manager,
                workflow_root=(
                    self.workflow_store.root if self.workflow_store is not None else None
                ),
                max_steps=EVOME_S13_MAX_STEPS,
            )
        self.strategy.team_event_provider = self._take_team_event_context
        self.cron_scheduler = (
            CronScheduler(
                store=self.schedule_store,
                worker=self.task_background_service,
                poll_seconds=EVOME_S12_POLL_SECONDS,
            )
            if self.schedule_store is not None
            and self.task_background_service is not None else None
        )
        if self.cron_scheduler is not None:
            self.cron_scheduler.start()  # skip downtime triggers, then poll
        # The coordinator has no execution authority; it collects status
        # snapshots only. Embedded/test runtimes do not opt in by default.
        self.harness: IntegratedHarness | None = None
        if (
            EVOME_S15_HARNESS_ENABLED and enable_harness
            and self.strategy.agent_loop is not None
        ):
            self.harness = IntegratedHarness(
                agent=self.strategy.agent_loop,
                background_queue=self.background_queue,
                task_worker=self.task_background_service,
                cron_scheduler=self.cron_scheduler,
                team_runtime=self.team_runtime,
                mcp_manager=self.mcp_manager,
                approval_store=self.tool_approval_store,
                workflow_runtime=self.workflow_runtime,
                on_team_message=self._remember_team_event,
                max_events=EVOME_S15_MAX_EVENTS,
                max_context_chars=EVOME_S15_EVENT_CONTEXT_CHARS,
            )
            self.strategy.harness_context_provider = self.harness.context_messages
            self.strategy.harness_delivery_callback = self.harness.settle_turn

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
            planning_enabled=EVOME_PLANNING_ENABLED and self.enable_agent_loop,
            plan_store=self.plan_store,
            plan_max_tasks=EVOME_PLAN_MAX_TASKS,
            plan_step_max_steps=EVOME_PLAN_STEP_MAX_STEPS,
            plan_step_max_continuations=EVOME_PLAN_STEP_MAX_CONTINUATIONS,
            plan_max_replans=EVOME_PLAN_MAX_REPLANS,
            plan_auto_replan=EVOME_PLAN_AUTO_REPLAN,
            plan_require_file_evidence=EVOME_PLAN_REQUIRE_FILE_EVIDENCE,
            subagent_enabled=EVOME_SUBAGENT_ENABLED and self.enable_agent_loop,
            subagent_max_steps=EVOME_SUBAGENT_MAX_STEPS,
            subagent_max_calls=EVOME_SUBAGENT_MAX_CALLS,
            skill_loading_enabled=EVOME_SKILL_LOADING_ENABLED and self.enable_agent_loop,
            skill_directory=EVOME_SKILL_DIRECTORY,
            skill_max_content_chars=EVOME_SKILL_MAX_CONTENT_CHARS,
            context_compact_enabled=EVOME_CONTEXT_COMPACT_ENABLED and self.enable_agent_loop,
            context_max_chars=EVOME_CONTEXT_MAX_CHARS,
            context_keep_recent_messages=EVOME_CONTEXT_KEEP_RECENT_MESSAGES,
            agent_memory_store=self.agent_memory_store,
            task_store=self.task_store,
            protected_schedule_directory=(
                self.schedule_store.root if self.schedule_store is not None else None
            ),
            protected_team_directory=(
                self._team_root if hasattr(self, "_team_root") else None
            ),
            protected_workflow_directory=(
                self.workflow_store.root if self.workflow_store is not None else None
            ),
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
        result = self._result_from_state(state, strategy_name=self.strategy_full_name)
        self._maybe_stage_agent_memory(
            user_input, termination=state.get("agent_termination_reason", ""),
        )
        return result

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
        self._maybe_stage_agent_memory(
            user_input, termination=state.get("agent_termination_reason", ""),
        )

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

    def _planner(self):
        agent = self.strategy.agent_loop
        if agent is None or agent.planner is None:
            raise ValueError("S05 Planning 未启用；请检查 EVOME_PLANNING_ENABLED=1")
        return agent.planner

    def list_plans(self):
        return self._planner().list()

    def get_plan(self, plan_id: str):
        return self._planner().get(plan_id)

    def describe_plan(self, plan_id: str, *, detail: bool = False) -> str:
        planner = self._planner()
        return planner.describe(planner.get(plan_id), detail=detail)

    def create_plan(self, goal: str):
        planner = self._planner()
        plan = planner.create_from_llm(goal)
        return planner.run(plan.id)

    def resume_plan(self, plan_id: str):
        return self._planner().run(plan_id)

    def replan(self, plan_id: str):
        return self._planner().replan(plan_id)

    def cancel_plan(self, plan_id: str):
        planner = self._planner()
        plan = planner.get(plan_id)
        for step in plan.steps:
            if step.approval_id:
                self.strategy.agent_loop.executor.deny(step.approval_id)
        return planner.cancel(plan_id)

    def run_research_subagent(self, task: str):
        agent = self.strategy.agent_loop
        if agent is None or agent.subagent is None:
            raise ValueError("S06 Subagent 未启用；检查 EVOME_AGENT_LOOP_ENABLED 与 EVOME_SUBAGENT_ENABLED")
        # Always enter via the parent's real ToolExecutor so permission checks
        # and S04 hooks cannot be bypassed by CLI commands.
        return agent.executor.execute("delegate_research", {"task": task})

    def list_instruction_skills(self) -> list[dict[str, str]]:
        agent = self.strategy.agent_loop
        if agent is None or agent.skill_loader is None:
            raise ValueError("S07 Skill Loading 未启用；检查 EVOME_AGENT_LOOP_ENABLED / EVOME_SKILL_LOADING_ENABLED")
        return agent.skill_loader.catalog()

    def load_instruction_skill(self, name: str):
        agent = self.strategy.agent_loop
        if agent is None or agent.skill_loader is None:
            raise ValueError("S07 Skill Loading 未启用；检查 EVOME_AGENT_LOOP_ENABLED / EVOME_SKILL_LOADING_ENABLED")
        # Use the normal S03 + S04 tool boundary, never call the loader
        # directly from the operator CLI.
        return agent.executor.execute("load_skill", {"name": name})

    def context_compaction_stats(self) -> dict:
        agent = self.strategy.agent_loop
        if agent is None or agent.context_compactor is None:
            return {"enabled": False}
        return agent.context_compactor.stats()

    def _agent_memory(self) -> AgentMemoryPipeline:
        if self.agent_memory_pipeline is None:
            raise ValueError(
                "S09 Agent Memory 未启用；检查 EVOME_AGENT_LOOP_ENABLED / "
                "EVOME_AGENT_MEMORY_ENABLED"
            )
        return self.agent_memory_pipeline

    def _maybe_stage_agent_memory(self, user_input: str, *, termination: str) -> None:
        if (
            not EVOME_AGENT_MEMORY_AUTO_STAGE
            or self.agent_memory_pipeline is None
            or termination != "final_answer"
        ):
            return
        try:
            self.agent_memory_pipeline.propose(user_input)
        except Exception:
            # An optional extraction failure must never invalidate the user
            # response, approval state, existing SQLite memory, or task audit.
            pass

    def list_agent_memories(self):
        return self._agent_memory().store.catalog()

    def recall_agent_memories(self, query: str):
        # Read through the same S03/S04 ToolExecutor boundary as Agent Loop.
        agent = self.strategy.agent_loop
        if agent is None or agent.agent_memory_store is None:
            raise ValueError("S09 Agent Memory 未启用")
        return agent.executor.execute("recall_agent_memory", {"query": query})

    def propose_agent_memories(self, user_input: str):
        return self._agent_memory().propose(user_input)

    def pending_agent_memories(self):
        return self._agent_memory().list_pending()

    def approve_agent_memory(self, candidate_id: str):
        # Operator-only: no LLM or Agent tool may persist a candidate.
        return self._agent_memory().confirm(candidate_id)

    def deny_agent_memory(self, candidate_id: str) -> bool:
        return self._agent_memory().deny(candidate_id)

    def preview_agent_memory_consolidation(self):
        return self._agent_memory().store.preview_consolidation()

    def consolidate_agent_memories(self):
        # Explicit operator-only action with snapshot/rollback.
        return self._agent_memory().store.consolidate_duplicates()

    def _durable_tasks(self) -> TaskStore:
        if self.task_store is None:
            raise ValueError(
                "S10 Task System 未启用；检查 EVOME_AGENT_LOOP_ENABLED "
                "和 EVOME_TASK_SYSTEM_ENABLED"
            )
        return self.task_store

    def list_durable_tasks(self):
        return self._durable_tasks().list_tasks()

    def ready_durable_tasks(self):
        return self._durable_tasks().ready_tasks()

    def get_durable_task(self, task_id: str):
        return self._durable_tasks().get_task(task_id)

    def create_durable_task(self, title: str):
        return self._durable_tasks().create_task(title)

    def add_durable_dependency(self, task_id: str, dependency_id: str):
        return self._durable_tasks().add_dependency(task_id, dependency_id)

    def claim_durable_task(self, task_id: str, owner: str = "operator"):
        return self._durable_tasks().claim_task(task_id, owner)

    def complete_durable_task(self, task_id: str, owner: str = "operator"):
        return self._durable_tasks().complete_task(task_id, owner)

    def release_durable_task(self, task_id: str, owner: str = "operator"):
        return self._durable_tasks().release_task(task_id, owner)

    def _task_worker(self) -> BackgroundTaskService:
        if self.task_background_service is None:
            raise ValueError(
                "S11 Background Tasks 未启用：请检查 S10 与 "
                "EVOME_S11_BACKGROUND_ENABLED=1"
            )
        return self.task_background_service

    def enqueue_background_task(self, task_id: str):
        # Operator-requested read-only work only. This never executes
        # arbitrary LLM-proposed commands or background writes.
        return self._task_worker().submit(task_id)

    def background_task_status(self, task_id: str):
        return self._task_worker().get_status(task_id)

    def list_background_tasks(self):
        return self._task_worker().list_status()

    def wait_for_durable_background_task(self, task_id: str, timeout: float = 30):
        return self._task_worker().wait(task_id, timeout=timeout)

    def recover_background_task(self, task_id: str):
        return self._task_worker().recover(task_id)

    def _cron(self) -> CronScheduler:
        if self.cron_scheduler is None:
            raise ValueError(
                "S12 Cron Scheduler 未启用；检查 Agent Loop、S10、S11 和 "
                "EVOME_S12_CRON_ENABLED=1"
            )
        return self.cron_scheduler

    def create_cron_at(self, *, at: str, title: str):
        return self._cron().create_at(at=at, title=title)

    def create_cron_recurring(self, *, expression: str, utc_offset: str, max_runs: int, title: str):
        return self._cron().create_cron(
            expression=expression, utc_offset=utc_offset,
            max_runs=max_runs, title=title,
        )

    def create_cron_session(self, *, seconds: float, title: str):
        return self._cron().create_session(delay_seconds=seconds, title=title)

    def list_cron_schedules(self):
        return self._cron().list()

    def show_cron_schedule(self, schedule_id: str):
        return self._cron().get(schedule_id)

    def pause_cron_schedule(self, schedule_id: str):
        return self._cron().pause(schedule_id)

    def resume_cron_schedule(self, schedule_id: str):
        return self._cron().resume(schedule_id)

    def cancel_cron_schedule(self, schedule_id: str):
        return self._cron().cancel(schedule_id)

    def poll_cron_schedules(self):
        return self._cron().tick()

    def cron_events(self):
        return self._cron().events()

    def _team(self) -> TeamRuntime:
        if self.team_runtime is None:
            raise ValueError(
                "S13 Agent Teams 未启用；检查 EVOME_S13_TEAMS_ENABLED=1 "
                "及 Agent Loop / S10 配置"
            )
        return self.team_runtime

    def team_propose(self, members: list[tuple[str, str]]):
        return self._team().propose(members)

    def team_approve(self, proposal_id: str):
        return self._team().approve(proposal_id)

    def team_reject(self, proposal_id: str):
        return self._team().reject(proposal_id)

    def team_resume(self, team_id: str):
        return self._team().resume(team_id)

    def team_status(self):
        return self._team().status()

    def team_bind(self, task_id: str, member: str, worktree: str = ""):
        return self._team().bind(task_id, member, worktree=worktree)

    def team_message(self, member: str, content: str):
        return self._team().send(member, content)

    def team_plan_request(self, member: str, content: str):
        return self._team().request_plan(member, content)

    def team_plan_respond(self, member: str, approved: bool):
        return self._team().respond_plan(member, approved)

    def team_stop_member(self, member: str):
        return self._team().request_shutdown(member)

    def team_recover(self, task_id: str):
        return self._team().recover(task_id)

    def team_shutdown(self):
        return self._team().shutdown()

    def _remember_team_event(self, item: dict[str, Any]) -> None:
        # Preserve the existing /team-inbox operator view even when S15
        # owns the main inbound notification stream.
        self._pending_team_events.append(item)
        self._pending_team_events = self._pending_team_events[-25:]

    def poll_team_events(self):
        if self.harness is not None:
            return [
                item for item in self.poll_harness_events()
                if item["source"] == "s13_team"
            ]
        if self.team_runtime is None:
            return []
        events = self.team_runtime.consume_lead_inbox()
        for item in events:
            self._remember_team_event(item)
        return events

    def team_inbox(self):
        self.poll_team_events()
        return list(self._pending_team_events)

    def _take_team_event_context(self) -> str:
        if not self._pending_team_events:
            return ""
        events = self._pending_team_events[:12]
        self._pending_team_events = self._pending_team_events[12:]
        lines = []
        for event in events:
            meta = event.get("metadata", {})
            lines.append(
                f"{event.get('from', '')}: {event.get('type', '')} "
                f"task={meta.get('task_id', '-')} status={meta.get('status', '-')} "
                f"{str(event.get('content', ''))[:500]}"
            )
        return "\n".join(lines)[:5000]

    def _revoke_mcp_approvals(self, names: set[str]) -> None:
        for pending in list(self.tool_approval_store.list_pending()):
            if pending.tool_name in names:
                self.tool_approval_store.pop(pending.id)

    def _mcp(self) -> MCPPluginManager:
        if self.mcp_manager is None:
            raise ValueError(
                "S14 MCP Plugin 未启用；检查 EVOME_S14_MCP_ENABLED=1 "
                "及 EVOME_AGENT_LOOP_ENABLED=1"
            )
        return self.mcp_manager

    def list_mcp_servers(self):
        return self._mcp().servers()

    def list_mcp_tools(self):
        return self._mcp().connected_tools()

    def connect_mcp_server(self, server: str):
        return self._mcp().connect(server)

    def disconnect_mcp_server(self, server: str):
        return self._mcp().disconnect(server)

    def invoke_mcp_tool(self, fullname: str, arguments: dict[str, Any]):
        if fullname not in {item["tool"] for item in self._mcp().connected_tools()}:
            raise ValueError("MCP tool is not currently connected")
        agent = self.strategy.agent_loop
        if agent is None:
            raise ValueError("Agent Loop disabled")
        return agent.executor.execute(fullname, arguments)

    def harness_status(self) -> dict[str, Any]:
        if self.harness is None:
            return {"state": "disabled"}
        return self.harness.status()

    def harness_events(self, limit: int = 20):
        if self.harness is None:
            return []
        return self.harness.events(limit=limit)

    def poll_harness_events(self):
        if self.harness is None:
            return []
        return self.harness.poll()

    def _workflow(self) -> WorkflowRuntime:
        if self.workflow_runtime is None:
            raise ValueError(
                "S16 Workflow 未启用：需要 EVOME_S16_WORKFLOW_ENABLED=1、"
                "Agent Loop 与后台运行"
            )
        return self.workflow_runtime

    def workflow_catalog(self):
        return self._workflow().registry.list()

    def workflow_list(self):
        return self._workflow().list()

    def workflow_show(self, run_id: str):
        return self._workflow().show(run_id)

    def workflow_events(self, run_id: str):
        return self._workflow().events(run_id)

    def workflow_wait(self, run_id: str, seconds: float = 30.0):
        return self._workflow().wait(run_id, timeout=seconds)

    def workflow_run(self, name: str, args: dict[str, Any] | None = None,
                     resume_from_run_id: str = ""):
        if self.strategy.agent_loop is None:
            raise ValueError("Agent Loop disabled")
        kwargs = {"name": name}
        if args is not None:
            kwargs["args"] = args
        if resume_from_run_id:
            kwargs["resume_from_run_id"] = resume_from_run_id
        # Identical S03 Policy / ToolExecutor / S04 Hooks as agent calls.
        return self.strategy.agent_loop.executor.execute("Workflow", kwargs)

    def workflow_unlock(self, run_id: str, *, confirmed: bool):
        return self._workflow().unlock_orphan(run_id, confirmed=confirmed)

    def list_pending_tool_approvals(self):
        return self.tool_approval_store.list_pending()

    def approve_tool_call(self, approval_id: str):
        agent = self.strategy.agent_loop
        if agent is None:
            return None
        result = agent.executor.approve(approval_id)
        if agent.planner is not None:
            agent.planner.on_approval(approval_id, result)
        return result

    def deny_tool_call(self, approval_id: str) -> bool:
        agent = self.strategy.agent_loop
        if agent is None:
            return False
        was_denied = agent.executor.deny(approval_id)
        if was_denied and agent.planner is not None:
            agent.planner.on_approval(approval_id, None, denied=True)
        return was_denied

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
        if self.workflow_runtime is not None:
            self.workflow_runtime.shutdown(wait=wait)
        if self.harness is not None:
            self.harness.shutdown()
        if self.mcp_manager is not None:
            self.mcp_manager.close()
        if self.team_runtime is not None:
            self.team_runtime.shutdown(wait=wait)
        # Stop new triggers before waiting for existing S11 worker jobs.
        if self.cron_scheduler is not None:
            self.cron_scheduler.shutdown(wait=wait)
        if self.task_background_service is not None:
            self.task_background_service.shutdown(wait=wait)
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
