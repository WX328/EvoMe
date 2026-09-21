from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.audit.service import AuditService
from app.background import BackgroundEvolutionQueue
from app.config import (
    DEEPSEEK_MODEL,
    EVOME_QDRANT_PATH,
    HYPOTHESIS_RELATION_CONFIDENCE,
    HYPOTHESIS_SCORE_THRESHOLD,
    HYPOTHESIS_TOP_K,
    REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS,
)
from app.context.builder import ContextBuilder
from app.evaluation.service import EvaluationService
from app.evolution.service import EvolutionMemoryService
from app.gates.experience_gate import ExperienceGate
from app.gates.personal_memory_gate import PersonalMemoryGate
from app.hypothesis.aggregator import HypothesisAggregator
from app.hypothesis.relation_judge import HypothesisRelationJudge
from app.hypothesis.revision_aggregation import RevisionAggregationService
from app.hypothesis.retriever import HybridHypothesisRetriever
from app.hypothesis.revision_signal import RevisionSignalService
from app.hypothesis.service import HypothesisService
from app.llm import DeepSeekClient
from app.memory.service import PersonalMemoryService
from app.models import StrategyRecord
from app.reflection.service import ReflectionService
from app.skills.builtin import build_default_skill_registry
from app.strategy.default_workflow import DefaultConversationStrategy


class EvoMeRuntime:
    def __init__(
        self,
        db: Session,
        llm=None,
        session_id: str | None = None,
        *,
        enable_background: bool = True,
    ) -> None:
        self.db = db
        self.llm = llm or DeepSeekClient()
        self.session_id = session_id or uuid4().hex
        self.enable_background = enable_background

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

        self.audit_service = AuditService(db)
        self.personal_memory_service = PersonalMemoryService(db)
        self.evolution_memory_service = EvolutionMemoryService(db)
        self.hypothesis_service = HypothesisService(db)
        self.revision_signal_service = RevisionSignalService(db)
        self.revision_aggregation_service = RevisionAggregationService(
            db, min_independent_signals=REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS
        )

        self.personal_memory_gate = PersonalMemoryGate(self.llm)
        self.experience_gate = ExperienceGate(self.llm)
        self.context_builder = ContextBuilder(self.llm, model_name=DEEPSEEK_MODEL)
        self.reflection_service = ReflectionService(self.llm)
        self.evaluation_service = EvaluationService(self.llm)
        self.skill_registry = build_default_skill_registry(self.personal_memory_service)

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
        experience_gate = ExperienceGate(llm)
        context_builder = ContextBuilder(llm, model_name=DEEPSEEK_MODEL)
        reflection_service = ReflectionService(llm)
        evaluation_service = EvaluationService(llm)
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
            experience_gate=experience_gate,
            context_builder=context_builder,
            evolution_memory_service=evolution_memory_service,
            reflection_service=reflection_service,
            hypothesis_aggregator=hypothesis_aggregator,
            evaluation_service=evaluation_service,
            skill_registry=skill_registry,
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
                    "Slow Path: evaluate -> audit -> memory/experience/reflection -> "
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
        return {
            "response": state.get("response", ""),
            "strategy_name": strategy_name,
            "evaluation_score": float(evaluation.get("score", 0.0)),
            "evaluation": evaluation,
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
        }

    def run(self, user_input: str) -> dict:
        """Legacy synchronous path retained for tests and batch callers."""
        state = self.strategy.invoke(user_input=user_input, session_id=self.session_id)
        evaluation = state.get("evaluation", {})
        score = float(evaluation.get("score", 0.0))
        success = bool(evaluation.get("success", False))
        self._record_strategy_result(score, success)
        return self._result_from_state(state, strategy_name=self.strategy_full_name)

    def _process_slow_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            state = strategy.process_slow_state(dict(payload))
            evaluation = state.get("evaluation", {})
            score = float(evaluation.get("score", 0.0))
            success = bool(evaluation.get("success", False))
            self._record_strategy_result_for_session(
                db,
                strategy_name=strategy.name,
                strategy_version=strategy.version,
                score=score,
                success=success,
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
        job_id = self.background_queue.submit(
            kind="slow_evolution",
            fn=lambda: self._process_slow_payload(payload),
        )
        self._last_background_job_id = job_id

        return {
            "response": response,
            "strategy_name": self.strategy_full_name,
            "background_job_id": job_id,
            "background_status": "queued",
            "relevant_memory_count": len(state.get("relevant_memories", [])),
        }

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

    def list_revision_signal_groups(self, hypothesis_id: int | None = None):
        self.db.expire_all()
        return self.revision_aggregation_service.group_signals(hypothesis_id=hypothesis_id)

    def get_evidence_summary(self, hypothesis_id: int, revision_number: int | None = None):
        self.db.expire_all()
        return self.revision_aggregation_service.build_evidence_summary(
            hypothesis_id=hypothesis_id,
            revision_number=revision_number,
        )

    def list_interactions(self, limit: int = 20):
        self.db.expire_all()
        return self.audit_service.list_recent(limit=limit)
