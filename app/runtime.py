from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.audit.service import AuditService
from app.config import (
    DEEPSEEK_MODEL,
    EVOME_QDRANT_PATH,
    HYPOTHESIS_RELATION_CONFIDENCE,
    HYPOTHESIS_SCORE_THRESHOLD,
    HYPOTHESIS_TOP_K,
)
from app.context.builder import ContextBuilder
from app.evaluation.service import EvaluationService
from app.evolution.service import EvolutionMemoryService
from app.gates.experience_gate import ExperienceGate
from app.gates.personal_memory_gate import PersonalMemoryGate
from app.hypothesis.aggregator import HypothesisAggregator
from app.hypothesis.relation_judge import HypothesisRelationJudge
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
    def __init__(self, db: Session, llm=None, session_id: str | None = None) -> None:
        self.db = db
        self.llm = llm or DeepSeekClient()
        self.session_id = session_id or uuid4().hex

        self.audit_service = AuditService(db)
        self.personal_memory_service = PersonalMemoryService(db)
        self.evolution_memory_service = EvolutionMemoryService(db)
        self.hypothesis_service = HypothesisService(db)
        self.revision_signal_service = RevisionSignalService(db)

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

        self.strategy = DefaultConversationStrategy(
            llm=self.llm,
            audit_service=self.audit_service,
            personal_memory_service=self.personal_memory_service,
            personal_memory_gate=self.personal_memory_gate,
            experience_gate=self.experience_gate,
            context_builder=self.context_builder,
            evolution_memory_service=self.evolution_memory_service,
            reflection_service=self.reflection_service,
            hypothesis_aggregator=self.hypothesis_aggregator,
            evaluation_service=self.evaluation_service,
            skill_registry=self.skill_registry,
        )
        self._ensure_strategy_record()

    @property
    def strategy_full_name(self) -> str:
        return f"{self.strategy.name}_v{self.strategy.version}"

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
                    "Retrieve Personal Memory -> Respond -> Evaluate -> Audit -> "
                    "Personal Memory Gate -> Experience Gate -> Context Builder -> "
                    "Reflection -> Strategy Hypothesis Aggregation"
                ),
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
        return record

    def _record_strategy_result(self, score: float, success: bool) -> None:
        stmt = select(StrategyRecord).where(
            StrategyRecord.name == self.strategy.name,
            StrategyRecord.version == self.strategy.version,
        )
        record = self.db.scalar(stmt)
        if record is None:
            return
        record.run_count += 1
        record.total_score += score
        if success:
            record.success_count += 1
        self.db.commit()

    def run(self, user_input: str) -> dict:
        state = self.strategy.invoke(user_input=user_input, session_id=self.session_id)
        evaluation = state.get("evaluation", {})
        score = float(evaluation.get("score", 0.0))
        success = bool(evaluation.get("success", False))
        self._record_strategy_result(score, success)

        return {
            "response": state.get("response", ""),
            "strategy_name": self.strategy_full_name,
            "evaluation_score": score,
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

    def list_personal_memories(self):
        return self.personal_memory_service.list_all()

    def list_experiences(self):
        return self.evolution_memory_service.list_experiences()

    def list_reflections(self):
        return self.evolution_memory_service.list_reflections()

    def list_hypotheses(self):
        return self.hypothesis_service.list_all()

    def list_hypothesis_revisions(self, hypothesis_id: int):
        return self.hypothesis_service.list_revisions(hypothesis_id)

    def list_revision_signals(self):
        return self.revision_signal_service.list_all()

    def count_invalid_revision_signals(self) -> int:
        return self.revision_signal_service.count_invalid()

    def list_interactions(self, limit: int = 20):
        return self.audit_service.list_recent(limit=limit)
