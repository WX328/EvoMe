from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.service import EvaluationService
from app.llm import DeepSeekClient
from app.memory.service import MemoryService
from app.models import StrategyRecord
from app.reflection.service import ReflectionService
from app.skills.builtin import build_default_skill_registry
from app.strategy.default_workflow import DefaultConversationStrategy


class EvoMeRuntime:
    def __init__(self, db: Session, llm=None) -> None:
        self.db = db
        self.llm = llm or DeepSeekClient()

        self.memory_service = MemoryService(db)
        self.reflection_service = ReflectionService(self.llm)
        self.evaluation_service = EvaluationService(self.llm)

        self.skill_registry = build_default_skill_registry(
            self.memory_service
        )

        self.strategy = DefaultConversationStrategy(
            llm=self.llm,
            memory_service=self.memory_service,
            reflection_service=self.reflection_service,
            evaluation_service=self.evaluation_service,
            skill_registry=self.skill_registry,
        )

        self._ensure_strategy_record()

    def _ensure_strategy_record(self) -> StrategyRecord:
        stmt = select(StrategyRecord).where(
            StrategyRecord.name == self.strategy.name
        )
        record = self.db.scalar(stmt)

        if record is None:
            record = StrategyRecord(
                name=self.strategy.name,
                version=1,
                status="active",
                description=(
                    "Recall Memory -> Respond -> Evaluate "
                    "-> Reflect -> Write New Memory"
                ),
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)

        return record

    def _record_strategy_result(
        self,
        score: float,
        success: bool,
    ) -> None:
        stmt = select(StrategyRecord).where(
            StrategyRecord.name == self.strategy.name
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
        state = self.strategy.invoke(user_input)

        evaluation = state.get("evaluation", {})
        score = float(evaluation.get("score", 0.0))
        success = bool(evaluation.get("success", False))

        self._record_strategy_result(score, success)

        return {
            "response": state.get("response", ""),
            "strategy_name": self.strategy.name,
            "evaluation_score": score,
            "evaluation": evaluation,
            "reflection": state.get("reflection", {}),
            "memory_saved": bool(
                state.get("memory_saved", False)
            ),
        }

    def list_memories(self):
        return self.memory_service.list_all()
