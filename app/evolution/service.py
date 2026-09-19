from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Experience, ReflectionRecord


class EvolutionMemoryService:
    def __init__(self, db: Session):
        self.db = db

    def save_experience(
        self,
        *,
        interaction_id: int,
        task_type: str,
        strategy_name: str,
        strategy_version: int,
        skills_used: list[str],
        result_summary: str,
        success: bool,
        score: float,
        experience_metadata: dict | None = None,
    ) -> Experience:
        experience = Experience(
            interaction_id=interaction_id,
            task_type=task_type,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            skills_used=skills_used,
            result_summary=result_summary,
            success=success,
            score=max(0.0, min(1.0, score)),
            experience_metadata=experience_metadata or {},
        )
        self.db.add(experience)
        self.db.commit()
        self.db.refresh(experience)
        return experience

    def save_reflection(
        self,
        *,
        experience_id: int,
        reflection_type: str,
        observation: str,
        lesson: str,
        strategy_signal: str = "",
        skill_signal: str = "",
        confidence: float = 0.8,
    ) -> ReflectionRecord:
        reflection = ReflectionRecord(
            experience_id=experience_id,
            reflection_type=reflection_type,
            observation=observation,
            lesson=lesson,
            strategy_signal=strategy_signal,
            skill_signal=skill_signal,
            confidence=max(0.0, min(1.0, confidence)),
            status="active",
        )
        self.db.add(reflection)
        self.db.commit()
        self.db.refresh(reflection)
        return reflection

    def list_experiences(self) -> list[Experience]:
        stmt = select(Experience).order_by(Experience.created_at.desc())
        return list(self.db.scalars(stmt).all())

    def list_reflections(self) -> list[ReflectionRecord]:
        stmt = select(ReflectionRecord).order_by(ReflectionRecord.created_at.desc())
        return list(self.db.scalars(stmt).all())
