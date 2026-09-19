from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InteractionRecord(Base):
    """原始交互审计日志；默认不参与常规 Memory Retrieval。"""
    __tablename__ = "interaction_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_input: Mapped[str] = mapped_column(Text)
    assistant_response: Mapped[str] = mapped_column(Text)
    strategy_name: Mapped[str] = mapped_column(String(100), index=True)
    record_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class PersonalMemory(Base):
    """关于用户及其世界的长期认识。"""
    __tablename__ = "personal_memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_type: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    source_interaction_id: Mapped[int | None] = mapped_column(
        ForeignKey("interaction_records.id"), nullable=True, index=True
    )
    memory_data: Mapped[dict] = mapped_column(JSON, default=dict)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Experience(Base):
    """EvoMe 自己的任务经验：这次怎么做、结果怎么样。"""
    __tablename__ = "experiences"

    id: Mapped[int] = mapped_column(primary_key=True)
    interaction_id: Mapped[int] = mapped_column(
        ForeignKey("interaction_records.id"), index=True
    )
    task_type: Mapped[str] = mapped_column(String(100), default="general", index=True)
    strategy_name: Mapped[str] = mapped_column(String(100), index=True)
    strategy_version: Mapped[int] = mapped_column(Integer, default=1)
    skills_used: Mapped[list] = mapped_column(JSON, default=list)
    result_summary: Mapped[str] = mapped_column(Text)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    user_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class ReflectionRecord(Base):
    """对 Experience 的反思结果。"""
    __tablename__ = "reflection_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    experience_id: Mapped[int] = mapped_column(ForeignKey("experiences.id"), index=True)
    reflection_type: Mapped[str] = mapped_column(String(32), index=True)
    observation: Mapped[str] = mapped_column(Text)
    lesson: Mapped[str] = mapped_column(Text)
    strategy_signal: Mapped[str] = mapped_column(Text, default="")
    skill_signal: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class StrategyRecord(Base):
    __tablename__ = "strategy_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active")
    description: Mapped[str] = mapped_column(Text, default="")
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    total_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class SkillUsage(Base):
    __tablename__ = "skill_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_name: Mapped[str] = mapped_column(String(100), index=True)
    implementation_name: Mapped[str] = mapped_column(String(100))
    success: Mapped[int] = mapped_column(Integer, default=1)
    score: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
