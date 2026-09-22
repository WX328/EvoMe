from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
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
    context_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class ReflectionRecord(Base):
    """对 Experience 的反思结果。"""
    __tablename__ = "reflection_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    experience_id: Mapped[int] = mapped_column(ForeignKey("experiences.id"), index=True)
    reflection_type: Mapped[str] = mapped_column(String(32), index=True)
    observation: Mapped[str] = mapped_column(Text)
    lesson: Mapped[str] = mapped_column(Text)  # V0.1.1 compatibility
    insight: Mapped[str] = mapped_column(Text, default="")
    applicable_context: Mapped[dict] = mapped_column(JSON, default=dict)
    expected_effect: Mapped[str] = mapped_column(Text, default="")
    canonical_signal: Mapped[str] = mapped_column(Text, default="")
    strategy_signal: Mapped[str] = mapped_column(Text, default="")
    skill_signal: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class StrategyHypothesis(Base):
    """多次 Reflection 聚合形成的、带适用环境的策略假设。"""
    __tablename__ = "strategy_hypotheses"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    canonical_signal: Mapped[str] = mapped_column(Text)
    insight: Mapped[str] = mapped_column(Text)
    proposed_change: Mapped[str] = mapped_column(Text)
    expected_effect: Mapped[str] = mapped_column(Text, default="")
    applicable_context: Mapped[dict] = mapped_column(JSON, default=dict)

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(32), default="observing", index=True)
    # V0.2 compatibility: revision 暂时保留，并与 current_revision 镜像。
    revision: Mapped[int] = mapped_column(Integer, default=1)
    current_revision: Mapped[int] = mapped_column(Integer, default=1, index=True)

    supporting_count: Mapped[int] = mapped_column(Integer, default=0)
    refinement_count: Mapped[int] = mapped_column(Integer, default=0)
    counter_count: Mapped[int] = mapped_column(Integer, default=0)
    independent_evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    independent_support_count: Mapped[int] = mapped_column(Integer, default=0)
    context_diversity: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    last_evidence_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class HypothesisRevision(Base):
    """StrategyHypothesis 的不可变认知快照。

    同一个 hypothesis_id 下通过 revision_number 形成 r1/r2/r3... 历史。
    V0.2.1 第一阶段只建立版本地基，不自动生成新 Revision。
    """
    __tablename__ = "hypothesis_revisions"
    __table_args__ = (
        UniqueConstraint(
            "hypothesis_id",
            "revision_number",
            name="uq_hypothesis_revision_number",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    revision_number: Mapped[int] = mapped_column(Integer, index=True)
    # Parent revision records lineage explicitly so rollback/new branches can
    # reconcile signal resolution against the active ancestry.
    parent_revision: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    title: Mapped[str] = mapped_column(String(200))
    canonical_signal: Mapped[str] = mapped_column(Text)
    insight: Mapped[str] = mapped_column(Text)

    # Revision 层从一开始就保留机器可比较的 WHEN / DO / EXPECT。
    # 对 V0.2 旧数据不做语义猜测：WHEN 直接迁移 applicable_context，
    # DO/EXPECT 先以 {"text": ...} 形式无损保存。
    when_data: Mapped[dict] = mapped_column(JSON, default=dict)
    do_data: Mapped[dict] = mapped_column(JSON, default=dict)
    expect_data: Mapped[dict] = mapped_column(JSON, default=dict)

    created_reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class HypothesisEvidence(Base):
    """Reflection 与 StrategyHypothesis 之间的证据关系。"""
    __tablename__ = "hypothesis_evidence"
    __table_args__ = (
        UniqueConstraint(
            "hypothesis_id",
            "hypothesis_revision",
            "reflection_id",
            name="uq_hypothesis_revision_reflection_evidence",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    hypothesis_revision: Mapped[int] = mapped_column(Integer, default=1, index=True)
    reflection_id: Mapped[int] = mapped_column(
        ForeignKey("reflection_records.id"), index=True
    )
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    relation_confidence: Mapped[float] = mapped_column(Float, default=0.8)
    relation_reason: Mapped[str] = mapped_column(Text, default="")
    is_independent: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class RevisionSignal(Base):
    """REFINES Evidence 对某个 Hypothesis Revision 提出的局部修改信号。

    Phase 2 仅记录信号，不自动聚合、生成 Proposal 或修改 Hypothesis。
    """
    __tablename__ = "revision_signals"
    __table_args__ = (
        UniqueConstraint(
            "hypothesis_id",
            "base_revision",
            "reflection_id",
            "dimension",
            "operation",
            "target_field",
            name="uq_revision_signal_identity",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    base_revision: Mapped[int] = mapped_column(Integer, default=1, index=True)
    reflection_id: Mapped[int] = mapped_column(
        ForeignKey("reflection_records.id"), index=True
    )
    # 逻辑引用 Evidence ID，不使用数据库 FK。原因：V0.2 -> V0.2.1 的 SQLite
    # migration 会重建 hypothesis_evidence 表；避免新表在旧库迁移时绑定到临时 backup 表。
    evidence_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    interaction_id: Mapped[int | None] = mapped_column(
        ForeignKey("interaction_records.id"), nullable=True, index=True
    )

    dimension: Mapped[str] = mapped_column(String(32), index=True)
    operation: Mapped[str] = mapped_column(String(32), index=True)
    target_field: Mapped[str] = mapped_column(String(100), index=True)

    before_data: Mapped[dict | list | str | int | float | bool | None] = mapped_column(
        JSON, nullable=True
    )
    observed_data: Mapped[dict | list | str | int | float | bool | None] = mapped_column(
        JSON, nullable=True
    )
    canonical_change: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    provenance_data: Mapped[dict] = mapped_column(JSON, default=dict)

    # Resolution is historical metadata. Whether a resolved signal still counts
    # as resolved for the current head is determined from revision lineage.
    resolution_status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    resolved_by_candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidate_revisions.id"), nullable=True, index=True
    )
    resolved_by_revision: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Ancestor revalidation may either carry the same semantic signal forward to
    # a descendant revision or retire it for that branch. Supersession is also
    # lineage-aware: rolling back before superseded_by_revision re-opens the
    # original signal instead of erasing history.
    superseded_by_signal_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_signals.id"), nullable=True, index=True
    )
    superseded_by_revision: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_note: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class RevisionSignalRevalidation(Base):
    """Audit record for re-checking one ancestor RevisionSignal against a newer head.

    Revalidation is immutable per evaluator version. A STILL_APPLIES result may
    materialize a new RevisionSignal on target_revision; the source signal stays
    historically intact and is only branch-superseded.
    """
    __tablename__ = "revision_signal_revalidations"
    __table_args__ = (
        UniqueConstraint(
            "source_signal_id",
            "target_revision",
            "evaluation_version",
            name="uq_revision_signal_revalidation_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    source_signal_id: Mapped[int] = mapped_column(
        ForeignKey("revision_signals.id"), index=True
    )
    source_revision: Mapped[int] = mapped_column(Integer, index=True)
    target_revision: Mapped[int] = mapped_column(Integer, index=True)
    evaluation_version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    reason: Mapped[str] = mapped_column(Text, default="")
    rebased_signal_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_signals.id"), nullable=True, index=True
    )
    evaluation_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class RevisionAnalysisRecord(Base):
    """对一个 RevisionSignalGroup 的语义一致性分析记录。

    Analysis 只回答“这些 Signal 是否指向同一个修订方向”，
    不会创建或激活新的 HypothesisRevision。candidate_fingerprint 用于避免
    在 Signal 集合没有变化时重复消耗 LLM Token。
    """
    __tablename__ = "revision_analysis_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    base_revision: Mapped[int] = mapped_column(Integer, default=1, index=True)
    dimension: Mapped[str] = mapped_column(String(32), index=True)
    operation: Mapped[str] = mapped_column(String(32), index=True)
    target_field: Mapped[str] = mapped_column(String(100), index=True)

    signal_ids: Mapped[list] = mapped_column(JSON, default=list)
    candidate_fingerprint: Mapped[str] = mapped_column(
        String(64), unique=True, index=True
    )

    consistent: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    direction_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class RevisionProposal(Base):
    """Revision Analyzer 形成的候选局部修订。

    V0.2.1 Phase 4 只生成 Proposal，不创建 r2、不修改当前 Hypothesis。
    """
    __tablename__ = "revision_proposals"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("revision_analysis_records.id"), unique=True, index=True
    )
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    base_revision: Mapped[int] = mapped_column(Integer, default=1, index=True)

    dimension: Mapped[str] = mapped_column(String(32), index=True)
    operation: Mapped[str] = mapped_column(String(32), index=True)
    target_field: Mapped[str] = mapped_column(String(100), index=True)

    before_data: Mapped[dict | list | str | int | float | bool | None] = mapped_column(
        JSON, nullable=True
    )
    proposed_after_data: Mapped[dict | list | str | int | float | bool | None] = mapped_column(
        JSON, nullable=True
    )
    reason: Mapped[str] = mapped_column(Text, default="")
    source_signal_ids: Mapped[list] = mapped_column(JSON, default=list)
    independent_signal_count: Mapped[int] = mapped_column(Integer, default=0)
    direction_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(32), default="proposed", index=True)

    # Phase 5.6 keeps the analyzer's raw proposal immutable and writes policy
    # normalization as a child proposal.  Existing proposals backfill to the
    # legacy/raw kind instead of being rewritten in place.
    parent_proposal_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_proposals.id"), nullable=True, index=True
    )
    proposal_kind: Mapped[str] = mapped_column(
        String(32), default="raw", index=True
    )
    normalization_version: Mapped[int] = mapped_column(Integer, default=0, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


class ProposalNormalizationRecord(Base):
    """Immutable audit trail for a raw proposal -> conditional-policy proposal.

    A failed normalization is recorded too. Idempotency is scoped by
    (source_proposal_id, normalization_version), allowing a corrected contract
    to retry without overwriting an earlier audit result.
    """
    __tablename__ = "proposal_normalization_records"
    __table_args__ = (
        UniqueConstraint(
            "source_proposal_id",
            "normalization_version",
            name="uq_proposal_normalization_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_proposal_id: Mapped[int] = mapped_column(
        ForeignKey("revision_proposals.id"), index=True
    )
    normalized_proposal_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_proposals.id"), nullable=True, unique=True, index=True
    )
    normalization_version: Mapped[int] = mapped_column(Integer, default=2, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    rule_count: Mapped[int] = mapped_column(Integer, default=0)
    source_signal_ids: Mapped[list] = mapped_column(JSON, default=list)
    reason: Mapped[str] = mapped_column(Text, default="")
    normalization_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


class CandidateRevision(Base):
    """RevisionProposal 应用到正式 base revision 后形成的实验性候选。

    Candidate 不是正式 HypothesisRevision，不会改变 current_revision。Phase 5
    只负责构建、Evidence Re-evaluation、Utility/Risk 分类。
    """
    __tablename__ = "candidate_revisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    base_revision: Mapped[int] = mapped_column(Integer, default=1, index=True)
    source_proposal_id: Mapped[int] = mapped_column(
        ForeignKey("revision_proposals.id"), unique=True, index=True
    )

    title: Mapped[str] = mapped_column(String(200), default="")
    insight: Mapped[str] = mapped_column(Text, default="")
    when_data: Mapped[dict] = mapped_column(JSON, default=dict)
    do_data: Mapped[dict] = mapped_column(JSON, default=dict)
    expect_data: Mapped[dict] = mapped_column(JSON, default=dict)
    canonical_signal: Mapped[str] = mapped_column(Text, default="")

    status: Mapped[str] = mapped_column(String(32), default="validating", index=True)

    base_fit_score: Mapped[float] = mapped_column(Float, default=0.0)
    candidate_fit_score: Mapped[float] = mapped_column(Float, default=0.0)
    fit_gain: Mapped[float] = mapped_column(Float, default=0.0)

    base_complexity_score: Mapped[int] = mapped_column(Integer, default=0)
    complexity_score: Mapped[int] = mapped_column(Integer, default=0)
    complexity_delta: Mapped[int] = mapped_column(Integer, default=0)

    risk_level: Mapped[str] = mapped_column(String(16), default="MEDIUM", index=True)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    utility_score: Mapped[float] = mapped_column(Float, default=0.0)
    evaluation_version: Mapped[int] = mapped_column(Integer, default=3, index=True)
    # Legacy candidates keep v1. Conditional response-policy candidates use the
    # semantic scorer introduced in Phase 5.6 and are never retroactively
    # re-scored.
    complexity_version: Mapped[int] = mapped_column(Integer, default=1, index=True)

    evidence_ids: Mapped[list] = mapped_column(JSON, default=list)
    evaluation_data: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


class CandidateRebaseRecord(Base):
    """Audit record for carrying a stale Candidate onto the current revision.

    A clean rebase is only allowed when every source signal was revalidated as
    STILL_APPLIES and the proposal target field did not change between the old
    base and the current head. The rebased Candidate is always re-evaluated.
    """
    __tablename__ = "candidate_rebase_records"
    __table_args__ = (
        UniqueConstraint(
            "source_candidate_id",
            "target_revision",
            name="uq_candidate_rebase_target",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    source_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_revisions.id"), index=True
    )
    source_proposal_id: Mapped[int] = mapped_column(
        ForeignKey("revision_proposals.id"), index=True
    )
    from_revision: Mapped[int] = mapped_column(Integer, index=True)
    target_revision: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)

    source_signal_ids: Mapped[list] = mapped_column(JSON, default=list)
    target_signal_ids: Mapped[list] = mapped_column(JSON, default=list)
    rebased_analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_analysis_records.id"), nullable=True, index=True
    )
    rebased_proposal_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_proposals.id"), nullable=True, index=True
    )
    rebased_candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidate_revisions.id"), nullable=True, index=True
    )
    reason: Mapped[str] = mapped_column(Text, default="")
    rebase_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


class RevisionApplicationEvent(Base):
    """Hypothesis Revision 的应用/回滚/人工拒绝审计事件。

    Revision 本身保持不可变；这个表记录 current head 为什么、由谁/哪条 Candidate
    发生了变化。AUTO_APPLY 也受 change budget 约束。
    """
    __tablename__ = "revision_application_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("strategy_hypotheses.id"), index=True
    )
    from_revision: Mapped[int] = mapped_column(Integer, index=True)
    to_revision: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidate_revisions.id"), nullable=True, index=True
    )
    proposal_id: Mapped[int | None] = mapped_column(
        ForeignKey("revision_proposals.id"), nullable=True, index=True
    )

    action: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(16), default="", index=True)
    utility_score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    event_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class LearningInvalidationEvent(Base):
    """Operator-confirmed invalidation of learning derived from one interaction.

    The source InteractionRecord is never deleted. This event records which derived
    artifacts were retired so contamination cleanup remains auditable.
    """
    __tablename__ = "learning_invalidation_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    interaction_id: Mapped[int] = mapped_column(
        ForeignKey("interaction_records.id"), index=True
    )
    reason: Mapped[str] = mapped_column(Text, default="")
    event_data: Mapped[dict] = mapped_column(JSON, default=dict)
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
