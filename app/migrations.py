from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.models import (
    HypothesisEvidence,
    HypothesisRevision,
    ProposalNormalizationRecord,
)


V0_2_SQLITE_COLUMNS = {
    "experiences": {
        "context_data": "JSON DEFAULT '{}'",
    },
    "reflection_records": {
        "insight": "TEXT DEFAULT ''",
        "applicable_context": "JSON DEFAULT '{}'",
        "expected_effect": "TEXT DEFAULT ''",
        "canonical_signal": "TEXT DEFAULT ''",
    },
    "strategy_hypotheses": {
        "refinement_count": "INTEGER DEFAULT 0",
        "independent_support_count": "INTEGER DEFAULT 0",
    },
}


V0_2_1_SQLITE_COLUMNS = {
    "strategy_hypotheses": {
        "current_revision": "INTEGER DEFAULT 1",
    },
    # Phase 5.1: existing Phase 5 candidates used reflection-level fit.
    # Mark them as evaluator v1 so they can be safely re-evaluated in place.
    "candidate_revisions": {
        "evaluation_version": "INTEGER DEFAULT 1",
        "complexity_version": "INTEGER DEFAULT 1",
    },
    "revision_proposals": {
        "parent_proposal_id": "INTEGER",
        "proposal_kind": "TEXT DEFAULT 'raw'",
        "normalization_version": "INTEGER DEFAULT 0",
    },
    "hypothesis_revisions": {
        "parent_revision": "INTEGER",
    },
    "revision_signals": {
        "provenance_data": "JSON DEFAULT '{}'",
        "resolution_status": "TEXT DEFAULT 'open'",
        "resolved_by_candidate_id": "INTEGER",
        "resolved_by_revision": "INTEGER",
        "resolved_at": "DATETIME",
        "superseded_by_signal_id": "INTEGER",
        "superseded_by_revision": "INTEGER",
        "superseded_at": "DATETIME",
        "resolution_note": "TEXT DEFAULT ''",
    },
}


def migrate_v0_2(engine: Engine) -> None:
    """对 V0.1.1 SQLite 数据库做非破坏性 V0.2 增量迁移。

    新表仍由 Base.metadata.create_all 创建；这里只负责给已有表补列。
    """
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table_name, columns in V0_2_SQLITE_COLUMNS.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            for column_name, definition in columns.items():
                if column_name in existing_columns:
                    continue
                conn.execute(
                    text(
                        f'ALTER TABLE "{table_name}" '
                        f'ADD COLUMN "{column_name}" {definition}'
                    )
                )


def _sqlite_rebuild_hypothesis_evidence(engine: Engine) -> None:
    """把 V0.2 evidence 表升级为 revision-aware schema。

    SQLite 不能直接删除旧 UNIQUE(hypothesis_id, reflection_id)，因此必须重建表。
    迁移保持 id、关系类型、置信度与时间不变，并把所有旧证据绑定到 revision=1。
    """
    inspector = inspect(engine)
    if "hypothesis_evidence" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("hypothesis_evidence")}
    unique_constraints = inspector.get_unique_constraints("hypothesis_evidence")
    has_revision = "hypothesis_revision" in columns
    has_revision_unique = any(
        set(item.get("column_names") or [])
        == {"hypothesis_id", "hypothesis_revision", "reflection_id"}
        for item in unique_constraints
    )
    if has_revision and has_revision_unique:
        return

    backup = "hypothesis_evidence_v0_2_backup"
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text(f'DROP TABLE IF EXISTS "{backup}"'))
        conn.execute(text(f'ALTER TABLE "hypothesis_evidence" RENAME TO "{backup}"'))

        # 旧表被 rename 后，其普通索引名仍占用 namespace。先删除，随后 ORM
        # 为新表重新创建索引。
        backup_inspector = inspect(conn)
        for index in backup_inspector.get_indexes(backup):
            name = index.get("name")
            if name:
                safe_name = str(name).replace('"', '""')
                conn.execute(text(f'DROP INDEX IF EXISTS "{safe_name}"'))

        HypothesisEvidence.__table__.create(bind=conn, checkfirst=False)

        old_columns = {
            column["name"] for column in backup_inspector.get_columns(backup)
        }
        revision_expr = (
            "COALESCE(hypothesis_revision, 1)"
            if "hypothesis_revision" in old_columns
            else "1"
        )
        conn.execute(
            text(
                f'''
                INSERT INTO hypothesis_evidence (
                    id,
                    hypothesis_id,
                    hypothesis_revision,
                    reflection_id,
                    relation_type,
                    relation_confidence,
                    relation_reason,
                    is_independent,
                    created_at
                )
                SELECT
                    id,
                    hypothesis_id,
                    {revision_expr},
                    reflection_id,
                    relation_type,
                    relation_confidence,
                    relation_reason,
                    is_independent,
                    created_at
                FROM "{backup}"
                '''
            )
        )
        conn.execute(text(f'DROP TABLE "{backup}"'))
        conn.execute(text("PRAGMA foreign_keys=ON"))


def _backfill_hypothesis_revisions(engine: Engine) -> None:
    """把 V0.2 StrategyHypothesis 当前内容无损固化为 Revision 快照。"""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "strategy_hypotheses" not in tables:
        return

    # 新表通常已由 Base.metadata.create_all 建立；这里保持 migration 自包含。
    HypothesisRevision.__table__.create(bind=engine, checkfirst=True)

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    id,
                    title,
                    canonical_signal,
                    insight,
                    proposed_change,
                    expected_effect,
                    applicable_context,
                    COALESCE(current_revision, revision, 1) AS current_revision,
                    created_at
                FROM strategy_hypotheses
                """
            )
        ).mappings().all()

        for row in rows:
            hypothesis_id = int(row["id"])
            revision_number = int(row["current_revision"] or 1)
            exists = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM hypothesis_revisions
                    WHERE hypothesis_id = :hypothesis_id
                      AND revision_number = :revision_number
                    LIMIT 1
                    """
                ),
                {
                    "hypothesis_id": hypothesis_id,
                    "revision_number": revision_number,
                },
            ).first()
            if exists is not None:
                continue

            raw_context = row["applicable_context"]
            if isinstance(raw_context, str):
                try:
                    when_data = json.loads(raw_context) if raw_context else {}
                except json.JSONDecodeError:
                    when_data = {"legacy_raw": raw_context}
            elif isinstance(raw_context, dict):
                when_data = raw_context
            else:
                when_data = {}

            conn.execute(
                text(
                    """
                    INSERT INTO hypothesis_revisions (
                        hypothesis_id,
                        revision_number,
                        title,
                        canonical_signal,
                        insight,
                        when_data,
                        do_data,
                        expect_data,
                        created_reason,
                        created_at
                    ) VALUES (
                        :hypothesis_id,
                        :revision_number,
                        :title,
                        :canonical_signal,
                        :insight,
                        :when_data,
                        :do_data,
                        :expect_data,
                        :created_reason,
                        :created_at
                    )
                    """
                ),
                {
                    "hypothesis_id": hypothesis_id,
                    "revision_number": revision_number,
                    "title": row["title"] or "",
                    "canonical_signal": row["canonical_signal"] or "",
                    "insight": row["insight"] or "",
                    "when_data": json.dumps(when_data, ensure_ascii=False),
                    "do_data": json.dumps(
                        {"text": row["proposed_change"] or ""}, ensure_ascii=False
                    ),
                    "expect_data": json.dumps(
                        {"text": row["expected_effect"] or ""}, ensure_ascii=False
                    ),
                    "created_reason": "migrated_from_v0.2_current_state",
                    "created_at": row["created_at"] or datetime.now(),
                },
            )



def _backfill_revision_lineage_and_signal_resolution(engine: Engine) -> None:
    """Phase 5.4: recover lineage/resolution from immutable application events.

    This is deliberately idempotent so an existing Phase 5.3 database (where r2
    already exists) can be upgraded without replaying any LLM work.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    required = {"hypothesis_revisions", "revision_application_events", "revision_signals"}
    if not required.issubset(tables):
        return

    with engine.begin() as conn:
        events = conn.execute(
            text(
                """
                SELECT id, hypothesis_id, from_revision, to_revision, candidate_id, action, event_data, created_at
                FROM revision_application_events
                WHERE action IN ('AUTO_APPLY', 'HUMAN_APPLY')
                  AND to_revision IS NOT NULL
                ORDER BY id ASC
                """
            )
        ).mappings().all()

        for event in events:
            conn.execute(
                text(
                    """
                    UPDATE hypothesis_revisions
                    SET parent_revision = COALESCE(parent_revision, :parent_revision)
                    WHERE hypothesis_id = :hypothesis_id
                      AND revision_number = :revision_number
                    """
                ),
                {
                    "parent_revision": int(event["from_revision"]),
                    "hypothesis_id": int(event["hypothesis_id"]),
                    "revision_number": int(event["to_revision"]),
                },
            )

            raw_data = event["event_data"]
            if isinstance(raw_data, str):
                try:
                    event_data = json.loads(raw_data) if raw_data else {}
                except json.JSONDecodeError:
                    event_data = {}
            elif isinstance(raw_data, dict):
                event_data = raw_data
            else:
                event_data = {}

            for signal_id in event_data.get("resolved_signal_ids", []) or []:
                try:
                    signal_id = int(signal_id)
                except (TypeError, ValueError):
                    continue
                conn.execute(
                    text(
                        """
                        UPDATE revision_signals
                        SET resolution_status = 'resolved',
                            resolved_by_candidate_id = COALESCE(resolved_by_candidate_id, :candidate_id),
                            resolved_by_revision = COALESCE(resolved_by_revision, :revision_number),
                            resolved_at = COALESCE(resolved_at, :resolved_at),
                            resolution_note = CASE
                                WHEN COALESCE(resolution_note, '') = '' THEN :note
                                ELSE resolution_note
                            END
                        WHERE id = :signal_id
                          AND hypothesis_id = :hypothesis_id
                        """
                    ),
                    {
                        "candidate_id": event["candidate_id"],
                        "revision_number": int(event["to_revision"]),
                        "resolved_at": event["created_at"] or datetime.now(),
                        "note": f"backfilled_from_revision_event_RE#{int(event['id'])}",
                        "signal_id": signal_id,
                        "hypothesis_id": int(event["hypothesis_id"]),
                    },
                )


def _sqlite_upgrade_proposal_normalization_versioning(engine: Engine) -> None:
    """Replace the Phase 5.6 v1 source-only UNIQUE with source+version.

    Existing PN rows, including failed v1 attempts, retain their ids and audit
    payloads. The rebuild changes only the idempotency key so v2 can create a
    new record for the same source proposal.
    """
    inspector = inspect(engine)
    table_name = "proposal_normalization_records"
    if table_name not in inspector.get_table_names():
        return

    unique_constraints = inspector.get_unique_constraints(table_name)
    has_versioned_unique = any(
        set(item.get("column_names") or [])
        == {"source_proposal_id", "normalization_version"}
        for item in unique_constraints
    )
    has_source_only_unique = any(
        set(item.get("column_names") or []) == {"source_proposal_id"}
        for item in unique_constraints
    )
    if has_versioned_unique and not has_source_only_unique:
        return

    backup = "proposal_normalization_records_v1_backup"
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text(f'DROP TABLE IF EXISTS "{backup}"'))
        conn.execute(text(f'ALTER TABLE "{table_name}" RENAME TO "{backup}"'))

        backup_inspector = inspect(conn)
        for index in backup_inspector.get_indexes(backup):
            name = index.get("name")
            if name:
                safe_name = str(name).replace('"', '""')
                conn.execute(text(f'DROP INDEX IF EXISTS "{safe_name}"'))

        ProposalNormalizationRecord.__table__.create(bind=conn, checkfirst=False)
        conn.execute(
            text(
                f'''
                INSERT INTO {table_name} (
                    id,
                    source_proposal_id,
                    normalized_proposal_id,
                    normalization_version,
                    status,
                    rule_count,
                    source_signal_ids,
                    reason,
                    normalization_data,
                    created_at,
                    updated_at
                )
                SELECT
                    id,
                    source_proposal_id,
                    normalized_proposal_id,
                    COALESCE(normalization_version, 1),
                    status,
                    rule_count,
                    source_signal_ids,
                    reason,
                    normalization_data,
                    created_at,
                    updated_at
                FROM "{backup}"
                '''
            )
        )
        conn.execute(text(f'DROP TABLE "{backup}"'))
        conn.execute(text("PRAGMA foreign_keys=ON"))


def migrate_v0_2_1(engine: Engine) -> None:
    """V0.2 -> V0.2.1 第一阶段迁移：Hypothesis revision-aware schema。

    目标：
    - StrategyHypothesis 增加 current_revision；
    - 现有 Hypothesis 当前内容固化为不可变 Revision；
    - HypothesisEvidence 绑定具体 revision，并移除旧的跨 revision 唯一约束；
    - 不改变当前 Hypothesis 的业务语义或成熟度。
    """
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table_name, columns in V0_2_1_SQLITE_COLUMNS.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            for column_name, definition in columns.items():
                if column_name in existing_columns:
                    continue
                conn.execute(
                    text(
                        f'ALTER TABLE "{table_name}" '
                        f'ADD COLUMN "{column_name}" {definition}'
                    )
                )

        if "strategy_hypotheses" in existing_tables:
            # V0.2 的 revision 就是当时的 current revision。V0.2 真实数据目前通常为 1，
            # 这里仍保留通用迁移语义。
            conn.execute(
                text(
                    """
                    UPDATE strategy_hypotheses
                    SET current_revision = COALESCE(revision, current_revision, 1)
                    """
                )
            )

    _sqlite_rebuild_hypothesis_evidence(engine)
    _backfill_hypothesis_revisions(engine)
    _backfill_revision_lineage_and_signal_resolution(engine)
    _sqlite_upgrade_proposal_normalization_versioning(engine)
