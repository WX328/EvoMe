from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.models import HypothesisEvidence, HypothesisRevision


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
