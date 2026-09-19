from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


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
