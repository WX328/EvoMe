from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import InteractionRecord


class AuditService:
    def __init__(self, db: Session):
        self.db = db

    def save_interaction(
        self,
        *,
        session_id: str,
        user_input: str,
        assistant_response: str,
        strategy_name: str,
        record_metadata: dict | None = None,
    ) -> InteractionRecord:
        record = InteractionRecord(
            session_id=session_id,
            user_input=user_input,
            assistant_response=assistant_response,
            strategy_name=strategy_name,
            record_metadata=record_metadata or {},
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get(self, interaction_id: int) -> InteractionRecord | None:
        return self.db.get(InteractionRecord, interaction_id)

    def list_recent(self, limit: int = 20) -> list[InteractionRecord]:
        stmt = (
            select(InteractionRecord)
            .order_by(InteractionRecord.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())
