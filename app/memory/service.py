from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Memory


def _normalize(text: str) -> str:
    return "".join(text.lower().split())


def _bigrams(text: str) -> set[str]:
    text = _normalize(text)
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i:i + 2] for i in range(len(text) - 1)}


def text_similarity(a: str, b: str) -> float:
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class MemoryService:
    def __init__(self, db: Session):
        self.db = db

    def save(
        self,
        *,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "runtime",
        memory_data: dict | None = None,
    ) -> Memory:
        memory = Memory(
            memory_type=memory_type,
            content=content.strip(),
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            memory_data=memory_data or {},
        )
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        return memory

    def list_all(self) -> list[Memory]:
        stmt = select(Memory).order_by(Memory.created_at.desc())
        return list(self.db.scalars(stmt).all())

    def is_duplicate(self, content: str, threshold: float = 0.72) -> bool:
        return any(
            text_similarity(content, m.content) >= threshold
            for m in self.list_all()
        )

    def search(self, query: str, top_k: int = 5) -> list[Memory]:
        scored: list[tuple[float, Memory]] = []

        for memory in self.list_all():
            similarity = text_similarity(query, memory.content)
            if similarity <= 0:
                continue

            score = (
                similarity * 0.70
                + memory.importance * 0.20
                + memory.confidence * 0.10
            )
            scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [m for _, m in scored[:top_k]]

        now = datetime.now()
        for memory in selected:
            memory.access_count += 1
            memory.last_accessed_at = now

        self.db.commit()
        return selected
