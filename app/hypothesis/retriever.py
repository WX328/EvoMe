from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from app.memory.service import text_similarity
from app.models import StrategyHypothesis
from app.vector.embedder import HashingTextEmbedder


@dataclass(frozen=True)
class HypothesisMatch:
    hypothesis_id: int
    score: float


class HypothesisRetriever(Protocol):
    def index(self, hypothesis: StrategyHypothesis) -> None: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.70,
    ) -> list[HypothesisMatch]: ...


class SqliteHypothesisRetriever:
    """Qdrant 不可用/测试环境下的安全回退。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def index(self, hypothesis: StrategyHypothesis) -> None:
        return None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.70,
    ) -> list[HypothesisMatch]:
        hypotheses = self.db.query(StrategyHypothesis).all()
        scored = [
            HypothesisMatch(hypothesis.id, text_similarity(query, hypothesis.canonical_signal))
            for hypothesis in hypotheses
        ]
        scored = [item for item in scored if item.score >= score_threshold]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class QdrantHypothesisRetriever:
    COLLECTION = "evome_strategy_hypotheses"

    def __init__(self, *, path: str, dimension: int = 256) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.embedder = HashingTextEmbedder(dimension=dimension)
        self.client = QdrantClient(path=path)
        collections = {item.name for item in self.client.get_collections().collections}
        if self.COLLECTION not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def index(self, hypothesis: StrategyHypothesis) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.COLLECTION,
            points=[
                PointStruct(
                    id=hypothesis.id,
                    vector=self.embedder.encode(hypothesis.canonical_signal),
                    payload={
                        "hypothesis_id": hypothesis.id,
                        "status": hypothesis.status,
                        "revision": hypothesis.current_revision,
                    },
                )
            ],
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.70,
    ) -> list[HypothesisMatch]:
        vector = self.embedder.encode(query)
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.COLLECTION,
                query=vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            points = result.points
        else:
            points = self.client.search(
                collection_name=self.COLLECTION,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
        return [HypothesisMatch(int(point.id), float(point.score)) for point in points]


class HybridHypothesisRetriever:
    """优先 Qdrant；不可用时回退 SQLite 粗筛。

    对 SQLite 内存数据库（pytest）强制使用回退，避免测试间共享 Qdrant 索引。
    """

    def __init__(self, db: Session, *, qdrant_path: str) -> None:
        self.db = db
        self.fallback = SqliteHypothesisRetriever(db)
        self.primary = None

        bind = db.get_bind()
        database_name = getattr(bind.url, "database", None)
        is_memory_db = database_name in (None, "", ":memory:")
        if is_memory_db:
            return

        try:
            self.primary = QdrantHypothesisRetriever(path=qdrant_path)
        except Exception as exc:
            print(f"⚠️ Qdrant 不可用，Hypothesis Retrieval 回退 SQLite：{exc}")

    def for_session(self, db: Session) -> "HybridHypothesisRetriever":
        """Reuse the same Qdrant client with a fresh SQLAlchemy session.

        The slow loop runs in a background thread and must never reuse the
        foreground SQLAlchemy Session. Reusing the already-open local Qdrant
        client also avoids opening the same embedded storage path twice.
        """
        clone = object.__new__(HybridHypothesisRetriever)
        clone.db = db
        clone.fallback = SqliteHypothesisRetriever(db)
        clone.primary = self.primary
        return clone

    def index(self, hypothesis: StrategyHypothesis) -> None:
        if self.primary is not None:
            try:
                self.primary.index(hypothesis)
                return
            except Exception as exc:
                print(f"⚠️ Qdrant 索引失败，继续使用 SQLite：{exc}")
        self.fallback.index(hypothesis)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.70,
    ) -> list[HypothesisMatch]:
        if self.primary is not None:
            try:
                return self.primary.search(
                    query, top_k=top_k, score_threshold=score_threshold
                )
            except Exception as exc:
                print(f"⚠️ Qdrant 检索失败，回退 SQLite：{exc}")
        return self.fallback.search(
            query, top_k=top_k, score_threshold=score_threshold
        )
