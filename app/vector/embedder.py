from __future__ import annotations

import hashlib
import math


class HashingTextEmbedder:
    """V0.2 的轻量本地向量器。

    它只负责 Qdrant 的粗召回，不负责最终语义判断。
    最终关系由 LLM Relation Judge 判断。
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = "".join(text.lower().split())
        if not normalized:
            return []
        tokens = list(normalized)
        tokens.extend(normalized[i : i + 2] for i in range(max(0, len(normalized) - 1)))
        return tokens

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big", signed=False)
            index = raw % self.dimension
            sign = 1.0 if ((raw >> 8) & 1) == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
