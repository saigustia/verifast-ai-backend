"""
Embedding generation + vector store interface.
The in-memory store here is a placeholder — swap it for Pinecone/Weaviate
in production. The interface is kept minimal so that swap doesn't require
touching retrieval_service.py.
"""
from typing import List, Tuple

import numpy as np

from models.schemas import DocumentChunk


class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._store: List[Tuple[DocumentChunk, np.ndarray]] = []  # replace with real vector DB

    def embed_text(self, text: str) -> np.ndarray:
        """
        TODO: call a real embedding API (OpenAI / Voyage / Cohere).
        Skeleton only — raises until wired up, so failures are loud
        instead of silently returning garbage vectors.
        """
        raise NotImplementedError("Wire up an embedding API call here.")

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            vector = self.embed_text(chunk.text)
            self._store.append((chunk, vector))

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        query_vec = self.embed_text(query)
        scored = [
            (chunk, self._cosine_similarity(query_vec, vec)) for chunk, vec in self._store
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
