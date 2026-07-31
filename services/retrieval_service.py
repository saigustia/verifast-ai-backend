"""
Hybrid retrieval: combines BM25 (keyword) with semantic (embedding) search.
Also enforces the "not found" threshold for legal/compliance use cases —
if the best combined score falls below the cutoff, retrieval reports
found=False instead of forcing an answer from weak context.
"""
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from config import RETRIEVAL_SCORE_THRESHOLD
from models.schemas import DocumentChunk
from services.embedding_service import EmbeddingService


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService, chunks: List[DocumentChunk]):
        self.embedding_service = embedding_service
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> Tuple[List[Tuple[DocumentChunk, float]], bool]:
        """
        alpha: weight for semantic score vs BM25 score (0.5 = equal weight).
        Returns (results, found) — found=False if the best score is below
        RETRIEVAL_SCORE_THRESHOLD (calibrate this against your own labeled
        validation set — see config.py).
        """
        semantic_results = self.embedding_service.semantic_search(query, top_k=top_k * 2)

        bm25_scores = {}
        if self.bm25:
            tokenized_query = query.lower().split()
            raw_scores = self.bm25.get_scores(tokenized_query)
            max_score = max(raw_scores) if len(raw_scores) else 1.0
            for chunk, score in zip(self.chunks, raw_scores):
                bm25_scores[chunk.chunk_id] = score / (max_score + 1e-8)

        combined = []
        for chunk, sem_score in semantic_results:
            bm25_score = bm25_scores.get(chunk.chunk_id, 0.0)
            final_score = alpha * sem_score + (1 - alpha) * bm25_score
            combined.append((chunk, final_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_results = combined[:top_k]

        found = bool(top_results) and top_results[0][1] >= RETRIEVAL_SCORE_THRESHOLD
        return top_results, found
