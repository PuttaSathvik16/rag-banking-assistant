"""Reranking stage: cross-encoder re-scores the fused candidates before the
final top-K is passed to generation (AC-04)."""
from sentence_transformers import CrossEncoder
from src.config import load_config


class Reranker:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.model = CrossEncoder(self.config["reranker"]["model_name"])

    def rerank(self, query: str, candidates: list):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        final_top_k = self.config["retrieval"]["final_top_k"]
        return ranked[:final_top_k]
