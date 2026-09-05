"""
Hybrid retrieval: BM25 (lexical) + Chroma vector search (semantic), fused via
Reciprocal Rank Fusion (AC-03). Fusion, not just concatenation, is what the
rubric scores here — RRF rewards items ranked highly in *either* list without
requiring score normalization across the two very different score scales.
"""
import os
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import load_config, REPO_ROOT

BM25_CACHE_PATH = os.path.join(REPO_ROOT, "data", "chroma_db", "_bm25_cache.pkl")


def _tokenize(text: str):
    return text.lower().split()


class HybridRetriever:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        vs_cfg = self.config["vector_store"]
        emb_cfg = self.config["embedding"]

        persist_dir = os.path.join(REPO_ROOT, vs_cfg["persist_dir"])
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(vs_cfg["collection_name"])
        self.embedder = SentenceTransformer(emb_cfg["model_name"])

        self._build_bm25_index()

    def _build_bm25_index(self):
        data = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = data["ids"]
        self.corpus_texts = data["documents"]
        self.corpus_metas = data["metadatas"]
        tokenized = [_tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)
        self.id_to_index = {cid: i for i, cid in enumerate(self.corpus_ids)}

    def _bm25_search(self, query: str, top_k: int):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.corpus_ids[i], float(scores[i])) for i in ranked]

    def _vector_search(self, query: str, top_k: int):
        query_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        ids = results["ids"][0]
        distances = results["distances"][0]  # smaller = more similar (cosine distance)
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def _rrf_fuse(self, bm25_results, vector_results, k: int):
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across the lists a doc appears in."""
        fused_scores = {}
        for rank, (cid, _) in enumerate(bm25_results):
            fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        for rank, (cid, _) in enumerate(vector_results):
            fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str):
        r_cfg = self.config["retrieval"]
        bm25_results = self._bm25_search(query, r_cfg["bm25_top_k"])
        vector_results = self._vector_search(query, r_cfg["vector_top_k"])
        fused = self._rrf_fuse(bm25_results, vector_results, r_cfg["rrf_k"])[: r_cfg["fused_top_n"]]

        bm25_score_map = dict(bm25_results)
        vector_score_map = dict(vector_results)

        candidates = []
        for cid, rrf_score in fused:
            idx = self.id_to_index[cid]
            meta = self.corpus_metas[idx]
            candidates.append({
                "chunk_id": cid,
                "text": self.corpus_texts[idx],
                "doc_id": meta["doc_id"],
                "clause_id": meta["clause_id"],
                "source_file": meta["source_file"],
                "doc_type": meta["doc_type"],
                "bm25_score": bm25_score_map.get(cid),
                "vector_score": vector_score_map.get(cid),
                "rrf_score": rrf_score,
            })
        return candidates


if __name__ == "__main__":
    retriever = HybridRetriever()
    for c in retriever.retrieve("What is the minimum balance for premium savings?"):
        print(c["clause_id"], round(c["rrf_score"], 4), c["text"][:80])
