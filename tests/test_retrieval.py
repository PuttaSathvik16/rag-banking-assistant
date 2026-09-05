"""
Requires the index to be built first: `python cli.py setup`.
Covers AC-03 (hybrid fusion) and AC-04 (reranking).
"""
import pytest
from src.retrieval import HybridRetriever
from src.reranker import Reranker
from src.config import load_config


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(load_config())


@pytest.fixture(scope="module")
def reranker():
    return Reranker(load_config())


def test_ac03_hybrid_fusion_returns_candidates(retriever):
    """AC-03: retrieval combines BM25 + semantic and fuses via RRF."""
    results = retriever.retrieve("What is the minimum balance for Premium Savings?")
    assert len(results) > 0
    top = results[0]
    assert top["bm25_score"] is not None or top["vector_score"] is not None
    assert "rrf_score" in top


def test_ac03_fusion_favors_docs_ranked_in_either_list(retriever):
    results = retriever.retrieve("premium savings minimum balance")
    source_files = {r["source_file"] for r in results}
    assert "premium-savings-disclosure.md" in source_files


def test_ac04_reranker_reduces_to_configured_top_k(retriever, reranker):
    """AC-04: reranked output is capped at config.retrieval.final_top_k."""
    config = load_config()
    candidates = retriever.retrieve("What is the overdraft fee?")
    reranked = reranker.rerank("What is the overdraft fee?", candidates)
    assert len(reranked) <= config["retrieval"]["final_top_k"]
    if len(reranked) > 1:
        assert reranked[0]["rerank_score"] >= reranked[-1]["rerank_score"]
