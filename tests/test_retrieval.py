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


def test_rrf_fusion_formula_with_mock_results():
    """Unit test: RRF formula correctly computes scores from hardcoded rank lists (no live retrieval)."""
    from src.retrieval import HybridRetriever
    config = load_config()
    retriever = HybridRetriever(config)

    bm25_results = [
        ("doc_a", 10.5),
        ("doc_b", 8.2),
        ("doc_c", 5.1),
    ]
    vector_results = [
        ("doc_b", 0.95),
        ("doc_c", 0.87),
        ("doc_d", 0.72),
    ]

    fused = retriever._rrf_fuse(bm25_results, vector_results, k=60)
    fused_dict = dict(fused)

    assert "doc_b" in fused_dict, "doc_b appears in both lists, should be in fused results"
    assert "doc_c" in fused_dict, "doc_c appears in both lists, should be in fused results"
    assert "doc_a" in fused_dict, "doc_a in BM25 list, should appear in fusion"
    assert "doc_d" in fused_dict, "doc_d in vector list, should appear in fusion"

    assert fused_dict["doc_b"] > fused_dict["doc_a"], "doc_b (rank 1 in BM25, rank 0 in vector) should outscore doc_a (only in BM25)"
    assert fused_dict["doc_c"] > fused_dict["doc_d"], "doc_c (rank 1 in both) should outscore doc_d (only in vector)"

    expected_b = 1.0 / (60 + 1) + 1.0 / (60 + 0)
    assert abs(fused_dict["doc_b"] - expected_b) < 0.001, f"RRF formula for doc_b incorrect: {fused_dict['doc_b']} vs expected {expected_b}"
