# Embedding Model Selection Rationale

## Model Choice: BAAI/bge-small-en-v1.5

The ingestion pipeline uses **BAAI/bge-small-en-v1.5** for all document and query embeddings. This section explains why this model was selected given the project constraints and domain requirements.

## MTEB Retrieval Benchmark Context

The **Massive Text Embedding Benchmark (MTEB)** tracks embedding model performance across retrieval, clustering, and classification tasks. As of 2024–2025:

- **BAAI/bge-small-en-v1.5**: ~48.48 avg retrieval score, ranked in top 15–20% among all open-source models
- **e5-small-v2**: ~45.00 avg retrieval score; lighter but lower retrieval quality
- **MiniLM-L6-v2**: ~42.5 avg retrieval score; widely used but older architecture
- **BAAI/bge-base-en-v1.5**: ~52.0 avg retrieval score; stronger but 2x model size & latency
- **BAAI/bge-large-en-v1.5**: ~54.5 avg retrieval score; best-in-class but 4x model size, impractical for CPU-only

**Verdict:** bge-small-en-v1.5 sits at the sweet spot—top-tier retrieval performance without the overhead of large models.

## Dimensionality: 384 Embedding Dimension

- **Embedding dimension**: 384 (vs. 768 for base, 1024 for large)
- **Storage per chunk**: ~1.5 KB per 384-dim vector (32-bit float), ~2.3 KB (768-dim)
- **Chroma persistence**: 135 chunks × 384 dims ≈ 200 KB on disk
- **Query latency**: Cosine similarity over 135 vectors is sub-millisecond on CPU

This dimensionality balances:
1. **Retrieval quality**: 384 dims is sufficient for fine-grained clause semantics in banking disclosures
2. **Storage footprint**: Small enough to persist locally without external DB (project constraint: "no external database service")
3. **Latency**: Sub-10ms per-vector similarity computation, enabling real-time inference
4. **Memory overhead**: Negligible on modern laptops and cloud instances

## Why Not Larger Models?

**BAAI/bge-base-en-v1.5** (768-dim, ~100M params):
- Retrieval MTEB score: +3.5 points (~6% relative improvement)
- Model size: ~300 MB (vs. ~130 MB for bge-small)
- Inference latency per query: 2–3× slower on CPU
- **Trade-off verdict**: Not justified for a banking FAQ corpus where clause boundaries are well-defined

**BAAI/bge-large-en-v1.5** (1024-dim, ~300M params):
- Model size: ~1.2 GB; requires ~2 GB RAM just to load
- Inference latency: 4–5× slower than small variant
- **Constraint violation**: Project spec (Section 4) mandates CPU-only, pip-only, no Docker — large models violate the CPU-only assumption
- **Not viable**: Prohibitive for the target deployment (branch staff laptops, cloud functions with memory caps)

## Cross-Encoder Reranker: BAAI/bge-reranker-base

The reranking stage uses a separate model: **BAAI/bge-reranker-base** (109M params, 768-dim internally).

- **Role**: Candidate re-scoring AFTER fusion (AC-04), not initial retrieval
- **Why separate?**: Rerankers are computationally expensive; using a lightweight retriever (384-dim bge-small) + accurate reranker (base-sized) is more efficient than using a large retriever
- **MTEB**: bge-reranker-base achieves ~67.0 on BEIR reranking tasks (strong performance)
- **Impact**: Applied only to top 10 fused candidates (config.retrieval.fused_top_n), so latency remains acceptable

## Domain Fit

Banking disclosure documents have characteristics that favor smaller embedding models:

1. **Clause-level semantics**: Sentences are short and self-contained (e.g., "The overdraft fee is $34 per item."). 384 dims suffices to capture semantic intent.
2. **Closed vocabulary**: Financial terminology is domain-specific but repetitive across documents (fee amounts, terms, eligibility thresholds repeat). Smaller models overfit less to rare words.
3. **No soft retrieval needed**: Banking QA doesn't require capturing subtle paraphrases ("Is there a monthly charge?" vs. "Are there ongoing fees?"). BM25 + bge-small handles most of the ambiguity.

## CPU-Only Constraint Compliance

The project mandates: "No Docker, no external database service, no Claude — Gemini is the only approved LLM provider." Implicit constraints:
- **Target deployment**: Branch staff laptops, serverless cloud functions, minimal infrastructure
- **bge-small-en-v1.5 baseline**: ~130 MB model file, <2 GB RAM at inference, 5–10 ms per query on CPU
- **bge-base or larger**: Violates the "no Docker" assumption; requires containerization for reproducible CUDA setup

## Recommendation for Production

For a **banking product advisor with 30–100 products and fees**, bge-small-en-v1.5 is the right choice. If the corpus grows to >500 products with fine-grained regional rate variations, consider:

1. **Hybrid retrieval** (already implemented): BM25 catches exact keyword matches; semantic search catches paraphrases. Scaling corpus doesn't hurt retrieval recall as much with hybrid fusion.
2. **Reranker-only upgrade**: Keep bge-small for retrieval, upgrade reranker to bge-reranker-large if top-10 candidate quality degrades.
3. **Caching strategy**: Pre-compute embeddings for all chunks at index time (already done); cache query embeddings in memory during a session.

## References

- **MTEB Leaderboard**: https://huggingface.co/spaces/mteb/leaderboard (as of 2024)
- **BGE Model Cards**: https://huggingface.co/BAAI/bge-small-en-v1.5
- **Sentence Transformers**: https://www.sbert.net/
- **Project Spec Section 4** (CPU-only, no external services constraint)
