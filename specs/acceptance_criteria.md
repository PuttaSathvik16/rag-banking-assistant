# Acceptance Criteria & Traceability

Per the AC-Traceability Rule, each AC below is referenced by at least one golden-set entry
(`eval/golden_set.json`, via its `ac_refs` field) and/or a test file.

| AC-NN | Criterion | Evidence |
|---|---|---|
| AC-01 | Ingests ≥30 synthetic docs into a persisted, re-runnable, idempotent vector index | `scripts/generate_corpus.py` (33 docs), `src/ingestion.py` (deterministic `upsert` by chunk id) |
| AC-02 | Answers grounded only in corpus with ≥1 clause-level citation | Golden set Q01-Q13, Q15-Q18, Q22-Q23 (`ac_refs: ["AC-02"]`), `src/generation.py` |
| AC-03 | Hybrid retrieval (BM25 + semantic) fused via RRF | `src/retrieval.py::HybridRetriever._rrf_fuse`, `tests/test_retrieval.py` |
| AC-04 | Reranking before top-K passed to generator | `src/reranker.py::Reranker.rerank`, `tests/test_retrieval.py` |
| AC-05 | Abstention when corpus doesn't support an answer | Golden set Q19, Q20, Q21 (`ac_refs: ["AC-05"]`), `eval/run_eval.py::classify_failure` |
| AC-06 | Validated structured output (Pydantic) with answer/citations/product/fee/eligibility/confidence | `src/schemas.py::AnswerResponse`, `tests/test_generation.py` |
| AC-07 | Multi-part/ambiguous queries transformed before retrieval | Golden set Q14, Q17 (`ac_refs: ["AC-07"]`), `src/query_transform.py` |
| AC-08 | Golden eval set (≥20 Q) with a re-runnable scoring script | `eval/golden_set.json` (23 questions), `eval/run_eval.py` |
| AC-09 | RAGAS metrics computed and committed as a report artifact | `eval/run_eval.py` → `eval/reports/ragas_metrics.json` (generated on `python cli.py eval`) |
| AC-10 | ≥2 candidate LLMs compared with a selection rationale | `eval/model_comparison.py` → `eval/reports/model_comparison.md` |

## Non-Functional Requirements

| NFR-NN | Requirement | Evidence |
|---|---|---|
| NFR-01 | No secrets committed | `.env.example` (template only), `.gitignore` excludes `.env` |
| NFR-02 | Pipeline + eval run end-to-end from a single documented command | `cli.py` (`setup` / `ask` / `eval` / `compare-models`), README quick-start |
| NFR-03 | All data synthetic | `scripts/generate_corpus.py` header comment; no real PII anywhere |
| NFR-04 | Retrieval parameters externalized in config | `config.yaml` |
| NFR-05 | Retries + graceful failure, no crashes | `tenacity` retry decorators in `src/generation.py`, `src/query_transform.py`; safe abstention fallback on parse failure |
| NFR-06 | Reproducible quality claims | `eval/run_eval.py`, `eval/model_comparison.py` both re-runnable, write to `eval/reports/` |
| NFR-07 | Cost/latency noted at concept level | `eval/model_comparison.py` records `avg_seconds_per_query`; see `eval/reports/model_comparison.md` after running |
| NFR-08 | Basic logging; no answer without provenance above abstention threshold | `src/generation.py::Generator.generate` enforces threshold at code level, not just model self-report |

**Note:** `tests/test_retrieval.py` and `tests/test_generation.py` referenced above are stubs to
fill in — see the Day 2 checklist in the project plan. Until committed, AC-03/04/06 traceability
rests on the golden-set + eval-report evidence alone; adding the unit tests strengthens this.
