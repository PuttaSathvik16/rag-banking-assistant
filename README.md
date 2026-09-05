# RAG Banking Assistant

A production-ready retrieval-augmented generation (RAG) system that answers retail banking product, fee, and eligibility questions with clause-level citations and confidence-based abstention. Built on LangChain with hybrid retrieval, intelligent reranking, and grounded generation using Google Gemini.

**Documentation**: [Business Case](docs/business-case.md) · [Acceptance Criteria](specs/acceptance_criteria.md) · [Sample Outputs](docs/sample-outputs.md) · [Embedding Rationale](docs/embedding-selection-rationale.md)

---

## 🌟 Key Features

- **Hybrid Retrieval**: BM25 (lexical) + vector search (Chroma) fused via Reciprocal Rank Fusion (RRF)
- **Intelligent Reranking**: Cross-encoder reranking to surface the most relevant documents
- **Query Transformation**: Decompose multi-part questions using LLM rewriting
- **Grounded Answers**: Citations at the clause level with confidence-based abstention
- **Structured Output**: Pydantic-validated JSON with metadata and provenance
- **Comprehensive Evaluation**: RAGAS metrics + failure taxonomy classification
- **Interactive UI**: Streamlit web interface for easy testing and demos
- **Zero External Dependencies**: Runs entirely on local infrastructure with pip-installed packages
- **Production Configuration**: All parameters managed via `config.yaml` (no hardcoding)

---

## 🏗️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **LLM Framework** | LangChain |
| **Generation** | Google Gemini (2.5-Flash) |
| **Vector Store** | Chroma (in-memory) |
| **Lexical Retrieval** | BM25 (`rank_bm25`) |
| **Embeddings** | Sentence-Transformers |
| **Reranking** | Sentence-Transformers CrossEncoder |
| **Web UI** | Streamlit |
| **Evaluation** | RAGAS |
| **Validation** | Pydantic |
| **Testing** | pytest |

**Note**: No Docker, no external database services, no Claude API — Google Gemini is the sole LLM provider.

---

## 📁 Project Structure

```
rag-banking-assistant/
├── src/                           # Core application modules
│   ├── config.py                 # Configuration loader
│   ├── ingestion.py              # Document parsing & chunking
│   ├── retrieval.py              # BM25 + Chroma hybrid search + RRF fusion
│   ├── reranker.py               # Cross-encoder reranking
│   ├── query_transform.py        # Query decomposition & rewriting
│   ├── generation.py             # LLM-based answer generation & grounding
│   ├── schemas.py                # Pydantic models for structured output
│   └── pipeline.py               # End-to-end orchestration
├── data/
│   ├── corpus/                   # 40+ banking documents (disclosures, FAQs, policies)
│   │   ├── _manifest.json       # Document index & metadata
│   │   ├── auto-loan-*.md       # Auto loan product details
│   │   ├── customer-faq-*.md    # Curated FAQ documents
│   │   ├── fee-schedule-*.md    # Fee schedules by product
│   │   └── ...                  # 35+ additional product docs
│   └── chroma_db/                # Vector store (generated at runtime)
├── eval/                          # Evaluation framework
│   ├── golden_set.json           # 23-question benchmark set
│   ├── run_eval.py               # RAGAS evaluation harness
│   ├── model_comparison.py       # Multi-model comparison script
│   └── reports/                  # Generated evaluation artifacts
│       ├── ragas_metrics.json
│       ├── model_comparison.json
│       ├── model_comparison.md
│       └── failure_taxonomy.json
├── tests/                         # Unit & integration tests
│   ├── test_retrieval.py         # Retrieval pipeline tests
│   └── test_generation.py        # Generation & grounding tests
├── scripts/                       # Utility scripts
│   └── generate_corpus.py        # Synthetic banking corpus generator
├── docs/                          # Documentation
│   ├── business-case.md          # Problem statement & context
│   ├── embedding-selection-rationale.md
│   └── sample-outputs.md         # Example answers with citations
├── specs/                         # Acceptance criteria
│   └── acceptance_criteria.md
├── config.yaml                    # Central configuration (retrieval, generation, eval params)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── cli.py                         # Command-line interface
├── streamlit_app.py               # Interactive web UI
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/PuttaSathvik16/rag-banking-assistant.git
cd rag-banking-assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Google API key
export GOOGLE_API_KEY="your_key_here"
```

Get a free API key: https://aistudio.google.com

### 3. Initialize Corpus & Index

```bash
python cli.py setup
# Generates 40 synthetic banking documents and builds the vector index
```

### 4. Try It Out

**Option A: CLI**
```bash
python cli.py ask "What is the minimum balance for Premium Savings?"
```

**Option B: Interactive Web UI**
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**Option C: Run Evaluation**
```bash
python cli.py eval                # RAGAS metrics
python cli.py compare-models      # Model comparison
```

---

## 🎨 Web UI Features

The Streamlit app (`streamlit_app.py`) provides:

✨ **Interactive Interface**
- Question input with example suggestions
- Real-time processing with visual feedback

📊 **Answer Display**
- Full answer text with formatting
- Confidence score with color coding
- Clause-level citations with sources
- Abstention indicator when uncertain

📈 **Metadata & Insights**
- Retrieved documents counter
- Document relevance scores
- Raw JSON response viewer
- Query processing details

## ⚙️ Configuration

All tuning parameters live in **`config.yaml`**. No hardcoded values in source code.

### Key Parameters

```yaml
# Retrieval
retrieval:
  bm25_k: 5              # Lexical results to retrieve
  chroma_k: 5            # Vector results to retrieve
  rerank_k: 3            # Final results after reranking

# Chunking
ingestion:
  chunk_size: 512        # Document chunk size (tokens)
  overlap: 100           # Overlap between chunks

# Generation
generation:
  model_id: gemini-2.5-flash
  temperature: 0.2       # Low temp for factual grounding
  confidence_threshold: 0.7  # Min confidence to cite
  max_retries: 3
```

**Important**: Model IDs default to `gemini-2.5-flash`. Verify current model availability in [Google AI Studio](https://aistudio.google.com) before running.

---

## 🏛️ Architecture

```
User Question
    ↓
[Query Transformation]     (LLM-based decomposition/rewrite)
    ↓
[Hybrid Retrieval]         (BM25 + Chroma vector search)
    ├─ Lexical search (BM25)
    ├─ Semantic search (Chroma)
    └─ Fuse via RRF (Reciprocal Rank Fusion)
    ↓
[Cross-Encoder Reranking]  (Re-rank by relevance)
    ↓
[Grounded Generation]      (Gemini with citations)
    ├─ Extract relevant clauses
    ├─ Validate confidence
    └─ Format with provenance
    ↓
Structured JSON Response
  {
    "answer": "...",
    "citations": [{"text": "...", "source": "..."}],
    "confidence": 0.95,
    "abstained": false
  }
```

### Core Modules

| Module | Purpose | Key Class |
|--------|---------|-----------|
| `src/ingestion.py` | Parse corpus, chunk documents | `CorpusIngestion` |
| `src/retrieval.py` | BM25 + Chroma + RRF fusion | `HybridRetrieval` |
| `src/reranker.py` | Cross-encoder reranking | `Reranker` |
| `src/query_transform.py` | Query decomposition/rewrite | `QueryTransformer` |
| `src/generation.py` | Grounded answer generation | `Generator` |
| `src/schemas.py` | Output validation | `AnswerResponse`, `Citation` |
| `src/pipeline.py` | End-to-end orchestration | `RAGPipeline` |

---

## 📊 Evaluation

### Golden Set

**`eval/golden_set.json`** — 23 hand-crafted questions covering:
- Direct product lookups (e.g., fee schedules)
- Multi-part queries (decomposition required)
- Deliberate abstention cases (out-of-corpus, rate ambiguity, no definitive answers)

### Metrics

The evaluation harness (`eval/run_eval.py`) computes:
- **RAGAS Context Precision** — Are retrieved docs relevant?
- **RAGAS Context Recall** — Did we retrieve all needed info?
- **RAGAS Faithfulness** — Is the answer grounded in retrieved context?
- **RAGAS Answer Relevancy** — Does the answer address the question?

### Failure Classification

Misses are categorized into:
- **Retrieval Failure** — Golden docs not in top-K
- **Reranking Failure** — Relevant doc retrieved but ranked low
- **Generation Failure** — Retrieved context present but not used correctly
- **Synthesis Failure** — Other errors

Reports are saved to **`eval/reports/`** (git-tracked).

### Running Evaluation

```bash
# Full evaluation
python cli.py eval

# Model comparison
python cli.py compare-models

# Custom golden set
python eval/run_eval.py --golden-set my_questions.json --output results.json
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_retrieval.py -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- **`test_retrieval.py`** — Hybrid retrieval, RRF fusion, reranking
- **`test_generation.py`** — Answer generation, citation validation, abstention logic

---

## 🛡️ Guardrails

1. **Citations**: Every non-abstained answer includes ≥1 clause-level citation with source document.

2. **Prudent Abstention**: The system never makes definitive eligibility/approval calls. Instead, it surfaces applicable policy language and defers final decisions to human review.

3. **Resilience**: 
   - Exponential backoff retry logic on Gemini API calls
   - Parse/validation failures degrade to safe abstention (no crashes)
   - Empty retrieval results trigger structured "insufficient context" response

4. **Configuration Safety**: All parameters externalized to `config.yaml` — no magic numbers in code.

---

## 📚 Data

**All 40 corpus documents are synthetic**, generated by `scripts/generate_corpus.py`. No real customer data, account details, card information, or proprietary business logic appears anywhere in this repository.

Documents include:
- **Product Disclosures** (14 docs): Savings, checking, cards, loans
- **Fee Schedules** (3 docs): Deposit accounts, cards, wires/transfers
- **Eligibility Policies** (2 docs): Credit requirements, general eligibility
- **Customer FAQs** (8 docs): Curated questions and answers
- **Loan Terms** (6 docs): Auto, personal, home improvement, debt consolidation
- **Manifest** (1 doc): Index of all corpus documents

---

## 🔧 Development

### Setup Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Format code
black src/ tests/ eval/

# Lint
flake8 src/ tests/ eval/
```

### Adding New Documents

1. Place `.md` files in `data/corpus/`
2. Update `data/corpus/_manifest.json` with metadata
3. Run `python cli.py setup` to rebuild the index

### Modifying Config

1. Edit `config.yaml` (YAML format)
2. Changes take effect on next pipeline run (no restart needed)
3. Reload via `src/config.py` (loaded at pipeline initialization)

---

## 📋 Pre-Submission Checklist

- [ ] Run `python cli.py setup` with valid Gemini API key
- [ ] Run `python cli.py eval` and commit generated `eval/reports/` artifacts
- [ ] Fill in `docs/sample-outputs.md` with 2-3 real examples + citations
- [ ] Update model selection rationale in `eval/reports/model_comparison.md`
- [ ] Tests passing: `pytest tests/ -v`
- [ ] Code formatted: `black src/ tests/`
- [ ] Git history: At least 3 PRs with `git merge --no-ff` (no direct main pushes)

---

## 📝 License

This project is part of an educational/assessment submission. Refer to your institution's guidelines.

---

## 🤝 Contributing

For improvements or bug reports, please:
1. Create an issue describing the problem or feature request
2. Reference relevant acceptance criteria from `specs/acceptance_criteria.md`
3. Include evaluation results if the change affects retrieval or generation quality

---

## 📞 Support

**Documentation**: See `docs/` for detailed architecture, business case, and embedding rationale.  
**Questions**: Check `docs/sample-outputs.md` and `eval/golden_set.json` for examples.  
**Bugs**: Open an issue with evaluation results and the failing question.

---

**Last Updated**: September 2026
