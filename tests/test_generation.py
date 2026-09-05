"""
Requires GOOGLE_API_KEY set (or mock the model call). Covers AC-06 (structured
output validation) and AC-05 (abstention behavior).
"""
import pytest
from src.schemas import AnswerResponse, Citation


def test_ac06_answer_response_schema_validates_minimal_payload():
    resp = AnswerResponse(
        answer="The monthly fee is $5.00.",
        citations=[Citation(doc_id="ACCT-01", clause_id="ACCT-01-3", source_file="everyday-checking-disclosure.md")],
        confidence=0.9,
        abstained=False,
    )
    assert resp.confidence == 0.9
    assert resp.citations[0].clause_id == "ACCT-01-3"


def test_ac06_confidence_bounds_are_enforced():
    with pytest.raises(Exception):
        AnswerResponse(answer="x", confidence=1.5, abstained=False)


def test_ac05_generator_abstains_with_empty_context():
    from src.generation import Generator
    from src.config import load_config
    try:
        gen = Generator(load_config())
    except RuntimeError:
        pytest.skip("GOOGLE_API_KEY not set in this environment")
    result = gen.generate("Any question", chunks=[])
    assert result.abstained is True
    assert result.confidence == 0.0


def test_context_budget_truncates_low_ranked_chunks():
    """Verify context budget drops lowest-ranked chunks when budget exceeded."""
    from src.generation import Generator
    from src.config import load_config
    try:
        gen = Generator(load_config())
    except RuntimeError:
        pytest.skip("GOOGLE_API_KEY not set in this environment")

    gen.max_context_tokens = 256  # Set tight budget to force truncation
    mock_chunks = [
        {"chunk_id": "c1", "doc_id": "doc1", "clause_id": "clause1", "text": "Lorem ipsum " * 50, "rerank_score": 0.9},
        {"chunk_id": "c2", "doc_id": "doc1", "clause_id": "clause2", "text": "Lorem ipsum " * 50, "rerank_score": 0.7},
        {"chunk_id": "c3", "doc_id": "doc1", "clause_id": "clause3", "text": "Lorem ipsum " * 50, "rerank_score": 0.5},
        {"chunk_id": "c4", "doc_id": "doc1", "clause_id": "clause4", "text": "Lorem ipsum " * 50, "rerank_score": 0.3},
    ]

    truncated = gen._enforce_context_budget(mock_chunks)
    assert len(truncated) < len(mock_chunks), f"Budget enforcement should drop chunks, got {len(truncated)} of {len(mock_chunks)}"
    if len(truncated) > 1:
        assert truncated[0]["rerank_score"] >= truncated[-1]["rerank_score"], "Chunks should be ordered by score"
