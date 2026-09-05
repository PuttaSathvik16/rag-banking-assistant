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
