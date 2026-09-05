from pydantic import BaseModel, Field
from typing import List, Optional


class Citation(BaseModel):
    doc_id: str = Field(..., description="Source document id, e.g. ACCT-01")
    clause_id: str = Field(..., description="Clause id, e.g. ACCT-01-3")
    source_file: str = Field(..., description="Filename in data/corpus/")


class RetrievedChunk(BaseModel):
    text: str
    doc_id: str
    clause_id: str
    source_file: str
    doc_type: str
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class AnswerResponse(BaseModel):
    """Validated structured output per AC-06 / 7.4."""
    answer: str = Field(..., description="Natural-language answer grounded in the corpus")
    citations: List[Citation] = Field(default_factory=list)
    applicable_product: Optional[str] = Field(None, description="Product name the answer pertains to, if any")
    relevant_fee_or_charge: Optional[str] = Field(None, description="Specific fee/charge referenced, if any")
    eligibility_criteria: Optional[str] = Field(None, description="Eligibility criteria referenced, if any")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Grounding confidence, 0-1")
    abstained: bool = Field(False, description="True if the system declined to answer due to insufficient grounding")
    abstention_reason: Optional[str] = Field(None)


class GoldenQuestion(BaseModel):
    question_id: str
    question: str
    reference_answer: str
    expected_source_files: List[str] = Field(default_factory=list)
    ac_refs: List[str] = Field(default_factory=list, description="AC-NN identifiers this question exercises")
