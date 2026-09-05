"""
Generation layer: takes reranked chunks + question, prompts Gemini to answer
strictly from the provided context, and validates the output against
AnswerResponse. Abstains rather than fabricating when grounding is weak
(AC-05), and never issues answers as definitive legal/financial advice
(Grounding & Advice Rule).
"""
import json
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import ValidationError

from src.config import load_config, get_api_key
from src.schemas import AnswerResponse, Citation

SYSTEM_PROMPT = """You are a retail-banking product & policy assistant. Answer ONLY using the
provided context clauses. Every factual claim must be traceable to a specific clause.

Rules:
- If the context does not contain enough information to answer confidently, set "abstained": true,
  explain why in "abstention_reason", and leave "answer" as a short statement that you cannot
  confirm this from available documentation.
- Never state a definitive eligibility/approval decision as financial advice — describe what the
  policy says, not what will happen to the specific customer.
- Cite every clause you use in "citations" with its doc_id, clause_id, and source_file exactly as
  given in the context.
- confidence should reflect how directly the context supports the answer (0.0-1.0).

Return ONLY valid JSON matching this shape, no markdown fences:
{{
  "answer": "...",
  "citations": [{{"doc_id": "...", "clause_id": "...", "source_file": "..."}}],
  "applicable_product": "..." or null,
  "relevant_fee_or_charge": "..." or null,
  "eligibility_criteria": "..." or null,
  "confidence": 0.0-1.0,
  "abstained": true/false,
  "abstention_reason": "..." or null
}}

Context clauses:
{context}

Question: {question}
"""


def _format_context(chunks: list) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[doc_id={c['doc_id']} clause_id={c['clause_id']} source_file={c['source_file']}]\n{c['text']}"
        )
    return "\n\n".join(lines)


class Generator:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        genai.configure(api_key=get_api_key())
        gen_cfg = self.config["generation"]
        self.model = genai.GenerativeModel(gen_cfg["model"])
        self.threshold = gen_cfg["abstention_confidence_threshold"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def _call_model(self, prompt: str) -> str:
        gen_cfg = self.config["generation"]
        resp = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=gen_cfg["temperature"],
                max_output_tokens=gen_cfg["max_output_tokens"],
            ),
        )
        return resp.text

    def generate(self, question: str, chunks: list) -> AnswerResponse:
        if not chunks:
            return AnswerResponse(
                answer="I don't have documentation to answer this question.",
                citations=[], confidence=0.0, abstained=True,
                abstention_reason="No retrieved context for this query.",
            )

        prompt = SYSTEM_PROMPT.format(context=_format_context(chunks), question=question)

        try:
            raw = self._call_model(prompt)
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            answer = AnswerResponse(**parsed)
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            # Graceful failure per NFR-05: safe abstention fallback, never a crash
            return AnswerResponse(
                answer="I'm unable to produce a validated answer right now. Please rephrase or try again.",
                citations=[], confidence=0.0, abstained=True,
                abstention_reason=f"Generation/validation error: {type(e).__name__}",
            )

        # enforce abstention threshold at the code level, not just the model's self-report
        if answer.confidence < self.threshold and not answer.abstained:
            answer.abstained = True
            answer.abstention_reason = (
                answer.abstention_reason or "Confidence below configured abstention threshold."
            )

        return answer
