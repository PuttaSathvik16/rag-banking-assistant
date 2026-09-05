"""Query transformation: rewrites/decomposes multi-part or ambiguous
questions before retrieval (AC-07)."""
import json
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import load_config, get_api_key

TRANSFORM_PROMPT = """You are a query preprocessor for a retail-banking retrieval system.
Given a user question, decide if it is a single clear question or a multi-part /
ambiguous one. If multi-part, decompose it into 2-4 standalone sub-questions that
can each be retrieved independently. If it references a pronoun or vague product
("it", "that account"), rewrite it to be self-contained if possible.

Return ONLY valid JSON, no markdown fences:
{{"sub_queries": ["...", "..."]}}

User question: {question}
"""


class QueryTransformer:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        genai.configure(api_key=get_api_key())
        self.model = genai.GenerativeModel(self.config["query_transform"]["model"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def transform(self, question: str) -> list:
        if not self.config["query_transform"]["enabled"]:
            return [question]
        try:
            resp = self.model.generate_content(TRANSFORM_PROMPT.format(question=question))
            text = resp.text.strip().strip("```json").strip("```").strip()
            parsed = json.loads(text)
            sub_queries = parsed.get("sub_queries") or [question]
            return sub_queries if sub_queries else [question]
        except Exception:
            # graceful fallback per NFR-05: never crash, degrade to the original query
            return [question]
