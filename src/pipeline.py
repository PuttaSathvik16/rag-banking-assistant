"""End-to-end RAG pipeline: query transform -> hybrid retrieve -> rerank -> generate.

Built using LCEL Runnable pattern for composability and clarity.
"""
from langchain_core.runnables import Runnable
from src.config import load_config
from src.retrieval import HybridRetriever
from src.reranker import Reranker
from src.query_transform import QueryTransformer
from src.generation import Generator
from src.schemas import AnswerResponse


class TransformRetrieveRunnable(Runnable):
    """Runnable that transforms query and retrieves candidates."""
    def __init__(self, transformer, retriever):
        self.transformer = transformer
        self.retriever = retriever

    def invoke(self, input: str, config=None):
        sub_queries = self.transformer.transform(input)
        all_candidates = []
        seen_chunk_ids = set()
        for sq in sub_queries:
            for c in self.retriever.retrieve(sq):
                if c["chunk_id"] not in seen_chunk_ids:
                    all_candidates.append(c)
                    seen_chunk_ids.add(c["chunk_id"])
        return {"candidates": all_candidates, "question": input}


class RerankerRunnable(Runnable):
    """Runnable that reranks candidates."""
    def __init__(self, reranker):
        self.reranker = reranker

    def invoke(self, input: dict, config=None):
        reranked = self.reranker.rerank(input["question"], input["candidates"])
        return {"reranked": reranked, "question": input["question"]}


class GeneratorRunnable(Runnable):
    """Runnable that generates final answer."""
    def __init__(self, generator):
        self.generator = generator

    def invoke(self, input: dict, config=None):
        return self.generator.generate(input["question"], input["reranked"])


class RAGPipeline:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.retriever = HybridRetriever(self.config)
        self.reranker = Reranker(self.config)
        self.query_transformer = QueryTransformer(self.config)
        self.generator = Generator(self.config)

        self.chain = self._build_chain()

    def _build_chain(self):
        """Build LCEL Runnable chain: transform -> retrieve -> rerank -> generate."""
        return (
            TransformRetrieveRunnable(self.query_transformer, self.retriever)
            | RerankerRunnable(self.reranker)
            | GeneratorRunnable(self.generator)
        )

    def answer(self, question: str) -> AnswerResponse:
        """Answer a question using the LCEL Runnable pipeline."""
        return self.chain.invoke(question)


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is the minimum balance for the Premium Savings account?"
    pipeline = RAGPipeline()
    result = pipeline.answer(question)
    print(result.model_dump_json(indent=2))
