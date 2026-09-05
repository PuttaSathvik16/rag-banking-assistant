"""End-to-end RAG pipeline: query transform -> hybrid retrieve -> rerank -> generate."""
from src.config import load_config
from src.retrieval import HybridRetriever
from src.reranker import Reranker
from src.query_transform import QueryTransformer
from src.generation import Generator
from src.schemas import AnswerResponse


class RAGPipeline:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.retriever = HybridRetriever(self.config)
        self.reranker = Reranker(self.config)
        self.query_transformer = QueryTransformer(self.config)
        self.generator = Generator(self.config)

    def answer(self, question: str) -> AnswerResponse:
        sub_queries = self.query_transformer.transform(question)

        all_candidates = []
        seen_chunk_ids = set()
        for sq in sub_queries:
            for c in self.retriever.retrieve(sq):
                if c["chunk_id"] not in seen_chunk_ids:
                    all_candidates.append(c)
                    seen_chunk_ids.add(c["chunk_id"])

        reranked = self.reranker.rerank(question, all_candidates)
        return self.generator.generate(question, reranked)


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is the minimum balance for the Premium Savings account?"
    pipeline = RAGPipeline()
    result = pipeline.answer(question)
    print(result.model_dump_json(indent=2))
