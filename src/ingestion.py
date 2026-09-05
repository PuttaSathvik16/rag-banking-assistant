"""
Ingestion pipeline: parses the clause-tagged markdown corpus, applies
token-aware chunking (using tiktoken for precise token counting), embeds with
a local sentence-transformers model, and idempotently upserts into a persisted
Chroma collection with source/clause metadata.
"""
import os
import re
import glob
import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

from src.config import load_config, REPO_ROOT

CORPUS_DIR = os.path.join(REPO_ROOT, "data", "corpus")


def parse_document(filepath: str):
    """Extract doc_id, doc_type, title, and (clause_id, heading, body) tuples."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    doc_id_match = re.search(r"^doc_id:\s*(.+)$", text, re.MULTILINE)
    doc_type_match = re.search(r"^doc_type:\s*(.+)$", text, re.MULTILINE)
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)

    doc_id = doc_id_match.group(1).strip() if doc_id_match else os.path.basename(filepath)
    doc_type = doc_type_match.group(1).strip() if doc_type_match else "unknown"
    title = title_match.group(1).strip() if title_match else os.path.basename(filepath)

    clause_pattern = re.compile(r"^##\s+\[(?P<cid>[\w-]+)\]\s+(?P<heading>.+)$", re.MULTILINE)
    matches = list(clause_pattern.finditer(text))
    clauses = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        clauses.append((m.group("cid"), m.group("heading").strip(), body))

    return doc_id, doc_type, title, clauses


def token_window_chunks(clause_text: str, chunk_size_tokens: int, overlap_tokens: int, min_chunk_tokens: int):
    """
    Token-aware windowing: splits text into overlapping chunks based on actual
    token count (using tiktoken) rather than sentence or character boundaries.
    This ensures precise control over context size for the LLM.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(clause_text.strip())

    if len(tokens) <= min_chunk_tokens:
        return [clause_text.strip()]

    chunks = []
    step = max(chunk_size_tokens - overlap_tokens, 1)

    for start_idx in range(0, len(tokens), step):
        end_idx = start_idx + chunk_size_tokens
        chunk_tokens = tokens[start_idx:end_idx]

        if len(chunk_tokens) >= min_chunk_tokens:
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)

        if end_idx >= len(tokens):
            break

    return chunks if chunks else [clause_text.strip()]


def build_index(config: dict = None, verbose: bool = True):
    config = config or load_config()
    chunk_cfg = config["chunking"]
    emb_cfg = config["embedding"]
    vs_cfg = config["vector_store"]

    persist_dir = os.path.join(REPO_ROOT, vs_cfg["persist_dir"])
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(vs_cfg["collection_name"])

    model = SentenceTransformer(emb_cfg["model_name"])

    filepaths = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    if verbose:
        print(f"Ingesting {len(filepaths)} documents from {CORPUS_DIR}")

    all_ids, all_texts, all_metadatas = [], [], []

    for filepath in filepaths:
        doc_id, doc_type, title, clauses = parse_document(filepath)
        source_file = os.path.basename(filepath)
        for clause_id, heading, body in clauses:
            chunks = token_window_chunks(
                body, chunk_cfg["chunk_size_tokens"], chunk_cfg["overlap_tokens"], chunk_cfg["min_chunk_tokens"]
            )
            for j, chunk_text in enumerate(chunks):
                chunk_id = f"{clause_id}::chunk{j}"  # deterministic -> idempotent upsert
                all_ids.append(chunk_id)
                all_texts.append(f"{heading}. {chunk_text}")
                all_metadatas.append({
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "clause_id": clause_id,
                    "heading": heading,
                    "source_file": source_file,
                    "title": title,
                })

    if not all_ids:
        raise RuntimeError("No chunks produced. Run scripts/generate_corpus.py first.")

    embeddings = model.encode(all_texts, show_progress_bar=verbose, normalize_embeddings=True).tolist()

    # upsert = idempotent: re-running ingestion overwrites existing ids rather than duplicating
    collection.upsert(ids=all_ids, documents=all_texts, metadatas=all_metadatas, embeddings=embeddings)

    if verbose:
        print(f"Indexed {len(all_ids)} chunks into Chroma collection '{vs_cfg['collection_name']}' at {persist_dir}")

    return collection, all_ids, all_texts, all_metadatas


if __name__ == "__main__":
    build_index()
