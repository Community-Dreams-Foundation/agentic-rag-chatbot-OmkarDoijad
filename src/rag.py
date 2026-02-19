from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from src.vectordb import VectorDB


@dataclass
class Citation:
    source: str
    chunk_id: int
    snippet: str


@dataclass
class RAGResult:
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]


def answer_question(
    question: str,
    vdb: VectorDB,
    embedder: SentenceTransformer,
    top_k: int = 4,
) -> RAGResult:
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()[0]
    docs, metas, dists = vdb.query(q_emb, top_k=top_k)

    retrieved = []
    citations: List[Citation] = []

    for doc, meta, dist in zip(docs, metas, dists):
        snippet = (doc[:220] + "…") if len(doc) > 220 else doc
        retrieved.append(
            {
                "text": doc,
                "meta": meta,
                "distance": float(dist),
            }
        )
        citations.append(
            Citation(
                source=str(meta.get("source", "unknown")),
                chunk_id=int(meta.get("chunk_id", -1)),
                snippet=snippet,
            )
        )

    if not retrieved:
        return RAGResult(
            answer="I couldn't find anything relevant in the uploaded documents.",
            citations=[],
            retrieved_chunks=[],
        )

    # Minimal “grounded” answer (extractive): return best chunk as answer.
    # You can replace this with an LLM later.
    best = retrieved[0]
    answer = best["text"].strip()

    return RAGResult(answer=answer, citations=citations, retrieved_chunks=retrieved)
