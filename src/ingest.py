from __future__ import annotations

import os
import hashlib
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text
from src.vectordb import VectorDB


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _stable_id(file_path: str, chunk_id: int, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(file_path.encode("utf-8", errors="ignore"))
    h.update(str(chunk_id).encode("utf-8"))
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def ingest_file(
    file_path: str,
    vdb: VectorDB,
    embedder: SentenceTransformer,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> int:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = _read_text_file(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    chunk_texts: List[str] = [c.text for c in chunks]
    embeddings = embedder.encode(chunk_texts, normalize_embeddings=True).tolist()

    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for c in chunks:
        ids.append(_stable_id(file_path, c.chunk_id, c.text))
        metadatas.append(
            {
                "source": os.path.basename(file_path),
                "path": file_path,
                "chunk_id": c.chunk_id,
            }
        )

    vdb.upsert(ids=ids, documents=chunk_texts, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)
