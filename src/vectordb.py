from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings


class VectorDB:
    """
    Thin wrapper around Chroma (persistent).
    Stores:
      - documents (chunk text)
      - embeddings
      - metadata (filename, chunk_id, etc.)
    """

    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "docs"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []
        return docs, metas, dists
