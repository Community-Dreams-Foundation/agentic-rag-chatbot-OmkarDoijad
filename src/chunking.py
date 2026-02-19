from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    chunk_id: int


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """
    Simple, reliable chunking:
    - Normalize whitespace
    - Chunk by characters with overlap
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, chunk_id=chunk_id))
            chunk_id += 1
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks
