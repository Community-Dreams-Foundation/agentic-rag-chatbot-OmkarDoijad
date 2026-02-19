from __future__ import annotations

import json
import os
from typing import Optional

import typer
from rich import print
from sentence_transformers import SentenceTransformer

from src.vectordb import VectorDB
from src.ingest import ingest_file
from src.rag import answer_question
from src.memory import append_memory

app = typer.Typer(no_args_is_help=True)


def _get_embedder() -> SentenceTransformer:
    # Small + fast embedding model (good for hackathon + judges)
    return SentenceTransformer("all-MiniLM-L6-v2")


def _get_vdb() -> VectorDB:
    return VectorDB(persist_dir="data/chroma", collection_name="docs")


@app.command()
def upload(path: str):
    """Ingest a local text/markdown file into Chroma."""
    vdb = _get_vdb()
    embedder = _get_embedder()
    n = ingest_file(path, vdb, embedder)
    print(f"[green]Uploaded[/green] {path} → indexed {n} chunks.")


@app.command()
def ask(question: str, top_k: int = 4):
    """Ask a question grounded in uploaded docs; prints citations."""
    vdb = _get_vdb()
    embedder = _get_embedder()
    res = answer_question(question, vdb, embedder, top_k=top_k)

    print("\n[bold]Answer:[/bold]")
    print(res.answer[:1200] + ("…" if len(res.answer) > 1200 else ""))

    if res.citations:
        print("\n[bold]Citations:[/bold]")
        for c in res.citations:
            print(f"- {c.source} (chunk {c.chunk_id}): {c.snippet}")

    # Minimal memory example (later you’ll make this selective)
    append_memory("USER", "User asked a question about uploaded documents.")


@app.command()
def sanity():
    """Run a minimal end-to-end flow and write artifacts/sanity_output.json."""
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("sample_docs", exist_ok=True)

    # Create a tiny deterministic doc for sanity
    doc_path = os.path.join("sample_docs", "sanity_doc.md")
    if not os.path.exists(doc_path):
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "# Sanity Doc\n\n"
                "AIMS stands for Automated Integrated Migration System.\n"
                "PTS stands for Project Tracking System.\n"
                "WLM stands for Workload Management System.\n"
            )

    vdb = _get_vdb()
    embedder = _get_embedder()

    ingested = ingest_file(doc_path, vdb, embedder)
    question = "What does AIMS stand for?"
    res = answer_question(question, vdb, embedder, top_k=3)

    append_memory("USER", "User is testing the bot using sanity flow.")
    append_memory("COMPANY", "Sanity flow verifies ingestion, retrieval, citations, and memory writing.")

    output = {
        "status": "ok",
        "ingested_chunks": ingested,
        "question": question,
        "answer_preview": res.answer[:300],
        "citations": [
            {"source": c.source, "chunk_id": c.chunk_id, "snippet": c.snippet}
            for c in res.citations
        ],
    }

    with open("artifacts/sanity_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("[green]Sanity complete[/green] → wrote artifacts/sanity_output.json")


if __name__ == "__main__":
    app()
