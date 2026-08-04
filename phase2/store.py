from __future__ import annotations

from typing import Any

import numpy as np
from pinecone import Index

UPSERT_BATCH_SIZE = 100


def clear_index(index: Index, *, namespace: str = "", index_name: str = "") -> None:
    """Remove all vectors in a namespace (default namespace when empty)."""
    index.delete(delete_all=True, namespace=namespace)
    label = index_name or "Pinecone index"
    ns_label = namespace or "(default)"
    print(f"Cleared all vectors in {label} namespace {ns_label}")


def embedding_to_list(embedding: np.ndarray) -> list[float]:
    """Convert one embedding (1d or a single row) to a Pinecone dense vector."""
    if embedding.ndim == 2:
        if embedding.shape[0] != 1:
            raise ValueError(f"expected one query vector, got shape {embedding.shape}")
        embedding = embedding[0]
    elif embedding.ndim != 1:
        raise ValueError(f"expected 1d or 2d embedding, got shape {embedding.shape}")
    return embedding.astype(np.float32).tolist()


def build_vectors(chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunk/embedding count mismatch: {len(chunks)} chunks, "
            f"{embeddings.shape[0]} embedding rows"
        )

    return [
        {
            "id": str(chunk["id"]),
            "values": embedding_to_list(row),
            "metadata": {
                "source": chunk["source"],
                "strategy": chunk["strategy"],
                "chunk_size": chunk["chunk_size"],
                "overlap": chunk["overlap"],
                "text": chunk["text"],
            },
        }
        for chunk, row in zip(chunks, embeddings, strict=True)
    ]


def upsert_chunks(
    chunks: list[dict],
    embeddings: np.ndarray,
    index: Index,
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
    index_name: str = "",
) -> int:
    """Upsert chunk embeddings; returns total vectors reported as upserted."""
    vectors = build_vectors(chunks, embeddings)
    upserted = 0
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        response = index.upsert(vectors=batch)
        upserted += response.upserted_count

    label = index_name or "Pinecone index"
    print(f"Upserted {upserted} vectors to {label}")
    return upserted


def match_to_result(match: Any) -> dict:
    meta = match.metadata or {}
    return {
        "id": int(match.id),
        "score": float(match.score),
        "source": meta.get("source", ""),
        "text": meta.get("text", ""),
    }


def matches_to_results(matches: list[Any]) -> list[dict]:
    return [match_to_result(match) for match in matches]


def search_chunks(
    query_embedding: np.ndarray,
    index: Index,
    *,
    top_k: int = 3,
) -> list[Any]:
    return index.query(
        vector=embedding_to_list(query_embedding),
        top_k=top_k,
        include_metadata=True,
    ).matches


def print_search_matches(matches: list[Any]) -> None:
    if not matches:
        print("No matches.")
        return

    for rank, match in enumerate(matches, start=1):
        meta = match.metadata or {}
        preview = (meta.get("text") or "").replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."

        print(
            f"{rank}. score={match.score:.4f}  id={match.id}  "
            f"source={meta.get('source', '—')}"
        )
        if preview:
            print(f"   {preview}")
