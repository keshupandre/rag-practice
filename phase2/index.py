
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import voyageai
from pinecone import Pinecone

from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts
from phase2.bm25_index import load_chunks, print_results
from phase2.chunk import collect_chunks
from phase2.paths import DEFAULT_STRATEGY, INDEX_DIR, index_paths
from phase2.store import clear_index, matches_to_results, search_chunks, upsert_chunks

DEFAULT_FILE_DIR = Path(__file__).resolve().parents[1] / "phase2" / "docs"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the Phase 2 vector index")
    parser.add_argument(
        "--file_dir",
        type=Path,
        default=DEFAULT_FILE_DIR,
        help="Directory of markdown files to index",
    )
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--strategy",
        choices=["paragraph", "fixed"],
        default=DEFAULT_STRATEGY,
        help="Chunking strategy",
    )
    parser.add_argument("--query", type=str, default="", help="Query Pinecone instead of indexing")
    parser.add_argument("--top_k", type=int, default=3, help="Results for --query")
    return parser.parse_args()


def write_local_index(
    chunks: list[dict],
    meta: dict,
    *,
    chunk_path: Path,
    meta_path: Path,
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with chunk_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"n_chunks: {len(chunks)}")
    print(f"saved: {chunk_path}")
    print(f"saved: {meta_path}")


def load_index_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise SystemExit(f"Missing index meta: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def restore_saved_index(strategy: str, index, *, settings) -> dict:
    """Load local artifacts and upsert to Pinecone (no re-embed)."""
    paths = index_paths(strategy)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing saved index artifact(s): {names}")

    meta = load_index_meta(paths["meta"])
    chunks = load_chunks(paths["chunks"])
    embeddings = np.load(paths["embeddings"])

    if len(chunks) != embeddings.shape[0]:
        raise SystemExit(
            f"Index mismatch for {strategy}: {len(chunks)} chunks in JSONL, "
            f"{embeddings.shape[0]} rows in embeddings file"
        )

    clear_index(index, index_name=settings.pinecone_index_name)
    upsert_chunks(
        chunks,
        embeddings,
        index,
        index_name=settings.pinecone_index_name,
    )
    print(f"restored Pinecone from {strategy} artifacts ({len(chunks)} chunks)")
    return meta


def run_query(
    query: str,
    client: voyageai.Client,
    index,
    settings,
    top_k: int,
) -> list[dict]:
    query_embedding = embed_texts(
        [query],
        client,
        model=settings.voyage_embedding_model,
    )
    matches = search_chunks(query_embedding, index, top_k=top_k)
    return matches_to_results(matches)


def run_index(
    args: argparse.Namespace,
    strategy: str,
    client: voyageai.Client,
    index,
    settings,
) -> dict[str, Path]:
    files = sorted(args.file_dir.glob("*.md"))
    pdf_files = sorted(args.file_dir.glob("*.pdf"))

    if not pdf_files and not files:
        print(
            f"warning: no *.pdf or *.md files in {args.file_dir.resolve()}; index not written",
            file=sys.stderr,
        )
        raise SystemExit(1)

    paths = index_paths(strategy)
    chunks = collect_chunks(
        files,
        pdf_files=pdf_files,
        strategy=strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_dir": str(args.file_dir.resolve()),
        "sources": [path.name for path in files + pdf_files],
        "strategy": strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "embed_model": settings.voyage_embedding_model,
        "pinecone_index": settings.pinecone_index_name,
        "n_chunks": len(chunks),
    }
    write_local_index(
        chunks,
        meta,
        chunk_path=paths["chunks"],
        meta_path=paths["meta"],
    )

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks],
        client,
        model=settings.voyage_embedding_model,
    )

    clear_index(index, index_name=settings.pinecone_index_name)
    upsert_chunks(
        chunks,
        embeddings,
        index,
        index_name=settings.pinecone_index_name,
    )

    np.save(paths["embeddings"], embeddings)
    print(f"saved: {paths['embeddings']}")
    print(f"embedding_dim: {embeddings.shape[1]}")
    return paths


def main() -> None:
    args = parse_args()
    settings = get_settings()

    client = voyageai.Client(api_key=settings.require_voyage_api_key())
    pc = Pinecone(api_key=settings.require_pinecone_api_key())
    index = pc.Index(name=settings.pinecone_index_name)

    if args.query:
        results = run_query(
            args.query,
            client=client,
            index=index,
            settings=settings,
            top_k=args.top_k,
        )
        print(f"\nQuery: {args.query}\n")
        print_results(results)
        return

    run_index(args, args.strategy, client, index, settings)


if __name__ == "__main__":
    main()
