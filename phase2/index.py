
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
from phase2.bm25_index import print_results
from phase2.chunk import chunk_text, load_markdown
from phase2.store import clear_index, matches_to_results, search_chunks, upsert_chunks

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
DEFAULT_FILE_DIR = ROOT_DIR / "phase2" / "docs"
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
        default="paragraph",
        help="Chunking strategy",
    )
    parser.add_argument("--query", type=str, default="", help="Query Pinecone instead of indexing")
    parser.add_argument("--top_k", type=int, default=3, help="Results for --query")
    return parser.parse_args()


def collect_chunks(
    files: list[Path],
    *,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    records: list[dict] = []
    next_id = 0

    for path in files:
        text = load_markdown(path)
        texts = chunk_text(text, strategy, chunk_size, overlap)
        for text_chunk in texts:
            records.append(
                {
                    "id": next_id,
                    "source": path.name,
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "text": text_chunk,
                }
            )
            next_id += 1
        print(f"file: {path.name}, n_chunks: {len(texts)}")

    return records


def write_local_index(chunks: list[dict], meta: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNK_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    INDEX_META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"n_chunks: {len(chunks)}")
    print(f"saved: {CHUNK_PATH}")
    print(f"saved: {INDEX_META_PATH}")


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
    client: voyageai.Client,
    index,
    settings,
) -> None:
    files = sorted(args.file_dir.glob("*.md"))
    if not files:
        print(
            f"warning: no *.md files in {args.file_dir.resolve()}; index not written",
            file=sys.stderr,
        )
        raise SystemExit(1)

    chunks = collect_chunks(
        files,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_dir": str(args.file_dir.resolve()),
        "sources": [path.name for path in files],
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "embed_model": settings.voyage_embedding_model,
        "pinecone_index": settings.pinecone_index_name,
        "n_chunks": len(chunks),
    }
    write_local_index(chunks, meta)

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

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"saved: {EMBEDDINGS_PATH}")
    print(f"embedding_dim: {embeddings.shape[1]}")


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

    run_index(args, client=client, index=index, settings=settings)


if __name__ == "__main__":
    main()
