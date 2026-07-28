import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import voyageai

from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts
from phase2.chunk import chunk_text, load_markdown

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"
DEFAULT_FILE_DIR = ROOT_DIR / "phase2" / "docs"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=Path, default=DEFAULT_FILE_DIR, help="directory containing the markdown files")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="chunk size")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="overlap")
    parser.add_argument("--strategy", type=str, default="paragraph", help="chunking strategy")
    return parser.parse_args()

def main()-> None:
    args = parse_args()
    settings = get_settings()

    client = voyageai.Client(api_key=settings.require_voyage_api_key())

    files = sorted(args.file_dir.glob("*.md"))
    if not files:
        print(f"warning: no *.md files in {args.file_dir.resolve()}; index not written", file=sys.stderr)
        raise SystemExit(1)

    all_chunks: list[dict] = []

    count = 0
    for file in files:
        text = load_markdown(file)
        chunks = chunk_text(text, args.strategy, args.chunk_size, args.overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": idx + count,
                    "source": file.name,
                    "strategy": args.strategy,
                    "chunk_size": args.chunk_size,
                    "overlap": args.overlap,
                    "text": chunk,
                }
            )
        count += len(chunks)
        print(f"file: {file.name}, n_chunks: {len(chunks)}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNK_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    index_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_dir": str(args.file_dir.resolve()),
        "sources": [f.name for f in files],
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "embed_model": settings.voyage_embedding_model,
        "n_chunks": count,
    }
    INDEX_META_PATH.write_text(json.dumps(index_meta, indent=2) + "\n", encoding="utf-8")

    print(f"n_chunks: {count}")
    print(f"saved: {CHUNK_PATH}")
    print(f"saved: {INDEX_META_PATH}")

    embeddings = embed_texts(
        [chunk["text"] for chunk in all_chunks],
        client,
        model=settings.voyage_embedding_model,
    )

    embeddings_path = INDEX_DIR / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"saved: {embeddings_path}")



if __name__ == "__main__":
    main()