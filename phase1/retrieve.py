import argparse
import json
from pathlib import Path

import numpy as np
import voyageai

from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts, print_ranked_chunks, rank_chunks

ROOT_DIR = Path(__file__).parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"

CHUNK_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    chunks.sort(key=lambda c: c["id"])
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()
    client = voyageai.Client(api_key=settings.require_voyage_api_key())

    chunks = load_chunks(CHUNK_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise SystemExit(
            f"Index mismatch: {len(chunks)} chunks in JSONL, "
            f"{embeddings.shape[0]} rows in embeddings.npy"
        )

    query_embedding = embed_texts([args.query], client, settings.voyage_embedding_model)[0]
    ranked = rank_chunks(chunks, query_embedding, embeddings, args.k)
    print_ranked_chunks(args.query, ranked)




if __name__ == "__main__":
    main()