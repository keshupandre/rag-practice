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
INDEX_BUILD_HINT = "run python -m phase1.index"


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    chunks.sort(key=lambda c: c["id"])
    return chunks


def load_index(
    chunk_path: Path = CHUNK_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> tuple[list[dict], np.ndarray]:
    missing = [path for path in (chunk_path, embeddings_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing index file(s): {names}. {INDEX_BUILD_HINT}")

    chunks = load_chunks(chunk_path)
    embeddings = np.load(embeddings_path)

    if len(chunks) != embeddings.shape[0]:
        raise SystemExit(
            f"Index mismatch: {len(chunks)} chunks in {chunk_path.name}, "
            f"{embeddings.shape[0]} rows in {embeddings_path.name}. {INDEX_BUILD_HINT}"
        )

    return chunks, embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()
    client = voyageai.Client(api_key=settings.require_voyage_api_key())

    chunks, embeddings = load_index()

    query_embedding = embed_texts([args.query], client, settings.voyage_embedding_model)[0]
    ranked = rank_chunks(chunks, query_embedding, embeddings, args.k)
    print_ranked_chunks(args.query, ranked)




if __name__ == "__main__":
    main()