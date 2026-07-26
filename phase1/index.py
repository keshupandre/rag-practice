import argparse
import json
from pathlib import Path
import numpy as np

import voyageai
from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts
from phase1.chunk import chunk_text, load_markdown

ROOT_DIR = Path(__file__).parents[1]
INDEX_DIR = ROOT_DIR/ "data" / "index"

DEFAULT_FILE_PATH = ROOT_DIR/ "phase1" / "docs" / "rag_notes.md"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default= DEFAULT_FILE_PATH,
        help="The path to the markdown file to parse"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The size of the chunks to create"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="The overlap between chunks"
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    settings = get_settings()

    client = voyageai.Client(api_key=settings.require_voyage_api_key())

    text = load_markdown(args.file_path)
    chunks = chunk_text(text, args.chunk_size, args.overlap)

    embeddings = embed_texts(chunks, client, settings.voyage_embedding_model)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNK_PATH.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            f.write(
                json.dumps(
                    {"id": idx, "source": str(args.file_path), "text": chunk},
                    ensure_ascii=False,
                )
                + "\n"
            )

    
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"n_chunks: {len(chunks)}")
    print(f"embedding_dim: {embeddings.shape[1]}")
    print(f"saved: {CHUNK_PATH}")
    print(f"saved: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()