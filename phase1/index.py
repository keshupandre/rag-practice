import argparse
import json
from pathlib import Path
import numpy as np

import faiss
import voyageai
from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts
from phase1.chunk import chunk_text, collect_chunks, load_markdown

ROOT_DIR = Path(__file__).parents[1]
INDEX_DIR = ROOT_DIR/ "data" / "index"

DEFAULT_FILE_DIR = ROOT_DIR/ "phase1" / "docs"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_dir",
        type=Path,
        default=DEFAULT_FILE_DIR,
        help="The directory to the markdown files to parse"
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

    markdown_files = sorted(args.file_dir.glob("*.md"))
    pdf_files = sorted(args.file_dir.glob("*.pdf"))

    if not markdown_files and not pdf_files:
        raise ValueError(f"No markdown or pdf files found in {args.file_dir}")

    chunks = collect_chunks(
        markdown_files,
        pdf_files,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks],
        client,
        settings.voyage_embedding_model,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNK_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    np.save(EMBEDDINGS_PATH, embeddings)

    dimension = embeddings.shape[1]
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"n_chunks: {len(chunks)}")
    print(f"embedding_dim: {embeddings.shape[1]}")
    print(f"saved: {CHUNK_PATH}")
    print(f"saved: {EMBEDDINGS_PATH}")
    print(f"saved: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()