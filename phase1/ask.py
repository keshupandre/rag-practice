import argparse
from pathlib import Path

from google import genai
import numpy as np
import voyageai
import faiss
from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts, rank_chunks
from phase1.retrieve import INDEX_BUILD_HINT, load_index
from phase1.rag_input import build_rag_input
ROOT_DIR = Path(__file__).parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"
INDEX_PATH = INDEX_DIR / "index.faiss"

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--rank_method", type=str, default="cosine", choices=["cosine", "faiss"])
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()

def rank_chunks_faiss(chunks: list[dict], query_embedding: np.ndarray, embeddings: np.ndarray, k: int)-> list[dict]:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index file: {INDEX_PATH}. {INDEX_BUILD_HINT}")

    index = faiss.read_index(str(INDEX_PATH))
    query_vector = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results: list[dict] = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "rank": int(idx),
            "score": float(score),
            "chunk": chunks[int(idx)],
        })
    return results

def main()-> None:
    args = parse_args()
    query = args.query
    k = args.k
    settings = get_settings()

    client = genai.Client(api_key=settings.require_gemini_api_key())
    embedding_client = voyageai.Client(api_key=settings.require_voyage_api_key())

    chunks, embeddings = load_index()

    query_embedding = embed_texts([query], embedding_client, settings.voyage_embedding_model)[0]
    ranked_chunks: list[dict] = []
    if args.rank_method == "cosine":
        ranked_chunks = rank_chunks(chunks, query_embedding, embeddings, k)
    elif args.rank_method == "faiss":
        ranked_chunks = rank_chunks_faiss(chunks, query_embedding, embeddings, k)
    else:
        raise ValueError(f"Invalid rank method: {args.rank_method}. Choose from 'cosine' or 'faiss'.")

    context ="\n\n".join(f"[chunk {r['chunk']['id']}] {r['chunk']['text']}" for r in ranked_chunks)
    rag_input = build_rag_input(query, context)
    interaction = client.interactions.create(model=settings.gemini_model, input=rag_input)
    print("\n\nAnswer:\n")
    print(interaction.output_text)

    source_ids = [r["chunk"]["id"] for r in ranked_chunks]
    print(f"\nSources: {source_ids}")


if __name__ == "__main__":
    main()