import argparse
from pathlib import Path

from pinecone import Pinecone
import voyageai

from phase0.config import get_settings
from phase2.bm25_index import bm25_search, build_bm25, load_chunks, print_results
from phase2.index import run_query

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"


def fuse_hybrid(
    bm25_results: list[dict],
    vector_results: list[dict],
    *,
    alpha: float,
    top_k: int,
) -> list[dict]:

    bm25_by_id = {r["id"]: r for r in bm25_results}
    vector_by_id = {r["id"]: r for r in vector_results}
    chunk_ids = set(bm25_by_id) | set(vector_by_id)

    fused: list[dict] = []
    for chunk_id in chunk_ids:
        bm25_hit = bm25_by_id.get(chunk_id)
        vector_hit = vector_by_id.get(chunk_id)
        bm25_score = float(bm25_hit["score"]) if bm25_hit else 0.0
        vector_score = float(vector_hit["score"]) if vector_hit else 0.0
        ref = bm25_hit or vector_hit or {}

        fused.append(
            {
                "id": chunk_id,
                "score": alpha * bm25_score + (1 - alpha) * vector_score,
                "bm25_score": bm25_score,
                "vector_score": vector_score,
                "text": ref.get("text", ""),
                "source": ref.get("source", ""),
            }
        )

    fused.sort(key=lambda row: row["score"], reverse=True)
    return fused[:top_k]

def normalize_bm25_score(bm25_results: list[dict]) -> list[dict]:
    max_score = max(result["score"] for result in bm25_results)
    min_score = min(result["score"] for result in bm25_results)
    return [{**result, "score": (result["score"] - min_score) / (max_score - min_score)} for result in bm25_results]

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the Phase 2 hybrid index")
    parser.add_argument("--query", type=str, default="", help="The query to search for")
    parser.add_argument("--top_k", type=int, default=5, help="The number of results to return")
    parser.add_argument("--chunks-path", type=Path, default=CHUNK_PATH, help="The path to the file to index")
    parser.add_argument("--alpha", type=float, default=0.5, help="The alpha value for the hybrid index")
    return parser.parse_args()


def main() -> None:
    args = arg_parser()
    query = args.query.strip()
    if not query:
        raise SystemExit("Pass --query to search the index")

    settings = get_settings()

    client = voyageai.Client(api_key=settings.require_voyage_api_key())
    pc = Pinecone(api_key=settings.require_pinecone_api_key())
    index = pc.Index(name=settings.pinecone_index_name)

    chunks = load_chunks(args.chunks_path)
    bm25 = build_bm25(chunks)
    bm25_results = bm25_search(bm25, chunks, query, args.top_k)
    normalized_bm25_results = normalize_bm25_score(bm25_results)

    vector_results = run_query(query, client, index, settings, args.top_k)

    hybrid_results = fuse_hybrid(
        normalized_bm25_results,
        vector_results,
        alpha=args.alpha,
        top_k=args.top_k,
    )
    print(f"\nQuery: {query}  (alpha={args.alpha})\n")
    print_results(hybrid_results)


if __name__ == "__main__":
    main()