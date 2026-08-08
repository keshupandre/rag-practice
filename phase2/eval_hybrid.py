

import argparse
import json
import time
from pathlib import Path

from pinecone import Pinecone
import voyageai

from phase0.config import get_settings
from phase2.bm25_index import bm25_search, build_bm25, load_chunks, print_results
from phase2.hybrid import fuse_hybrid, normalize_bm25_score
from phase2.index import run_query
from phase2.paths import DEFAULT_STRATEGY, INDEX_BUILD_HINT, resolve_chunks_path

STRATEGY_CHOICES = ("paragraph", "fixed")
EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
EVAL_PATH = EVAL_DIR / "golden_phase2.jsonl"

def load_queries(path: Path) -> list[dict]:
    queries: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            queries.append(
                {
                    "query": data["query"],
                    "expected_sources": data["expected_sources"],
                }
            )
    return queries


def hit_at_k(results: list[dict], expected_sources: list[str]) -> bool:
    expected = set(expected_sources)
    return any(result.get("source") in expected for result in results)

def arg_parser()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 2 hybrid index")
    parser.add_argument("--top_k", type=int, default=5, help="The number of results to return")
    parser.add_argument("--alpha", type=float, default=0.5, help="The alpha value for the hybrid index")
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default=DEFAULT_STRATEGY,
        help="Chunking strategy used at index time (must match Pinecone upsert)",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Override chunks JSONL (default: data/index/chunks_<strategy>.jsonl)",
    )
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH, help="The path to the file to index")
    parser.add_argument(
        "--embed-delay",
        type=float,
        default=30.0,
        help="Seconds to wait before each Voyage embed (free tier ~3 RPM; use 0 to disable)",
    )
    return parser.parse_args()



def main() -> None:
    args = arg_parser()

    settings = get_settings()

    client = voyageai.Client(api_key=settings.require_voyage_api_key())
    pc = Pinecone(api_key=settings.require_pinecone_api_key())
    index = pc.Index(name=settings.pinecone_index_name)

    queries = load_queries(args.eval_path)

    chunks_path = resolve_chunks_path(args.strategy, args.chunks_path)
    if not chunks_path.exists():
        raise SystemExit(
            f"Missing chunks file: {chunks_path}. "
            f"{INDEX_BUILD_HINT.format(strategy=args.strategy)}"
        )

    print(f"Using chunks: {chunks_path}  (strategy={args.strategy})")
    chunks = load_chunks(chunks_path)
    bm25 = build_bm25(chunks)

    hits = 0
    hits_vector = 0
    hits_bm25 = 0
    for i, query in enumerate(queries):
        if i > 0 and args.embed_delay > 0:
            print(f"Waiting {args.embed_delay:.0f}s (Voyage rate limit)...")
            time.sleep(args.embed_delay)

        bm25_results = bm25_search(bm25, chunks, query["query"], args.top_k)
        normalized_bm25_results = normalize_bm25_score(bm25_results)
        vector_results = run_query(query["query"], client, index, settings, args.top_k)
        hybrid_results = fuse_hybrid(
            normalized_bm25_results,
            vector_results,
            alpha=args.alpha,
            top_k=args.top_k,
        )

        actual_sources = [result["source"] for result in hybrid_results]
        matched_vector = hit_at_k(vector_results, query["expected_sources"])
        matched_bm25 = hit_at_k(bm25_results, query["expected_sources"])
        matched = hit_at_k(hybrid_results, query["expected_sources"])

        print(f"\nQuery: {query['query']}  (alpha={args.alpha})\n")
        print_results(hybrid_results)
        print(f"Expected: {query['expected_sources']}")
        print(f"Actual (top-{args.top_k}): {actual_sources}")
        print(f"Vector hit: {matched_vector}")
        print(f"BM25 hit: {matched_bm25}")
        print("Hit @ k" if matched else "Miss @ k")

        if matched:
            hits += 1
        if matched_vector:
            hits_vector += 1
        if matched_bm25:
            hits_bm25 += 1

    if queries:
        recall = hits / len(queries)
        recall_vector = hits_vector / len(queries)
        recall_bm25 = hits_bm25 / len(queries)
        print(f"Vector hit: {hits_vector}/{len(queries)} ({recall_vector:.0%})")
        print(f"BM25 hit: {hits_bm25}/{len(queries)} ({recall_bm25:.0%})")
        print(f"\nRecall@{args.top_k}: {hits}/{len(queries)} ({recall:.0%})")


if __name__ == "__main__":
    main()