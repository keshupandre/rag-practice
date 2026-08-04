import argparse
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"
CHUNK_PATH = INDEX_DIR / "chunks.jsonl"


def load_chunks(path:Path)-> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    data.sort(key=lambda c: c["id"])
    return data

def build_bm25(chunks: list[dict])-> BM25Okapi:
    documents = [chunk["text"] for chunk in chunks]
    tokenized_documents = [doc.lower().split() for doc in documents]

    bm25 = BM25Okapi(tokenized_documents)
    return bm25

def bm25_search(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]

    return [
        {**chunks[index], "score": float(bm25_scores[index])}
        for index in top_k_indices
    ]

def print_results(results: list[dict]) -> None:
    for rank, result in enumerate(results, start=1):
        preview = (result.get("text") or "").replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(
            f"{rank}. score={result.get('score', '—')}  "
            f"id={result.get('id', '—')}  source={result.get('source', '—')}"
        )
        if "bm25_score" in result or "vector_score" in result:
            print(
                f"   bm25={result.get('bm25_score', '—')}  "
                f"vector={result.get('vector_score', '—')}"
            )
        if preview:
            print(f"   {preview}")
        print("-" * 100)

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the Phase 2 bm25 sparse index")
    parser.add_argument("--top_k", type=int, default=5, help="The number of results to return")
    parser.add_argument("--query", type=str, default="", help="The query to search for")
    parser.add_argument("--chunks-path", type=Path, default=CHUNK_PATH, help="The path to the file to index")
    return parser.parse_args()

def main()-> None:
    args = parse_args()
    chunks = load_chunks(args.chunks_path)
    query = args.query

    if not query.strip():
        raise SystemExit("Pass --query to search the index")

    bm25 = build_bm25(chunks)

    results = bm25_search(bm25, chunks, query, args.top_k)
    print_results(results)




if __name__ == "__main__":
    main()