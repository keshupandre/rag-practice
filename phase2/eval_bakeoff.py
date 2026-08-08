
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

from pinecone import Pinecone
import voyageai

from phase0.config import get_settings
from phase2.eval_hybrid import hit_at_k, load_queries
from phase2.index import load_index_meta, restore_saved_index, run_index, run_query
from phase2.paths import index_paths

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT_DIR / "data" / "eval"
EVAL_PATH = EVAL_DIR / "golden_phase2.jsonl"
DEFAULT_FILE_DIR = ROOT_DIR / "phase2" / "docs"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
BAKEOFF_STRATEGIES = ("paragraph", "fixed")


@dataclass
class StrategyEvalResult:
    strategy: str
    n_chunks: int
    hits: int | None = None
    total: int | None = None
    misses: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        if self.hits is None or not self.total:
            return None
        return self.hits / self.total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunking bake-off: index and eval each strategy")
    parser.add_argument(
        "--file_dir",
        type=Path,
        default=DEFAULT_FILE_DIR,
        help="Directory of markdown files to index",
    )
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--top_k", type=int, default=3, help="Results for eval recall@k")
    parser.add_argument("--embed_delay", type=float, default=30.0, help="Delay between Voyage embed calls")
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH, help="Golden-set JSONL path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--index-only",
        action="store_true",
        help="Build and save artifacts only (no golden-set scoring)",
    )
    mode.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-upsert saved artifacts to Pinecone and re-score (no re-embed)",
    )
    return parser.parse_args()


def eval_strategy(
    strategy: str,
    queries: list[dict],
    *,
    client: voyageai.Client,
    index,
    settings,
    top_k: int,
    embed_delay: float,
) -> tuple[int, int, list[dict]]:
    hits = 0
    misses: list[dict] = []

    for i, query in enumerate(queries):
        if i > 0 and embed_delay > 0:
            print(f"Waiting {embed_delay:.0f}s (Voyage rate limit)...")
            time.sleep(embed_delay)

        pinecone_results = run_query(query["query"], client, index, settings, top_k)
        if hit_at_k(pinecone_results, query["expected_sources"]):
            hits += 1
        else:
            actual_sources = [row.get("source") for row in pinecone_results]
            misses.append({**query, "actual_sources": actual_sources})
            print(
                f"  MISS [{strategy}] {query['query']!r} "
                f"(expected {query['expected_sources']}, got {actual_sources})"
            )

    return hits, len(queries), misses


def print_summary_table(results: list[StrategyEvalResult], top_k: int) -> None:
    has_eval = any(result.hits is not None for result in results)
    print(f"\n{'=' * 68}")
    print(f"Bake-off summary" + (f" (recall@{top_k})" if has_eval else " (index only)"))
    print(f"{'=' * 68}")

    if has_eval:
        print(f"{'strategy':<12} {'n_chunks':>8} {'hits':>6} {'total':>6} {'recall':>8}")
        print(f"{'-' * 68}")
        for result in results:
            recall = f"{result.recall:.0%}" if result.recall is not None else "—"
            print(
                f"{result.strategy:<12} {result.n_chunks:>8} {result.hits:>6} "
                f"{result.total:>6} {recall:>8}"
            )
    else:
        print(f"{'strategy':<12} {'n_chunks':>8}")
        print(f"{'-' * 68}")
        for result in results:
            print(f"{result.strategy:<12} {result.n_chunks:>8}")

    print(f"{'=' * 68}")


def print_miss_report(results: list[StrategyEvalResult]) -> None:
    evaluated = [result for result in results if result.hits is not None]
    if not evaluated:
        return

    any_misses = any(result.misses for result in evaluated)
    if not any_misses:
        print("\nNo misses — all queries hit for every strategy.")
        return

    print("\nMiss report")
    print("-" * 68)
    for result in evaluated:
        if not result.misses:
            print(f"\n{result.strategy}: (none)")
            continue
        print(f"\n{result.strategy} ({len(result.misses)} miss(es)):")
        for miss in result.misses:
            print(f"  query:    {miss['query']}")
            print(f"  expected: {miss['expected_sources']}")
            print(f"  actual:   {miss.get('actual_sources', [])}")


def main() -> None:
    args = parse_args()
    settings = get_settings()

    do_index = not args.eval_only
    do_eval = not args.index_only

    client = voyageai.Client(api_key=settings.require_voyage_api_key())
    pc = Pinecone(api_key=settings.require_pinecone_api_key())
    index = pc.Index(name=settings.pinecone_index_name)

    queries = load_queries(args.eval_path) if do_eval else []
    results: list[StrategyEvalResult] = []

    for strategy in BAKEOFF_STRATEGIES:
        paths = index_paths(strategy)

        if do_index:
            print(f"\n=== Index: {strategy} ===")
            run_index(args, strategy, client, index, settings)
            print(f"artifacts: {paths['chunks']}, {paths['meta']}, {paths['embeddings']}")

        if do_eval:
            print(f"\n=== Eval: {strategy} (recall@{args.top_k}) ===")
            if args.eval_only:
                meta = restore_saved_index(strategy, index, settings=settings)
            else:
                meta = load_index_meta(paths["meta"])

            hits, total, misses = eval_strategy(
                strategy,
                queries,
                client=client,
                index=index,
                settings=settings,
                top_k=args.top_k,
                embed_delay=args.embed_delay,
            )
            results.append(
                StrategyEvalResult(
                    strategy=strategy,
                    n_chunks=meta["n_chunks"],
                    hits=hits,
                    total=total,
                    misses=misses,
                )
            )
        else:
            meta = load_index_meta(paths["meta"])
            results.append(StrategyEvalResult(strategy=strategy, n_chunks=meta["n_chunks"]))

    print_summary_table(results, args.top_k)
    print_miss_report(results)


if __name__ == "__main__":
    main()
