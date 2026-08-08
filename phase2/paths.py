"""Shared Phase 2 index artifact paths."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"
DEFAULT_STRATEGY = "paragraph"
INDEX_BUILD_HINT = "run python -m phase2.index --strategy {strategy}"


def index_paths(strategy: str) -> dict[str, Path]:
    return {
        "chunks": INDEX_DIR / f"chunks_{strategy}.jsonl",
        "meta": INDEX_DIR / f"index_meta_{strategy}.json",
        "embeddings": INDEX_DIR / f"embeddings_{strategy}.npy",
    }


def resolve_chunks_path(
    strategy: str = DEFAULT_STRATEGY,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return override
    return index_paths(strategy)["chunks"]
