
import argparse
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table

import numpy as np
import voyageai

from phase0.config import get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "phase0" / "sentences.txt"

console = Console()

def load_sentences(path: Path) -> list[str]:
    sentences: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                sentences.append(sentence)

    return sentences

def embed_texts(texts: list[str], client, model: str) -> np.ndarray:
    response = client.embed(texts, model=model)
    embeddings = response.embeddings
    vectors = np.array([item for item in embeddings], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors

def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T

from typing import TypedDict


class RankedChunk(TypedDict):
    rank: int
    score: float
    chunk: dict


def rank_chunks(
    chunks: list[dict],
    query_embedding: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> list[RankedChunk]:
    scores = corpus_emb @ query_embedding
    order = np.argsort(scores)[::-1][:k]

    results: list[RankedChunk] = []
    for rank, idx in enumerate(order, start=1):
        i = int(idx)
        results.append(
            {
                "rank": rank,
                "score": float(scores[i]),
                "chunk": chunks[i],
            }
        )
    return results


def print_ranked_chunks(query: str, ranked: list[RankedChunk]) -> None:
    table = Table(title=f'Query: "{query}"')
    table.add_column("Rank", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Text")
    for item in ranked:
        chunk = item["chunk"]
        text = chunk["text"].replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        table.add_row(
            str(item["rank"]),
            str(chunk.get("id", "")),
            f"{item['score']:.3f}",
            str(chunk.get("source", "")),
            text,
        )
    console.print(table)


def query_neighbors(
    client,
    model: str,
    sentences: list[str],
    corpus_emb: np.ndarray,
    query: str,
    k: int,
) -> list[dict]:
    q = embed_text(client, model, [query])[0]
    chunks = [{"id": i, "source": "", "text": s} for i, s in enumerate(sentences)]
    ranked = rank_chunks(chunks, q, corpus_emb, k)

    table = Table(title=f'Query: "{query}"')
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Sentence")
    for item in ranked:
        table.add_row(
            str(item["rank"]),
            f"{item['score']:.3f}",
            item["chunk"]["text"],
        )
    console.print(table)
    return [{"rank": r["rank"], "score": r["score"], "text": r["chunk"]["text"]} for r in ranked]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playground for embeddings")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to a newline-delimited sentences file.",
    )
    parser.add_argument(
        "--query",
        type= str,
        default= "",
        help= "Add your query"
    )
    parser.add_argument(
        "--k",
        type= int,
        default= 3
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    settings = get_settings()
    if settings.voyage_api_key is None:
        raise RuntimeError("Set VOYAGE_API_KEY in the repository .env file.")

    client = voyageai.Client(api_key=settings.voyage_api_key.get_secret_value())

    sentences = load_sentences(args.corpus)[:5]
    embeddings = embed_texts(sentences, client, settings.voyage_embedding_model)

    similarity_matrix = cosine_matrix(embeddings)

    for i, sentence in enumerate(sentences):
        print(f"\n {i} : {sentence}")
        for j, other in enumerate(sentences):
            if i == j:
                continue
            print(f"    vs {j} : {similarity_matrix[i][j]:.3f} {other}")


    if args.query :
        print(f"\n \n query embedding for {args.query} \n")

        query_neighbour(args.query,sentences,client,settings.voyage_embedding_model,args.k,embeddings)




if __name__ == "__main__":
    main()
