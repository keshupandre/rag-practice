
import argparse
from pathlib import Path
import sys

import numpy as np
import voyageai

from phase0.config import get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "phase0" / "sentences.txt"


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

def query_neighbour(query: str, sentences: list[str], client, model: str, k: int, corpus_embeddings: np.ndarray) -> None:
    query_embedding= embed_texts([query],client,model)[0]
    scores= corpus_embeddings @ query_embedding
    order= np.argsort(scores)[::-1][:k]

    for rank, idx in enumerate(order):
        print(f"Rank {rank+1}: {sentences[idx]} (score: {scores[idx]:.3f})")


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
