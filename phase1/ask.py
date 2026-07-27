import argparse
from pathlib import Path

from google import genai
import numpy as np
import voyageai

from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts, rank_chunks
from phase1.retrieve import load_chunks

ROOT_DIR = Path(__file__).parents[1]
INDEX_DIR = ROOT_DIR / "data" / "index"

CHUNK_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()

def main()-> None:
    args = parse_args()
    query = args.query
    k = args.k
    settings = get_settings()

    client = genai.Client(api_key=settings.require_gemini_api_key())
    embedding_client = voyageai.Client(api_key=settings.require_voyage_api_key())

    chunks = load_chunks(CHUNK_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    query_embedding = embed_texts([query], embedding_client, settings.voyage_embedding_model)[0]
    ranked_chunks = rank_chunks(chunks,query_embedding, embeddings, k)

    context ="\n\n".join(f"[chunk {r['chunk']['id']}] {r['chunk']['text']}" for r in ranked_chunks)

    interaction = client.interactions.create(
        model=settings.gemini_model,
        input=f"""
        Use the provided background reference data to answer the user query accurately.
        If the answer cannot be found in the context, state that clearly.

        [Reference Data]
        {context}

        [User Query]
        {query}
        """
    )
    print("\n\nAnswer:\n")
    print(interaction.output_text)

    source_ids = [r["chunk"]["id"] for r in ranked_chunks]
    print(f"\nSources: {source_ids}")


if __name__ == "__main__":
    main()