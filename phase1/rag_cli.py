import argparse
from datetime import UTC, datetime
from pathlib import Path

from google import genai
import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
import voyageai

from phase0.chat.cli import add_token_usage, append_transcript, default_transcript_path, print_history, print_token_table
from phase0.config import get_settings
from phase0.embeddings.playground import embed_texts, rank_chunks
from phase1.ask import rank_chunks_faiss
from phase1.rag_input import build_rag_input
from phase1.retrieve import load_index

console = Console()

ROOT_DIR = Path(__file__).parents[1]

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--rank_method", type=str, default="cosine", choices=["cosine", "faiss"])
    parser.add_argument(
        "--transcript",
        type= Path,
        default= default_transcript_path()
    )
    return parser.parse_args()       


def main() -> None:
    args = parse_args()
    settings = get_settings()

    client = genai.Client(api_key=settings.require_gemini_api_key())
    embedding_client = voyageai.Client(api_key=settings.require_voyage_api_key())
    
    chunks, embeddings = load_index()
    history: list[tuple[str, str]] = []
    previous_interaction_id: str | None = None

    tokens = {
        "Input": 0,
        "Output": 0,
        "Thought": 0,
        "Total": 0,
    }
    console.print(
        Panel(
            "Type a message, `/history`, `/tokens`, `/clear`, or `/quit`.",
            title="Gemini Chat",
            border_style="blue",
        )
    )
    console.print(f"[dim]Transcript: {args.transcript}[/]")

    while True:
        try:
            user_text = console.input("[bold cyan]You > [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye[/]")
            break

        if not user_text:
            continue

        if user_text in {"/quit", "quit", "/exit", "exit", ":q", "/q"}:
            break

        if user_text == "/history":
            print_history(history)
            continue

        if user_text == "/clear":
            history.clear()
            previous_interaction_id = None
            console.print("[yellow]Conversation history cleared.[/]")
            continue

        if user_text in {"/usage", "/tokens"}:
            print_token_table(tokens, "Session token usage")
            continue

        query_embedding = embed_texts([user_text], embedding_client, settings.voyage_embedding_model)[0]
        ranked_chunks: list[dict] = []
        if args.rank_method == "cosine":
            ranked_chunks = rank_chunks(chunks,query_embedding, embeddings, args.k)
        elif args.rank_method == "faiss":
            ranked_chunks = rank_chunks_faiss(chunks, query_embedding, embeddings, args.k)
        else:
            raise ValueError(f"Invalid rank method: {args.rank_method}. Choose from 'cosine' or 'faiss'.")

        context ="\n\n".join(f"[chunk {r['chunk']['id']}] {r['chunk']['text']}" for r in ranked_chunks)
        rag_input = build_rag_input(user_text, context)
        create_kwargs: dict = {
            "model": settings.gemini_model,
            "input": rag_input
        }
        if previous_interaction_id is not None:
            create_kwargs["previous_interaction_id"] = previous_interaction_id

        parent_interaction_id = previous_interaction_id
        interaction = client.interactions.create(**create_kwargs)

        assistant_text = interaction.output_text or ""
        source_ids = [r["chunk"]["id"] for r in ranked_chunks]

        console.print()
        console.print("[bold green]Gemini[/]")
        console.print(escape(assistant_text))
        console.print(f"[dim]Sources: {source_ids}[/]")

        history.append((user_text, assistant_text))
        previous_interaction_id = interaction.id

        turn_usage = add_token_usage(interaction.usage, tokens)
        append_transcript(
            args.transcript,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "chat_turn",
                "model": settings.gemini_model,
                "interaction_id": interaction.id,
                "previous_interaction_id": parent_interaction_id,
                "user": user_text,
                "assistant": assistant_text,
                "token_usage": turn_usage,
            },
        )
        if turn_usage:
            usage_text = " · ".join(f"{label}: {value:,}" for label, value in turn_usage.items())
            console.print(f"[dim]This response — {usage_text}[/]")





if __name__ == "__main__":
    main()