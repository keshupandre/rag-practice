# Phase 0 — Foundations

This pahse of the RAG Engineer Roadmap focus on Python fluency, LLM APIs, embeddings, and prompt basics. This folder contains two small exercises: an embedding-similarity playground and a Gemini terminal chat client.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add only the keys you plan to use to `.env`. The chat CLI requires:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
```

`config.py` reads the repository-level `.env` and exposes a typed `Settings` object through `get_settings()`.

## Embedding playground

Run the playground from the repository root:

```bash
python -m phase0.embeddings.playground
```

The default corpus is [sentences.txt](sentences.txt). Its first five sentences intentionally form three groups: an embeddings/RAG pair, a cooking pair, and an unrelated Saturn sentence. This makes the cosine-similarity output easy to inspect.

The playground uses `VOYAGE_API_KEY` and `VOYAGE_EMBEDDING_MODEL` from `.env`.

## Gemini chat CLI

Start the chat client with:

```bash
python -m phase0.chat.cli
```

Commands:

| Command | Action |
| --- | --- |
| `/history` | Show messages from this session |
| `/tokens` or `/usage` | Show cumulative token counts |
| `/clear` | Clear local history and reset Gemini conversation context |
| `/quit`, `/exit`, `:q` | Exit the chat |

### Transcripts

Every completed chat turn is appended to a JSONL file. By default, the CLI creates a timestamped file under `phase0/chat/transcripts/`; this folder is intentionally ignored by Git because transcripts may contain private prompts or responses.

Use a specific path when you want to retain a named session:

```bash
python -m phase0.chat.cli --transcript phase0/chat/transcripts/demo.jsonl
```

Each line contains a self-contained record with the timestamp, model, Gemini interaction IDs, user message, assistant response, and per-turn token usage.
