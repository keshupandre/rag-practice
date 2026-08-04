# Phase 1 — Core RAG Pipeline (from scratch)

Build the classic **load → chunk → embed → retrieve → generate** loop without LangChain or LlamaIndex. This phase uses **Voyage** for embeddings and **Gemini** for generation.

## What you built

| Piece | Module | Purpose |
| --- | --- | --- |
| Chunking | `chunk.py` | Split markdown into overlapping character windows |
| Indexing | `index.py` | Embed chunks and save `chunks.jsonl` + `embeddings.npy` |
| Retrieval | `retrieve.py` | Rank chunks by cosine similarity to a query |
| One-shot Q&A | `ask.py` | Retrieve context → Gemini answer + source ids |
| Interactive RAG | `rag_cli.py` | Chat loop with retrieval each turn, tokens, JSONL transcript |

Shared retrieval helpers live in `phase0/embeddings/playground.py`:

- `embed_texts()` — batch embed + L2 normalize
- `rank_chunks()` — top-k by dot product (cosine on normalized vectors)
- `print_ranked_chunks()` — Rich table for debugging retrieval

## Prerequisites

1. Complete **Phase 0** (embeddings intuition + chat basics).
2. Python 3.11+ and project venv from the repo root:

   ```bash
   cd /path/to/RAGs
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Environment variables in `.env` (see `.env.example`):

   | Variable | Used for |
   | --- | --- |
   | `VOYAGE_API_KEY` | Embedding chunks and queries |
   | `VOYAGE_EMBEDDING_MODEL` | Default `voyage-3` |
   | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Answers in `ask.py` / `rag_cli.py` |
   | `GEMINI_MODEL` | Default `gemini-2.5-flash` |

## Quick start

From the repository root:

```bash
source .venv/bin/activate

# 1) Build the index (default corpus: phase1/docs/rag_notes.md)
python -m phase1.index

# 2) Debug retrieval only
python -m phase1.retrieve --query "Why use overlap when chunking?" --k 3

# 3) Single question end-to-end
python -m phase1.ask --query "What is RAG?" --k 3

# 4) Interactive cited chat
python -m phase1.rag_cli --k 3
```

## Architecture

```text
  rag_notes.md
       │
       ▼
  chunk_text()          chunk_size=500, overlap=50
       │
       ▼
  embed_texts()         Voyage API
       │
       ├──────────────────────────────┐
       ▼                              ▼
  data/index/chunks.jsonl      data/index/embeddings.npy
  (id, source, text)           (n_chunks × dim)

  User query
       │
       ▼
  embed_texts([query])
       │
       ▼
  rank_chunks()           scores = corpus_emb @ query_vec
       │
       ▼
  Prompt with [chunk id] + text
       │
       ▼
  Gemini interactions API  → answer + Sources: [ids]
```

Each chat turn in `rag_cli.py` **re-retrieves** top-k chunks for the latest user message. That is the usual RAG pattern (fresh context per question).

## Commands reference

### `chunk.py` — chunking only

```bash
python -m phase1.chunk
python -m phase1.chunk --file_path phase1/docs/rag_notes.md --chunk_size 500 --overlap 50
```

Prints chunk count and a sample chunk (no API calls).

### `index.py` — build the vector index

```bash
python -m phase1.index
python -m phase1.index --file_path phase1/docs/rag_notes.md --chunk_size 500 --overlap 50
```

Writes:

- `data/index/chunks.jsonl` — one JSON object per line: `id`, `source`, `text`
- `data/index/embeddings.npy` — float matrix, row `i` aligns with chunk `id` `i`

Re-run indexing whenever you change the corpus or chunk settings.

### `retrieve.py` — retrieval without generation

```bash
python -m phase1.retrieve --query "your question" --k 3
```

Checks that JSONL row count matches `embeddings.npy` rows before searching.

### `ask.py` — one-shot grounded answer

```bash
python -m phase1.ask --query "your question" --k 3
```

Prints the model answer and `Sources: [chunk ids]`.

### `rag_cli.py` — interactive RAG chat

```bash
python -m phase1.rag_cli --k 3
python -m phase1.rag_cli --k 3 --transcript data/transcripts/my-session.jsonl
```

| Input | Action |
| --- | --- |
| Normal message | Retrieve → answer with sources |
| `/history` | Show local turn history |
| `/tokens` or `/usage` | Session token table |
| `/clear` | Clear chat history (Gemini thread id reset) |
| `/quit`, `/exit`, `:q` | End session |

Each turn is appended to the transcript path (default: new file under `data/transcripts/`).

## Data layout

```text
phase1/
  docs/rag_notes.md     # default practice corpus (safe to commit)
  chunk.py
  index.py
  retrieve.py
  ask.py
  rag_cli.py

data/index/             # generated — gitignored by default
  chunks.jsonl
  embeddings.npy
```

**Rule:** row `i` in `embeddings.npy` must always match the chunk with `"id": i` in `chunks.jsonl`.

## Concepts to own before Phase 2

1. **Why normalize embeddings?** Dot product on unit vectors equals cosine similarity.
2. **Why overlap?** Reduces cutting important sentences across chunk boundaries.
3. **Why two files?** Text/metadata in JSONL; numeric vectors in `.npy` for fast math.
4. **Retrieval vs generation** — wrong chunks → wrong answers even with a strong LLM.
5. **Citations** — `Sources` lists which chunk ids were retrieved (not yet enforced in the prompt; you can add “cite [chunk N]” in the template).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Set VOYAGE_API_KEY` | Add key to `.env` |
| `Set GEMINI_API_KEY` | Add key to `.env` |
| `FileNotFoundError` on `chunks.jsonl` | Run `python -m phase1.index` |
| Index mismatch error | Re-run `index.py` (stale or partial index) |
| Answer ignores docs | Try higher `--k`; check retrieval with `retrieve.py` first |
| Plain `quit` does not exit `rag_cli` | Use `/quit` or `/exit` |

## Phase 1 checklist

- [ ] `chunk.py` produces sensible overlap between adjacent chunks
- [ ] `index.py` saves JSONL + `.npy` with matching counts
- [ ] `retrieve.py` ranks relevant sections for hand-written queries
- [ ] `ask.py` answers from notes, not general knowledge only
- [ ] `rag_cli.py` logs tokens and JSONL transcripts

## Next

**Phase 2** — paragraph chunking, metadata, persistent vector DB, hybrid (BM25 + dense) search. See the root [README](../README.md) roadmap.
