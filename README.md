# RAG Engineer Roadmap

A project-led path from LLM fundamentals to a production-ready, agentic RAG application.


## How to use this roadmap

Work through the phases in order. For each one, study the listed concepts, build the projects, and record what you measured. Build a pipeline from scratch at least once before relying on a framework.

Suggested weekly split:

| Learn | Build | Measure |
| --- | --- | --- |
| 30% — documentation and one paper or essay | 50% — ship the phase projects | 20% — golden questions, latency, and cost per query |

## Roadmap

### Phase 0 — Foundations

**Goal:** Python fluency, LLM APIs, and an intuition for how embeddings represent meaning.

Learn:

- Python: typing, `asyncio`, `pathlib`, and Pydantic
- HTTP APIs, environment secrets, and Jupyter notebooks
- Tokenization, context windows, and temperature
- Embeddings, cosine similarity, and vector dimensions
- System/user roles and few-shot prompting

Build:

1. **Embedding playground** — embed 50 sentences, compute a cosine-similarity matrix, and visualize nearest neighbours.  
   Stack: Python, NumPy, sentence-transformers or OpenAI.
2. **Chat CLI** — chat with an LLM, track token use, and save JSONL transcripts.  
   Stack: Python, OpenAI/Anthropic SDK, Rich.


### Phase 1 — Core RAG Pipeline (Weeks 3–4)

**Goal:** Build the load → chunk → embed → retrieve → generate loop end to end.

Learn:

- PDF, Markdown, and HTML document loading
- Fixed-size and recursive text splitting
- In-memory vector stores such as FAISS or Chroma
- Top-*k* retrieval and grounded prompting
- Source attribution and citations

Build:

1. **Personal docs Q&A** — ingest a Markdown/PDF folder and return answers with inline citations.  
   Stack: LangChain or LlamaIndex, Chroma, FastAPI.
2. **RAG from scratch** — implement chunking, a FAISS index, and prompt assembly without a RAG framework.  
   Stack: Python, FAISS, tiktoken.


### Phase 2 — Chunking & Vector Databases (Weeks 5–6)

**Goal:** Treat indexing as a product: choose chunking deliberately, use metadata, and operate a durable store.

Learn:

- Semantic, late, and hierarchical chunking
- Metadata filters for source, date, and access-control tags
- Upserts, deletes, namespaces, and re-embedding plans
- Hybrid search with BM25 and dense vectors
- Index versioning

Build:

1. **Chunking bake-off** — compare fixed, recursive, and semantic chunking on one corpus and measure recall@*k*.  
   Stack: Python, Chroma or Qdrant, custom metrics.
2. **Hybrid search service** — expose BM25 + dense retrieval, filters, and tenant namespaces through an API.  
   Stack: Qdrant or Elasticsearch, FastAPI, Docker.


### Phase 3 — Advanced Retrieval (Weeks 7–9)

**Goal:** Improve answer quality with query transformations, rerankers, and multi-step retrieval.

Learn:

- Query rewriting, HyDE, and multi-query retrieval
- Cross-encoder, Cohere, or BGE reranking
- Parent-document and sentence-window retrieval
- Graph RAG and entity linking fundamentals
- Context compression and packing

Build:

1. **Rerank + rewrite pipeline** — compare baseline top-*k* retrieval with rewritten queries plus a cross-encoder reranker on a fixed evaluation set.  
   Stack: LangChain/LlamaIndex, BGE reranker, Qdrant.
2. **Parent-document retriever** — retrieve small chunks but return their parent sections; compare groundedness with naïve RAG.  
   Stack: LlamaIndex and a vector database.


### Phase 4 — Evaluation & Production (Weeks 10–12)

**Goal:** Ship reliable RAG with metrics, observability, latency and cost controls, and security safeguards.

Learn:

- RAGAS, DeepEval, and custom golden sets
- Faithfulness, answer relevancy, and context precision
- Tracing with LangSmith, Phoenix, or OpenTelemetry
- Caching, batching, rate limits, and cost caps
- PII redaction and prompt-injection defences

Build:

1. **Evaluation harness** — create a 50+ question golden set, run RAGAS in CI, and fail pull requests when faithfulness drops.  
   Stack: RAGAS, pytest, GitHub Actions.
2. **Production RAG API** — add authentication, rate limits, traces, structured logs, health checks, and Docker Compose deployment.  
   Stack: FastAPI, Redis, Phoenix or LangSmith, Docker.


### Phase 5 — Agentic & Multimodal RAG (Weeks 13–16)

**Goal:** Go beyond single-shot text RAG with tools, agents, and non-text modalities.

Learn:

- Tool-calling agents that decide whether to retrieve
- Multi-hop, corrective, and Self-RAG patterns
- Multimodal embeddings for tables, images, and audio
- Long-context versus RAG trade-offs
- Domain-specific systems for code, legal, or support use cases

Build:

1. **Corrective RAG agent** — grade retrieval quality, rewrite weak queries, and optionally web-search for missing context.  
   Stack: LangGraph, Tavily/SerpAPI, vector database.
2. **Capstone: domain RAG product** — build an ingest UI, cited chat, evaluation dashboard, authentication, deployment, and evaluation suite for a domain you care about.  
   Stack: your chosen stack.


## Milestones

|  Evidence of progress |
| --- |
| Embed text and chat through an API |
| A cited RAG app works over local documents |
| Hybrid search runs on a persistent vector database |
| Rewrite + rerank measurably beats the baseline |
| Evaluation runs in CI and the production API is traced |
| A domain-focused agentic RAG product is shipped |

## Core stack map

Pick one tool from each layer early, then add alternatives only when a concrete need arises.

| Layer | Options |
| --- | --- |
| Orchestration | LangChain, LlamaIndex, Haystack, LangGraph |
| Vector stores | Chroma, FAISS, Qdrant, Weaviate, Pinecone, pgvector, Elasticsearch |
| Embeddings | OpenAI, Cohere, Voyage AI, BGE, E5, Nomic |
| Rerankers | Cohere Rerank, BGE reranker, cross-encoders |
| Evaluation | RAGAS, DeepEval, TruLens, custom golden sets |
| Observability | LangSmith, Arize Phoenix, OpenTelemetry, Helicone |
| Serving | FastAPI, Docker, Redis, Celery/RQ for ingestion jobs |
| Optional frontend | Streamlit, Gradio, Next.js chat UI |

## Portfolio checklist

For each project, publish:

- A concise problem statement
- A simple architecture diagram
- Before/after evaluation metrics
- A short demo video

The strongest RAG portfolio shows measured improvements rather than only a collection of tool logos.
