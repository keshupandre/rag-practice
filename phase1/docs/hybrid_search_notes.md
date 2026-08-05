# Hybrid Search & Metadata (Phase 2 Test Doc)

## Hybrid search
Combines dense vector retrieval with sparse keyword search (for example BM25).
Dense search captures paraphrases and semantic similarity.
Sparse search excels at exact terms, SKUs, and rare entity names.
A common pattern is to retrieve candidates from both paths, then fuse scores with RRF or weighted sums.
Hybrid search often beats pure embedding retrieval on mixed corpora.


## Metadata filters
Attach fields such as `source`, `doc_version`, `tenant_id`, or `updated_at` to each chunk.
Filters run before or after vector search depending on the database.
Pre-filtering shrinks the search space for multi-tenant apps.
Post-filtering is simpler but can return fewer than k results if many hits fail the filter.
Always index metadata you plan to filter on.


## Namespaces and re-embedding
Namespaces isolate indices per customer, environment, or experiment.
Changing the embedding model usually requires a full re-embed of all chunks.
Version your index (`embed_model`, `chunk_size`, `chunk_strategy`) in config or chunk metadata.
Keep a migration plan: dual-write, backfill, cutover, then retire the old index.


## Evaluation hooks
Maintain a small golden set of questions with expected source files or chunk ids.
Track recall@k and whether the correct `source` field appears in top results.
Log retrieval latency and embedding cost per query when you add hybrid or reranking.
Use this document alongside `rag_notes.md` to verify multi-file indexing and source attribution.
