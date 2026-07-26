# RAG Basics Cheat Sheet

## RAG (Retrieval-Augmented Generation)
Combines information retrieval with an LLM to generate accurate responses.
Retrieves relevant documents before answering the user's query.
Reduces hallucinations by grounding responses in external knowledge.
Useful for chatbots, document Q&A, and enterprise knowledge bases.
Pipeline: **Query → Retrieve → Generate**.


## Chunking
Splits large documents into smaller chunks for efficient retrieval.
Helps embedding models process documents effectively.
Chunk size affects retrieval accuracy and context quality.
Too small loses context; too large retrieves irrelevant information.
Common chunk size: **200–1000 tokens**.


## Chunk Overlap
Repeats a portion of text between consecutive chunks.
Prevents important information from being split across chunks.
Improves retrieval for content spanning chunk boundaries.
Typical overlap is **10–20%** of the chunk size.
Excessive overlap increases storage and processing cost.


## Grounding
Forces the LLM to answer using retrieved documents.
Improves factual accuracy and reduces hallucinations.
Ensures responses are based on trusted sources.
Essential for domain-specific and enterprise applications.
Often includes citations or references to source documents.


## Embeddings
Convert text into numerical vectors representing semantic meaning.
Similar texts have vectors close to each other.
Enable semantic search instead of keyword matching.
Generated using embedding models.
Stored in vector databases for similarity search.


## Retrieval
Finds the most relevant document chunks for a query.
Uses vector similarity search on embeddings.
Returns Top-K matching chunks as context.
Retrieval quality directly impacts answer quality.
Can use semantic, keyword, or hybrid search.


## Common RAG Failures
Poor chunking can split important context.
Weak embeddings reduce retrieval accuracy.
Wrong or irrelevant chunks may be retrieved.
LLM may hallucinate if grounding is weak.
Outdated knowledge base leads to incorrect answers.
