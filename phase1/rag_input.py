

def build_rag_input(query: str, context: str) -> str:
    """Prompt that requires inline [chunk N] citations in the answer."""
    return f"""
Answer using ONLY the reference data below.

Rules:
- Cite evidence inline with chunk ids, e.g. [chunk 2].
- If the answer is not in the reference data, say you do not know.
- Do not use outside knowledge.

[Reference Data]
{context}

[User Query]
{query}
""".strip()