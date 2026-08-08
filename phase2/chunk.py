
import argparse
from pathlib import Path
import fitz

def load_markdown(path:Path)->str:
    with open(path,"r",encoding="utf-8") as f:
        return f.read()

def chunk_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = para if not current else current + "\n\n" + para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) > chunk_size:
            chunks.extend(chunk_fixed(para, chunk_size, overlap))
        else:
            current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_fixed(text:str,chunk_size:int,overlap:int)-> list[str]:
    chunks=[]
    start= 0

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    while start < len(text):
        chunk = text[start:start+chunk_size]
        start += chunk_size-overlap
        chunks.append(chunk.strip())

    return chunks


def chunk_text(text:str, strategy:str, chunk_size:int, overlap:int)-> list[str]:

    if strategy == "paragraph":
        return chunk_paragraphs(text, chunk_size, overlap)
    elif strategy == "fixed":
        return chunk_fixed(text,chunk_size,overlap)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

def collect_chunks(
    files: list[Path],
    pdf_files: list[Path] = [],
    *,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    records: list[dict] = []
    next_id = 0

    for path in files:
        text = load_markdown(path)
        texts = chunk_text(text, strategy, chunk_size, overlap)
        for text_chunk in texts:
            records.append(
                {
                    "id": next_id,
                    "source": path.name,
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "text": text_chunk,
                }
            )
            next_id += 1
        print(f"file: {path.name}, n_chunks: {len(texts)}")
    
    for path in pdf_files:
        doc = fitz.open(path)
        text = ""

        for page_num, page in enumerate(doc):
            text += page.get_text()
        
        texts = chunk_text(text, strategy, chunk_size, overlap)
        for text_chunk in texts:
            records.append({
                "id": next_id,
                "source": path.name,
                "strategy": strategy,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "text": text_chunk,
            })
            next_id += 1
        print(f"file: {path.name}, n_chunks: {len(texts)}")

    return records
