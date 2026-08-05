

import argparse
from pathlib import Path
import fitz 

def load_markdown(path:Path)->str:
    with open(path,"r",encoding="utf-8") as f:
        return f.read()

def chunk_text(text:str,chunk_size:int,overlap:int)-> list[str]:
    chunks=[]
    start= 0

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    while start < len(text):
        chunk = text[start:start+chunk_size]
        start += chunk_size-overlap
        chunks.append(chunk.strip())

    return chunks


def collect_chunks(
    markdown_files: list[Path],
    pdf_files: list[Path] = [],
    *,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    records: list[dict] = []
    next_id = 0

    for path in markdown_files:
        text = load_markdown(path)
        texts = chunk_text(text, chunk_size, overlap)
        for text_chunk in texts:
            records.append(
                {
                    "id": next_id,
                    "source": path.name,
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
        
        texts = chunk_text(text, chunk_size, overlap)
        for text_chunk in texts:
            records.append({
                "id": next_id,
                "source": path.name,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "text": text_chunk,
            })
            next_id += 1
        print(f"file: {path.name}, n_chunks: {len(texts)}")

    return records