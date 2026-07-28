
import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FILE_PATH = ROOT_DIR / "phase2" / "docs" / "rag_notes.md"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

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


def chunk_text(text:str,strategy:str, chunk_size:int, overlap:int)-> list[str]:

    if strategy == "paragraph":
        return chunk_paragraphs(text, chunk_size, overlap)
    elif strategy == "fixed":
        return chunk_fixed(text,chunk_size,overlap)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=Path, default=DEFAULT_FILE_PATH)
    parser.add_argument("--strategy", default="paragraph", choices=["paragraph", "fixed"])
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    return parser.parse_args()

def main()-> None:
    args = parse_args()
    
    text = load_markdown(args.file_path)
    chunks = chunk_text(text, args.strategy, args.chunk_size, args.overlap)

    print(f"n_chunks: {len(chunks)}")




if __name__ == "__main__":
    main()