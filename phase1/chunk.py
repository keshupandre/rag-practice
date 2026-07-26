

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_FILE_PATH = ROOT/"docs"/"rag_notes.md"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

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

def parse_arg() -> argparse.Namespace:
    parser= argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type= Path,
        default= DEFAULT_FILE_PATH
    )
    parser.add_argument(
        "--chunk_size",
        type= int,
        default= DEFAULT_CHUNK_SIZE
    )
    parser.add_argument(
        "--overlap",
        type= int,
        default= DEFAULT_OVERLAP
    )

    return parser.parse_args()

def main() -> None:
    args= parse_arg()

    text= load_markdown(args.file_path)

    chunks= chunk_text(text,args.chunk_size,args.overlap)

    print(f"Total Chunks : {len(chunks)}")
    print(f"Chunk length : {len(chunks[0])}")
    print(f"sample Chunk : {chunks[0][:120]} ....")


if __name__ == "__main__":
    main()