import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embed_store import get_collection
from src.bm25 import build_and_save

BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"


def main():
    collection = get_collection()

    print("Building BM25 index from ChromaDB...")
    bm25_index = build_and_save(collection, BM25_CACHE_PATH)

    size_mb = os.path.getsize(BM25_CACHE_PATH) / (1024 * 1024)
    print(f"Saved BM25 cache to {BM25_CACHE_PATH} "
          f"({bm25_index.chunk_count} chunks, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
