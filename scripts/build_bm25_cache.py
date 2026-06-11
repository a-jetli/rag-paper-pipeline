import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embed_store import get_collection
from src.bm25 import BM25Index

BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"


def main():
    collection = get_collection()

    print("Loading chunks from ChromaDB...")
    all_data = collection.get(include=["documents", "metadatas"])
    documents = []
    for i, (doc_text, metadata) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
        bm25_text = doc_text
        if metadata["chunk_type"] == "passage":
            marker = "Content Passage:\n"
            marker_pos = doc_text.find(marker)
            if marker_pos != -1:
                bm25_text = doc_text[marker_pos + len(marker):]
        documents.append({
            "chunk_text": bm25_text,
            "chunk_id": all_data["ids"][i],
            "paper_id": metadata["paper_id"],
            "title": metadata["title"],
            "authors": metadata["authors"],
            "chunk_index": metadata["chunk_index"],
            "chunk_type": metadata["chunk_type"],
        })

    print(f"Building BM25 index from {len(documents)} chunks...")
    bm25_index = BM25Index(documents)

    bm25_index.save(BM25_CACHE_PATH)
    size_mb = os.path.getsize(BM25_CACHE_PATH) / (1024 * 1024)
    print(f"Saved BM25 cache to {BM25_CACHE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
