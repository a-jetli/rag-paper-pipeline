"""
Phase 2: Ingestion.

Reads data/corpus_manifest.json (produced by build_corpus_manifest.py), downloads
each paper's PDF, then extracts -> chunks -> embeds -> stores into ChromaDB.

Run order for a full corpus rebuild:
    1. python scripts/build_corpus_manifest.py   (Phase 1, selection -> manifest)
    2. python scripts/build_index.py             (this file: download + embed)
    3. python scripts/build_bm25_cache.py        (rebuild the keyword index)

Idempotent: papers already in ChromaDB are skipped, so a interrupted run can be
re-run and it resumes where it left off.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.ingest import download_pdfs
from src.parse import extract_text
from src.chunk import chunk_text, extract_abstract, exclude_garbage_sections
from src.bm25 import build_and_save
from src.embed_store import embed_texts, store_chunks, get_collection

MANIFEST_PATH = "data/corpus_manifest.json"
PDF_DIR = "data/pdfs"
BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"
# text-embedding-3-small pricing, used only for a rough live cost readout.
EMBED_USD_PER_MTOKEN = 0.02


def paper_exists_in_db(paper_id: str, collection) -> bool:
    """Check if a paper is already indexed in ChromaDB."""
    try:
        results = collection.get(where={"paper_id": {"$eq": paper_id}}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def load_manifest() -> list[dict]:
    """Load the Phase-1 corpus manifest (the list of papers to ingest)."""
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def main():
    load_dotenv()

    papers = load_manifest()
    print(f"Loaded {len(papers)} papers from {MANIFEST_PATH}.")
    os.makedirs(PDF_DIR, exist_ok=True)

    print("\n=== Phase A: downloading PDFs ===")
    papers = download_pdfs(papers, PDF_DIR)
    papers_with_pdfs = [p for p in papers if "pdf_path" in p]
    print(f"{len(papers_with_pdfs)}/{len(papers)} PDFs available.")

    print("\n=== Phase B: chunk + embed + store ===")
    collection = get_collection()
    total = len(papers_with_pdfs)
    indexed = skipped = failed = 0
    total_chunks = 0
    est_tokens = 0

    for i, paper in enumerate(papers_with_pdfs, start=1):
        if paper_exists_in_db(paper["paper_id"], collection):
            skipped += 1
        else:
            try:
                raw_text = extract_text(paper["pdf_path"])
                if len(raw_text) < 100:
                    failed += 1
                    continue

                text = exclude_garbage_sections(raw_text)

                extracted_abstract, abstract_end = extract_abstract(text)
                passage_text = text[abstract_end:].lstrip() if extracted_abstract else text
                manifest_abstract = paper.get("abstract", "")
                if not isinstance(manifest_abstract, str):
                    manifest_abstract = ""
                abstract = extracted_abstract or manifest_abstract.strip()

                chunks = chunk_text(passage_text)
                if not chunks:
                    failed += 1
                    continue

                enriched_chunks = [
                    f"Paper Title: {paper['title']}\n"
                    f"Abstract Summary: {manifest_abstract}\n"
                    f"Content Passage:\n{chunk_body}"
                    for chunk_body in chunks
                ]

                # Embed everything BEFORE storing anything, so a rate-limit failure
                # mid-paper leaves no partially-indexed paper behind (clean retry).
                abstract_embedding = embed_texts([abstract]) if abstract else None
                embeddings = embed_texts(enriched_chunks)

                if abstract:
                    store_chunks([abstract], abstract_embedding, paper, collection, chunk_type="abstract")
                    est_tokens += len(abstract) // 4
                store_chunks(enriched_chunks, embeddings, paper, collection, chunk_type="passage")

                est_tokens += sum(len(c) // 4 for c in enriched_chunks)
                total_chunks += len(chunks)
                indexed += 1
            except Exception as e:
                failed += 1
                print(f"  [embed] FAILED {paper['paper_id']}: {e}")

        if i % 10 == 0 or i == total:
            est_cost = est_tokens / 1_000_000 * EMBED_USD_PER_MTOKEN
            print(f"  [embed] {i}/{total}  indexed={indexed} skipped={skipped} "
                  f"failed={failed} chunks={total_chunks} ~${est_cost:.2f}")

    est_cost = est_tokens / 1_000_000 * EMBED_USD_PER_MTOKEN
    print(f"\nDone. {indexed} new papers indexed ({total_chunks} chunks), "
          f"{skipped} already present, {failed} skipped/failed.")
    print(f"Estimated embedding spend this run: ~${est_cost:.2f}")

    # Rebuild the keyword index here rather than leaving it as a second command.
    # BM25 is derived from Chroma, so skipping this step leaves the two describing
    # different corpora and retrieval silently degrades with no error anywhere.
    if indexed:
        print("\nRebuilding the BM25 keyword index from Chroma...")
        bm25_index = build_and_save(collection, BM25_CACHE_PATH)
        print(f"Saved {BM25_CACHE_PATH} ({bm25_index.chunk_count} chunks)")
    else:
        print("\nNo new papers indexed; BM25 cache left as is.")


if __name__ == "__main__":
    main()
