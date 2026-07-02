"""
Anchor-recall eval: for every hand-verified "anchor"/"curated" paper in the
corpus manifest, build a simple canonical query from its title and check
whether retrieval actually surfaces that paper for its own query.

No LLM calls, no new dependencies — this only exercises the retrieval +
rerank layer (not the full planner/grader agent), so it's fast and cheap
to rerun as a regression check after any retrieval/rerank change.

Usage: python3 scripts/eval_anchor_recall.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.embed_store import get_collection, embed_texts
from src.bm25 import BM25Index
from src.retrieve import run_full_retrieval
from src.rerank import TRUSTED_TIERS

MANIFEST_PATH = "data/corpus_manifest.json"
BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    gold_papers = [p for p in manifest if p.get("tier") in TRUSTED_TIERS]
    print(f"Evaluating {len(gold_papers)} trusted-tier papers ({TRUSTED_TIERS})...")

    collection = get_collection()
    bm25_index = BM25Index.load(BM25_CACHE_PATH)

    queries = [f"What is {p['title']} and how does it work?" for p in gold_papers]
    embeddings = embed_texts(queries)

    misses = []
    hits_by_tier = {}
    total_by_tier = {}

    for paper, query, embedding in zip(gold_papers, queries, embeddings):
        tier = paper["tier"]
        total_by_tier[tier] = total_by_tier.get(tier, 0) + 1

        results = run_full_retrieval(query, collection, bm25_index, embedding)
        found = any(c["paper_id"] == paper["paper_id"] for c in results)

        if found:
            hits_by_tier[tier] = hits_by_tier.get(tier, 0) + 1
        else:
            misses.append(paper)

    total = len(gold_papers)
    hits = total - len(misses)

    print(f"\nOverall recall: {hits}/{total} ({100*hits/total:.1f}%)")
    for tier in sorted(total_by_tier):
        h = hits_by_tier.get(tier, 0)
        t = total_by_tier[tier]
        print(f"  {tier}: {h}/{t} ({100*h/t:.1f}%)")

    if misses:
        print(f"\n{len(misses)} papers not retrieved for their own canonical query:")
        for p in misses:
            print(f"  [{p['tier']}] {p['paper_id']}  {p['title']}")


if __name__ == "__main__":
    main()
