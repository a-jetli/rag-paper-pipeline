"""
Latency benchmark harness for the retrieval layer.

Three independent measurements, each with warmup + repeated trials, reporting
median (primary) and mean:

  1. rerank   - TinyBERT-L-2 vs MiniLM-L-12 on identical cached candidate pools.
                Pure CPU, no API calls. This is the reranker-swap claim.
  2. fanout   - threaded vs sequential per-sub-query retrieval, using precomputed
                query embeddings so the timed section is entirely local
                (Chroma + BM25 + rerank). This is the concurrency claim.
  3. bm25load - loading the pre-serialized BM25 pickle vs rebuilding the index
                from ChromaDB in memory. This is the startup claim.

Candidate pools are built once via the real retrieval path and cached to disk,
so every rerank trial sees byte-identical input and only the model varies.
Building the cache is the only step that calls the OpenAI embedding API
(one batched call for the benchmark queries, a fraction of a cent).

Usage:
    python3 scripts/bench_latency.py                # all three
    python3 scripts/bench_latency.py rerank fanout  # a subset
    python3 scripts/bench_latency.py --trials 15
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from flashrank import Ranker, RerankRequest

from src.bm25 import BM25Index
from src.embed_store import get_collection, embed_texts
from src.retrieve import retrieve, RRF_POOL_SIZE

BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"
POOL_CACHE_PATH = "scripts/.bench_pools.json"

DEPLOYED_MODEL = "ms-marco-TinyBERT-L-2-v2"
HEAVY_MODEL = "ms-marco-MiniLM-L-12-v2"

# Single-topic queries, used for the rerank benchmark (one pool each).
QUERIES = [
    "What is the transformer architecture and how does self-attention work?",
    "How does retrieval augmented generation reduce hallucination?",
    "What is reinforcement learning from human feedback?",
    "How do diffusion models generate images?",
    "What are scaling laws for large language models?",
    "How does LoRA perform parameter efficient fine tuning?",
    "What is contrastive learning for sentence embeddings?",
    "How does chain of thought prompting improve reasoning?",
    "What is mixture of experts routing in large models?",
    "How does batch normalization stabilize training?",
    "What is the vision transformer and how does it patch images?",
    "How does speculative decoding accelerate inference?",
]

# Compound queries decomposed into sub-queries, mirroring what the planner
# emits. Used for the fan-out benchmark, where the whole point is >1 sub-query.
SUB_QUERY_SETS = [
    [
        "transformer self-attention architecture",
        "recurrent neural network sequence modeling limitations",
        "convolutional sequence to sequence models",
    ],
    [
        "retrieval augmented generation grounding",
        "dense passage retrieval for open domain question answering",
        "BM25 lexical baseline retrieval effectiveness",
    ],
    [
        "reinforcement learning from human feedback reward model",
        "direct preference optimization alignment",
    ],
]


def _stats(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _fmt(label: str, s: dict) -> str:
    return (
        f"  {label:<28} median {s['median_s']:7.3f}s   mean {s['mean_s']:7.3f}s   "
        f"min {s['min_s']:7.3f}s   max {s['max_s']:7.3f}s   n={s['n']}"
    )


def build_pool_cache(collection, bm25_index) -> list[dict]:
    """Run real retrieval once per query and cache the RRF candidate pools."""
    print(f"Building candidate pools for {len(QUERIES)} queries (one batched embedding call)...")
    embeddings = embed_texts(QUERIES)
    pools = []
    for query, embedding in zip(QUERIES, embeddings):
        merged = retrieve(query, collection, bm25_index, embedding)
        pools.append({"query": query, "chunks": merged})
        print(f"  {len(merged):>3} candidates  {query[:60]}")
    with open(POOL_CACHE_PATH, "w") as f:
        json.dump(pools, f)
    print(f"Cached to {POOL_CACHE_PATH}")
    return pools


def load_pool_cache(collection, bm25_index) -> list[dict]:
    if os.path.exists(POOL_CACHE_PATH):
        with open(POOL_CACHE_PATH) as f:
            pools = json.load(f)
        print(f"Loaded {len(pools)} cached candidate pools from {POOL_CACHE_PATH}")
        return pools
    return build_pool_cache(collection, bm25_index)


def bench_rerank(pools: list[dict], trials: int) -> dict:
    """Time both cross-encoders over identical candidate pools."""
    print("\n=== 1. RERANK MODEL (identical pools, CPU only, no API) ===")
    sizes = [len(p["chunks"]) for p in pools]
    print(f"Pool sizes: min {min(sizes)}, max {max(sizes)}, cap RRF_POOL_SIZE={RRF_POOL_SIZE}")

    results = {}
    for label, model_name in (("TinyBERT-L-2 (deployed)", DEPLOYED_MODEL), ("MiniLM-L-12 (heavy)", HEAVY_MODEL)):
        print(f"\nLoading {model_name}...")
        load_start = time.perf_counter()
        ranker = Ranker(model_name=model_name)
        load_s = time.perf_counter() - load_start

        requests = [
            RerankRequest(
                query=p["query"],
                passages=[{"id": i, "text": c["chunk_text"]} for i, c in enumerate(p["chunks"])],
            )
            for p in pools
        ]

        ranker.rerank(requests[0])  # warmup: first inference pays one-time setup

        per_pool = []
        for _ in range(trials):
            for req in requests:
                t0 = time.perf_counter()
                ranker.rerank(req)
                per_pool.append(time.perf_counter() - t0)

        s = _stats(per_pool)
        s["model"] = model_name
        s["cold_load_s"] = load_s
        results[label] = s
        print(_fmt(label, s))
        print(f"  {'cold model load':<28} {load_s:7.3f}s (one time, singleton)")

    tiny = results["TinyBERT-L-2 (deployed)"]
    mini = results["MiniLM-L-12 (heavy)"]
    speedup = mini["median_s"] / tiny["median_s"]
    reduction = 100 * (1 - tiny["median_s"] / mini["median_s"])
    print(f"\n  >> TinyBERT is {speedup:.1f}x faster per rerank call "
          f"({reduction:.1f}% lower median latency) than MiniLM-L-12")
    results["speedup_x"] = speedup
    results["reduction_pct"] = reduction
    return results


def bench_fanout(collection, bm25_index, trials: int) -> dict:
    """Threaded vs sequential per-sub-query retrieval, embeddings precomputed."""
    from concurrent.futures import ThreadPoolExecutor
    from src.retrieve import run_full_retrieval

    print("\n=== 2. RETRIEVAL FAN-OUT (threaded vs sequential, embeddings precomputed) ===")

    prepared = []
    for subs in SUB_QUERY_SETS:
        prepared.append((subs, embed_texts(subs)))

    def sequential(subs, embs):
        return [run_full_retrieval(q, collection, bm25_index, e) for q, e in zip(subs, embs)]

    def threaded(subs, embs):
        with ThreadPoolExecutor(max_workers=max(1, len(subs))) as pool:
            return list(pool.map(
                lambda args: run_full_retrieval(args[0], collection, bm25_index, args[1]),
                zip(subs, embs),
            ))

    # Warm the FlashRank singleton and Chroma caches before timing either mode.
    sequential(*prepared[0])

    results = {}
    for label, fn in (("sequential", sequential), ("threaded", threaded)):
        samples = []
        for _ in range(trials):
            for subs, embs in prepared:
                t0 = time.perf_counter()
                fn(subs, embs)
                samples.append(time.perf_counter() - t0)
        s = _stats(samples)
        results[label] = s
        print(_fmt(f"{label} ({len(SUB_QUERY_SETS)} query sets)", s))

    speedup = results["sequential"]["median_s"] / results["threaded"]["median_s"]
    reduction = 100 * (1 - results["threaded"]["median_s"] / results["sequential"]["median_s"])
    print(f"\n  >> Threaded fan-out is {speedup:.2f}x faster "
          f"({reduction:.1f}% lower median latency) on 2-3 sub-queries")
    results["speedup_x"] = speedup
    results["reduction_pct"] = reduction

    # Equivalence check: same chunk ids, same order, both modes.
    mismatches = 0
    for subs, embs in prepared:
        seq_ids = [[c["chunk_id"] for c in r] for r in sequential(subs, embs)]
        thr_ids = [[c["chunk_id"] for c in r] for r in threaded(subs, embs)]
        if seq_ids != thr_ids:
            mismatches += 1
    results["order_mismatches"] = mismatches
    print(f"  >> Result equivalence: {len(prepared) - mismatches}/{len(prepared)} "
          f"query sets returned identical chunk ids in identical order")
    return results


E2E_QUERIES = [
    "How does the transformer architecture differ from recurrent models, and how did retrieval augmented generation build on it?",
    "Compare reinforcement learning from human feedback with direct preference optimization for aligning language models.",
    "What are scaling laws for language models and how does mixture of experts routing change the compute tradeoff?",
]


def bench_e2e(collection, bm25_index, trials: int) -> dict:
    """Full agent invoke, deployed reranker vs heavy reranker.

    Includes planner/grader/synthesizer LLM calls, so absolute numbers carry
    network variance. The comparison is still apples to apples: same queries,
    same graph, same corpus, only RERANKER_MODEL differs.
    """
    import src.rerank as rr
    from src.agent.graph import build_graph

    print("\n=== 5. END TO END AGENT (full pipeline, reranker swapped) ===")
    print(f"{len(E2E_QUERIES)} compound queries x {trials} trials per model")

    agent = build_graph(collection, bm25_index)

    def invoke(q):
        return agent.invoke({
            "original_query": q, "is_compound": False, "sub_queries": [],
            "all_sub_queries": [], "accumulated_context": [], "context_sufficient": False,
            "missing_elements": [], "retry_count": 0, "final_answer": "", "citations": [],
        })

    results = {}
    for label, model_name in (("TinyBERT-L-2 (deployed)", DEPLOYED_MODEL), ("MiniLM-L-12 (heavy)", HEAVY_MODEL)):
        # Swap the module-level singleton so the graph picks up the other model.
        rr._ranker = None
        rr.RERANKER_MODEL = model_name
        rr._get_ranker()  # warm before timing

        invoke(E2E_QUERIES[0])  # warm LLM path / connection pool

        samples = []
        for _ in range(trials):
            for q in E2E_QUERIES:
                t0 = time.perf_counter()
                invoke(q)
                samples.append(time.perf_counter() - t0)
        s = _stats(samples)
        s["model"] = model_name
        results[label] = s
        print(_fmt(label, s))

    rr._ranker = None
    rr.RERANKER_MODEL = DEPLOYED_MODEL

    tiny = results["TinyBERT-L-2 (deployed)"]
    mini = results["MiniLM-L-12 (heavy)"]
    speedup = mini["median_s"] / tiny["median_s"]
    reduction = 100 * (1 - tiny["median_s"] / mini["median_s"])
    print(f"\n  >> Deployed reranker gives {speedup:.2f}x lower end-to-end latency "
          f"({reduction:.1f}% reduction) vs MiniLM-L-12")
    results["speedup_x"] = speedup
    results["reduction_pct"] = reduction
    return results


def bench_embed(trials: int) -> dict:
    """One batched embedding call vs one call per sub-query (network bound)."""
    print("\n=== 4. QUERY EMBEDDING (1 batched call vs N sequential calls) ===")

    batched, per_query = [], []
    for _ in range(trials):
        for subs in SUB_QUERY_SETS:
            t0 = time.perf_counter()
            embed_texts(subs)
            batched.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for q in subs:
                embed_texts([q])
            per_query.append(time.perf_counter() - t0)

    results = {"batched": _stats(batched), "per_query": _stats(per_query)}
    print(_fmt("1 batched call", results["batched"]))
    print(_fmt("N sequential calls", results["per_query"]))

    speedup = results["per_query"]["median_s"] / results["batched"]["median_s"]
    reduction = 100 * (1 - results["batched"]["median_s"] / results["per_query"]["median_s"])
    print(f"\n  >> Batching is {speedup:.2f}x faster "
          f"({reduction:.1f}% lower median latency) on 2-3 sub-queries")
    results["speedup_x"] = speedup
    results["reduction_pct"] = reduction
    return results


def bench_bm25_load(collection, trials: int) -> dict:
    """Pickle load vs in-memory rebuild from ChromaDB."""
    print("\n=== 3. BM25 INDEX STARTUP (pickle load vs in-memory rebuild) ===")

    size_mb = os.path.getsize(BM25_CACHE_PATH) / (1024 * 1024)
    print(f"Pickle: {size_mb:.1f} MB")

    load_samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        BM25Index.load(BM25_CACHE_PATH)
        load_samples.append(time.perf_counter() - t0)

    # The rebuild path in src/api.py: pull every chunk from Chroma, strip the
    # enrichment prefix, tokenize, build BM25Okapi. Timed end to end because
    # that is what the server would otherwise do at startup.
    rebuild_trials = max(1, trials // 3)  # rebuild is slow; fewer trials
    rebuild_samples = []
    for _ in range(rebuild_trials):
        t0 = time.perf_counter()
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
        BM25Index(documents)
        rebuild_samples.append(time.perf_counter() - t0)

    results = {"pickle_load": _stats(load_samples), "rebuild": _stats(rebuild_samples), "pickle_mb": size_mb}
    print(_fmt("pickle load", results["pickle_load"]))
    print(_fmt("rebuild from ChromaDB", results["rebuild"]))

    speedup = results["rebuild"]["median_s"] / results["pickle_load"]["median_s"]
    reduction = 100 * (1 - results["pickle_load"]["median_s"] / results["rebuild"]["median_s"])
    print(f"\n  >> Pickle load is {speedup:.1f}x faster "
          f"({reduction:.1f}% lower) than rebuilding at startup")
    results["speedup_x"] = speedup
    results["reduction_pct"] = reduction
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benches", nargs="*", default=None,
                        choices=["rerank", "fanout", "bm25load", "embed", "e2e"],
                        help="which benchmarks to run (default: all)")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rebuild-pools", action="store_true",
                        help="re-run retrieval to rebuild the cached candidate pools")
    parser.add_argument("--out", default="scripts/.bench_results.json")
    args = parser.parse_args()

    benches = args.benches or ["rerank", "fanout", "embed", "bm25load", "e2e"]

    print(f"Trials: {args.trials}   Benchmarks: {', '.join(benches)}")
    print(f"Python: {sys.version.split()[0]}   CPUs: {os.cpu_count()}")

    collection = get_collection()
    bm25_index = BM25Index.load(BM25_CACHE_PATH)
    print(f"Corpus: {collection.count()} chunks, {len(bm25_index.documents)} BM25 docs")

    out = {"trials": args.trials, "cpu_count": os.cpu_count()}

    if "rerank" in benches:
        if args.rebuild_pools and os.path.exists(POOL_CACHE_PATH):
            os.remove(POOL_CACHE_PATH)
        pools = load_pool_cache(collection, bm25_index)
        out["rerank"] = bench_rerank(pools, args.trials)
    if "fanout" in benches:
        out["fanout"] = bench_fanout(collection, bm25_index, args.trials)
    if "e2e" in benches:
        out["e2e"] = bench_e2e(collection, bm25_index, args.trials)
    if "embed" in benches:
        out["embed"] = bench_embed(args.trials)
    if "bm25load" in benches:
        out["bm25load"] = bench_bm25_load(collection, args.trials)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nRaw results written to {args.out}")


if __name__ == "__main__":
    main()
