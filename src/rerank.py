import json

from flashrank import Ranker, RerankRequest

RELEVANCE_THRESHOLD = 0.5       # bar for chunks with strong upstream (RRF) support
STRICT_RELEVANCE_THRESHOLD = 0.8  # bar for chunks the RRF pool barely surfaced
RRF_TRUST_CUTOFF = 10           # rrf_rank <= this uses RELEVANCE_THRESHOLD, else STRICT
# The fallback never admits a chunk below the base relevance bar — it only
# exists to give a weakly-RRF-supported chunk a shot at the lower (non-strict)
# bar instead of being held to STRICT_RELEVANCE_THRESHOLD.
FALLBACK_MIN_SCORE = RELEVANCE_THRESHOLD
RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"

MANIFEST_PATH = "data/corpus_manifest.json"
TRUSTED_TIERS = {"anchor", "curated"}  # hand-picked, field-defining papers

_ranker: Ranker | None = None
_paper_tiers: dict[str, str] | None = None


def _get_ranker() -> Ranker:
    """Lazily load the FlashRank cross-encoder once and reuse it."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=RERANKER_MODEL)
    return _ranker


def _get_paper_tiers() -> dict[str, str]:
    """Lazily load paper_id -> tier from the corpus manifest once and reuse it."""
    global _paper_tiers
    if _paper_tiers is None:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        _paper_tiers = {p["paper_id"]: p.get("tier", "") for p in manifest}
    return _paper_tiers


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank chunks with a local FlashRank cross-encoder.

    Args:
        query: The search query
        chunks: List of chunk dicts, each with at least 'chunk_text' key,
            ordered by RRF rank (index 0 = strongest upstream support).
        top_n: Maximum number of top results to return (default 5)

    Returns:
        Up to top_n chunks with an added 'relevance_score' key, sorted by
        relevance_score descending. A chunk clears the bar at
        RELEVANCE_THRESHOLD if it had strong upstream support (rrf_rank <=
        RRF_TRUST_CUTOFF), otherwise it needs STRICT_RELEVANCE_THRESHOLD —
        a weakly-supported chunk that the reranker scores unusually high is
        treated with more suspicion than one that also had independent
        semantic/BM25 support. If nothing clears the bar, the single best
        chunk is returned only if it clears FALLBACK_MIN_SCORE; otherwise
        the result is empty rather than forcing through a low-confidence
        citation.

        Chunks from hand-verified "anchor"/"curated" papers (see
        TRUSTED_TIERS) that also had strong upstream support (rrf_rank <=
        RRF_TRUST_CUTOFF) are treated as having already cleared
        RELEVANCE_THRESHOLD, same as any other strongly-supported chunk —
        a raw cross-encoder score has no way to know a paper is the
        canonical source for a concept, and has been observed to bury
        exactly this kind of chunk under a lexical false positive
        elsewhere in the pool. This never lowers a chunk's real score,
        only raises weak ones up to the bar every other strongly-supported
        chunk is already held to.
    """
    if not chunks:
        return []

    ranker = _get_ranker()

    passages = [
        {"id": i, "text": chunk["chunk_text"]}
        for i, chunk in enumerate(chunks)
    ]

    results = ranker.rerank(RerankRequest(query=query, passages=passages))

    tiers = _get_paper_tiers()

    reranked_chunks = []
    for item in results:
        rrf_rank = item["id"] + 1
        chunk = chunks[item["id"]].copy()
        chunk["relevance_score"] = item["score"]
        chunk["rrf_rank"] = rrf_rank
        if rrf_rank <= RRF_TRUST_CUTOFF and tiers.get(chunk["paper_id"]) in TRUSTED_TIERS:
            chunk["relevance_score"] = max(chunk["relevance_score"], RELEVANCE_THRESHOLD)
        reranked_chunks.append(chunk)

    reranked_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

    top = reranked_chunks[:top_n]
    filtered = [
        c for c in top
        if c["relevance_score"] >= (
            RELEVANCE_THRESHOLD if c["rrf_rank"] <= RRF_TRUST_CUTOFF else STRICT_RELEVANCE_THRESHOLD
        )
    ]

    if not filtered:
        if reranked_chunks[0]["relevance_score"] >= FALLBACK_MIN_SCORE:
            return [reranked_chunks[0]]
        return []

    return filtered
