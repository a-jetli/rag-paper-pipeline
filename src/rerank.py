from flashrank import Ranker, RerankRequest

RELEVANCE_THRESHOLD = 0.5
RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    """Lazily load the FlashRank cross-encoder once and reuse it."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=RERANKER_MODEL)
    return _ranker


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank chunks with a local FlashRank cross-encoder.

    Args:
        query: The search query
        chunks: List of chunk dicts, each with at least 'chunk_text' key
        top_n: Maximum number of top results to return (default 5)

    Returns:
        Up to top_n chunks with an added 'relevance_score' key, sorted by
        relevance_score descending and filtered by RELEVANCE_THRESHOLD. If no
        chunk clears the threshold, the single best chunk is returned so the
        result is never empty.
    """
    if not chunks:
        return []

    ranker = _get_ranker()

    passages = [
        {"id": i, "text": chunk["chunk_text"]}
        for i, chunk in enumerate(chunks)
    ]

    results = ranker.rerank(RerankRequest(query=query, passages=passages))

    reranked_chunks = []
    for item in results:
        chunk = chunks[item["id"]].copy()
        chunk["relevance_score"] = item["score"]
        reranked_chunks.append(chunk)

    reranked_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

    top = reranked_chunks[:top_n]
    filtered = [c for c in top if c["relevance_score"] >= RELEVANCE_THRESHOLD]

    if not filtered:
        return [reranked_chunks[0]]

    return filtered
