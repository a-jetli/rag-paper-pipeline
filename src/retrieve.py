import chromadb
from src.embed_store import embed_texts
from src.bm25 import BM25Index
from src.rerank import rerank

TOP_K = 14          # semantic passage results
ABSTRACT_K = 6      # semantic abstract results
BM25_K = 20         # BM25 results (retrieve more, RRF will filter)
RRF_K = 60          # RRF constant
RRF_POOL_SIZE = 24  # candidates kept after RRF merge, before reranking


def _parse_results(results: dict, chunk_type: str) -> list[dict]:
    """
    Parse a ChromaDB query() response into a list of dicts.

    ChromaDB query() returns a dict with keys:
        - 'ids': list[list[str]]
        - 'documents': list[list[str]]
        - 'metadatas': list[list[dict]]
        - 'distances': list[list[float]]

    Each is wrapped in an outer list (for multiple queries), so index [0] for single query.
    """
    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, chunk_text, metadata, distance in zip(ids, documents, metadatas, distances):
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "paper_id": metadata["paper_id"],
            "title": metadata["title"],
            "authors": metadata["authors"],
            "distance": distance,
            "chunk_index": metadata["chunk_index"],
            "chunk_type": chunk_type,
        })

    return chunks


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict]
) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    Args:
        semantic_results: list of dicts from semantic search,
            ordered by distance ascending (best first)
        bm25_results: list of dicts from BM25 search,
            ordered by bm25_score descending (best first)

    Returns: list of dicts sorted by rrf_score descending,
    limited to top RRF_POOL_SIZE results.
    """
    rrf_scores = {}

    for rank, result in enumerate(semantic_results, 1):
        chunk_id = result["chunk_id"]
        contribution = 1 / (RRF_K + rank)
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {"rrf_score": 0, "data": result}
        rrf_scores[chunk_id]["rrf_score"] += contribution

    for rank, result in enumerate(bm25_results, 1):
        chunk_id = result["chunk_id"]
        contribution = 1 / (RRF_K + rank)
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {"rrf_score": 0, "data": result}
        else:
            # NOTE: this also overwrites "chunk_text" with the BM25 version,
            # which strips the "Paper Title / Abstract Summary" enrichment
            # prefix (see BM25Index — it indexes stripped text so repeated
            # header terms don't inflate IDF). So a chunk matched by both
            # signals gets the shorter, stripped text downstream (reranker
            # and synthesizer both read chunk_text), while a chunk matched
            # by semantic search alone keeps the full enriched text. Verified
            # this doesn't change which chunks get selected in practice, but
            # it does mean the exact text reaching the LLM isn't uniform
            # across chunks for reasons unrelated to relevance.
            rrf_scores[chunk_id]["data"].update(result)
        rrf_scores[chunk_id]["rrf_score"] += contribution

    merged = []
    for chunk_id, item in sorted(rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True):
        result = item["data"].copy()
        result["rrf_score"] = item["rrf_score"]
        merged.append(result)

    return merged[:RRF_POOL_SIZE]


def run_full_retrieval(
    query: str,
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """
    Full Level 3 pipeline in one call: embed → semantic search → BM25 → RRF → rerank.

    Returns reranked chunks filtered by relevance threshold (top 5 max).

    query_embedding: precomputed embedding for this query. Pass this when the
    caller has already batch-embedded multiple sub-queries in one API call,
    to avoid a redundant per-query embedding round trip.
    """
    merged = retrieve(query, collection, bm25_index, query_embedding)
    return rerank(query, merged, top_n=8)


def retrieve(
    query: str,
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: semantic search + BM25, merged with RRF.

    Steps:
    1. Embed the query (or use precomputed query_embedding, if provided)
    2. Run semantic search on abstracts (top ABSTRACT_K)
    3. Run semantic search on passages (top TOP_K)
    4. Combine semantic abstract + passage results into one list
    5. Run BM25 search (top BM25_K)
    6. Merge semantic and BM25 results using reciprocal_rank_fusion()
    7. Return the merged results

    The function signature now requires a bm25_index parameter.
    """
    if query_embedding is None:
        query_embedding = embed_texts([query])[0]

    # Query 1: abstract chunks only
    abstract_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=ABSTRACT_K,
        where={"chunk_type": "abstract"},
    )
    abstract_chunks = _parse_results(abstract_results, chunk_type="abstract")

    # Query 2: passage chunks only
    passage_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where={"chunk_type": "passage"},
    )
    passage_chunks = _parse_results(passage_results, chunk_type="passage")

    # Combine semantic results
    semantic_results = abstract_chunks + passage_chunks
    semantic_results.sort(key=lambda chunk: chunk["distance"])

    # BM25 search
    bm25_results = bm25_index.query(query, n_results=BM25_K)

    # Merge with RRF
    merged = reciprocal_rank_fusion(semantic_results, bm25_results)

    return merged
