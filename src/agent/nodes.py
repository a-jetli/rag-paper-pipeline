from openai import OpenAI
from src.agent.state import AgentState, QueryPlan, InformationCheck
from src.generate import (
    SYSTEM_PROMPT,
    GENERATION_MODEL,
    SYNTHESIZER_MODEL,
    STRUCTURED_EFFORT,
    SYNTHESIS_EFFORT,
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

PLANNER_SYSTEM = (
    "You are a query planning assistant for a research paper search system. "
    "Analyze the user's question and determine if it requires information from "
    "multiple distinct papers or topics (compound) or can be answered from a single "
    "source (simple). For compound queries, decompose into 1-3 standalone search "
    "queries, each targeting one specific aspect. For simple queries, sub_queries "
    "should contain exactly one entry: the original question rewritten to be "
    "keyword-rich and specific for document retrieval. sub_queries must never be "
    "empty. Do not use pronouns or references to other sub-queries — each must "
    "stand alone."
)

GRADER_SYSTEM = (
    "You are an information sufficiency evaluator. Given a user's question and a set "
    "of retrieved text passages from research papers, decide whether they provide "
    "enough to write a solid, well-grounded answer. Judge reasonably — the passages "
    "do not need to be exhaustive. Mark the context SUFFICIENT if the main aspects of "
    "the question are covered, even if some detail is thin. Only mark it INSUFFICIENT "
    "when a clearly central part of the question is entirely absent (for example, one "
    "of the named methods in a comparison has no coverage at all). When insufficient, "
    "list only the specific missing elements."
)

REFORMULATOR_SYSTEM = (
    "You are a search query reformulator. The original question was not fully answered "
    "because some information is missing. Generate 1-3 new search queries that "
    "specifically target the missing information. Use different keywords and phrasing "
    "than previous attempts. Make queries keyword-rich and specific for document retrieval."
)


def planner_node(state: AgentState) -> dict:
    response = _get_client().beta.chat.completions.parse(
        model=GENERATION_MODEL,
        reasoning_effort=STRUCTURED_EFFORT,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": state["original_query"]},
        ],
        response_format=QueryPlan,
    )
    plan: QueryPlan = response.choices[0].message.parsed
    # The prompt says sub_queries must never be empty, but Pydantic only checks
    # the type — list[str] accepts []. An empty list would retrieve nothing and
    # fail silently, so fall back to searching the original question as typed.
    sub_queries = plan.sub_queries or [state["original_query"]]
    return {
        "is_compound": plan.is_compound,
        "sub_queries": sub_queries,
        "all_sub_queries": sub_queries,
    }


def make_retriever_node(collection, bm25_index):
    def retriever_node(state: AgentState) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        from src.embed_store import embed_texts
        from src.retrieve import run_full_retrieval

        sub_queries = state["sub_queries"]

        # One batched embedding call for all sub-queries instead of one call each.
        embeddings = embed_texts(sub_queries)

        # Fetch all sub-queries concurrently. Each worker only reads shared
        # resources (collection, bm25_index, the FlashRank singleton) and
        # returns its own local list — no shared state is written inside a
        # worker, so results are merged/deduped here, single-threaded, after
        # every future has completed.
        with ThreadPoolExecutor(max_workers=max(1, len(sub_queries))) as pool:
            per_query_results = list(pool.map(
                lambda args: run_full_retrieval(args[0], collection, bm25_index, args[1]),
                zip(sub_queries, embeddings),
            ))

        existing_ids = {c["chunk_id"] for c in state["accumulated_context"]}

        # Merge this pass's per-sub-query results. A chunk surfaced by several
        # sub-queries is kept once at its *highest* relevance score; the previous
        # first-seen rule discarded a stronger later score for no reason other
        # than sub-query ordering. Sorting by that score then hands the
        # synthesizer its context in relevance order rather than planner order.
        #
        # Caveat: these scores come from separate reranker calls against
        # different sub-queries, so they are only approximately comparable.
        # That still beats the arbitrary planner ordering it replaces, and the
        # max-by-id merge is correct regardless of ordering.
        best_by_id: dict[str, dict] = {}
        for results in per_query_results:
            for chunk in results:
                if chunk["chunk_id"] in existing_ids:
                    continue
                incumbent = best_by_id.get(chunk["chunk_id"])
                if incumbent is None or chunk.get("relevance_score", 0.0) > incumbent.get("relevance_score", 0.0):
                    best_by_id[chunk["chunk_id"]] = chunk

        new_chunks = sorted(
            best_by_id.values(),
            key=lambda c: c.get("relevance_score", 0.0),
            reverse=True,
        )
        return {"accumulated_context": new_chunks}

    return retriever_node


def grader_node(state: AgentState) -> dict:
    papers: dict[str, list[str]] = {}
    for chunk in state["accumulated_context"]:
        pid = chunk["paper_id"]
        title = chunk["title"]
        key = f"{title} [{pid}]"
        papers.setdefault(key, []).append(chunk["chunk_text"][:200])

    summary_lines = []
    for paper_key, snippets in papers.items():
        combined = " ".join(snippets)[:400]
        summary_lines.append(f"- {paper_key}: {combined}")

    context_summary = "\n".join(summary_lines) if summary_lines else "(no context retrieved)"

    user_msg = (
        f"Question: {state['original_query']}\n\n"
        f"Available context from retrieved papers:\n{context_summary}"
    )

    response = _get_client().beta.chat.completions.parse(
        model=GENERATION_MODEL,
        reasoning_effort=STRUCTURED_EFFORT,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=InformationCheck,
    )
    check: InformationCheck = response.choices[0].message.parsed
    return {
        "context_sufficient": check.has_sufficient_info,
        "missing_elements": check.missing_elements,
        "retry_count": state["retry_count"] + 1,
    }


def reformulator_node(state: AgentState) -> dict:
    missing_str = "\n".join(f"- {m}" for m in state["missing_elements"])
    user_msg = (
        f"Original question: {state['original_query']}\n\n"
        f"Missing information:\n{missing_str}\n\n"
        "Generate new search queries to find this missing information."
    )
    response = _get_client().beta.chat.completions.parse(
        model=GENERATION_MODEL,
        reasoning_effort=STRUCTURED_EFFORT,
        messages=[
            {"role": "system", "content": REFORMULATOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=QueryPlan,
    )
    plan: QueryPlan = response.choices[0].message.parsed
    # Same empty-list guard as the planner. Here the sensible fallback is the
    # gaps the grader named, since finding those is the whole point of the retry.
    sub_queries = plan.sub_queries or state["missing_elements"] or [state["original_query"]]
    return {"sub_queries": sub_queries, "all_sub_queries": sub_queries}


def synthesizer_node(state: AgentState) -> dict:
    context_parts = []
    citations = {}

    for chunk in state["accumulated_context"]:
        context_parts.append(
            f"[Paper: {chunk['title']} | Authors: {chunk['authors']}]\n{chunk['chunk_text']}"
        )
        if chunk["paper_id"] not in citations:
            citations[chunk["paper_id"]] = {
                "title": chunk["title"],
                "authors": chunk["authors"],
                "paper_id": chunk["paper_id"],
            }

    context = "\n---\n".join(context_parts)
    user_prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {state['original_query']}"

    response = _get_client().chat.completions.create(
        model=SYNTHESIZER_MODEL,
        reasoning_effort=SYNTHESIS_EFFORT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content
    return {"final_answer": answer, "citations": list(citations.values())}
