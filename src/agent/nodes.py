from openai import OpenAI
from src.agent.state import AgentState, QueryPlan, InformationCheck
from src.generate import SYSTEM_PROMPT, GENERATION_MODEL, SYNTHESIZER_MODEL, TEMPERATURE

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
    "queries, each targeting one specific aspect. Make sub-queries keyword-rich and "
    "specific — they will be used for independent document retrieval. Do not use "
    "pronouns or references to other sub-queries — each must stand alone."
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
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": state["original_query"]},
        ],
        response_format=QueryPlan,
    )
    plan: QueryPlan = response.choices[0].message.parsed
    return {"is_compound": plan.is_compound, "sub_queries": plan.sub_queries}


def make_retriever_node(collection, bm25_index):
    def retriever_node(state: AgentState) -> dict:
        from src.retrieve import run_full_retrieval

        existing_ids = {c["chunk_id"] for c in state["accumulated_context"]}
        new_chunks = []
        for sub_query in state["sub_queries"]:
            results = run_full_retrieval(sub_query, collection, bm25_index)
            for chunk in results:
                if chunk["chunk_id"] not in existing_ids:
                    existing_ids.add(chunk["chunk_id"])
                    new_chunks.append(chunk)
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
        temperature=TEMPERATURE,
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
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": REFORMULATOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=QueryPlan,
    )
    plan: QueryPlan = response.choices[0].message.parsed
    return {"sub_queries": plan.sub_queries}


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
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content
    return {"final_answer": answer, "citations": list(citations.values())}
