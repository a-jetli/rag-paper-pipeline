from openai import OpenAI

GENERATION_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

SYSTEM_PROMPT = """You are a research assistant that answers questions about AI and machine learning papers.
Answer using ONLY the provided context below — do not use knowledge from your training data.

Do your best to address the question from what the context provides. If the context covers the
topic even partially, synthesize the most complete answer you can and briefly note any aspect the
context does not cover. Only respond with "I don't have enough information to answer that based on
the available papers." when the context is essentially unrelated to the question."""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Build prompt from retrieved chunks and generate an answer.

    Uses GPT-4o-mini, temperature=0 for deterministic output.

    Args:
        query: the user's question
        retrieved_chunks: list of dicts from retrieve()

    Returns: tuple of (answer_text, deduplicated_citations)
    """
    context_parts = []
    citations = {}

    for chunk in retrieved_chunks:
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

    user_prompt = f"""Context:
---
{context}
---

Question: {query}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content

    citation_list = list(citations.values())

    return answer, citation_list
