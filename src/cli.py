from src.embed_store import get_collection
from src.bm25 import BM25Index
from src.agent.graph import build_graph
from dotenv import load_dotenv
load_dotenv()


def list_all_papers(collection) -> None:
    results = collection.get()
    papers = {}
    for i, metadata in enumerate(results["metadatas"]):
        paper_id = metadata["paper_id"]
        if paper_id not in papers:
            papers[paper_id] = {
                "title": metadata["title"],
                "authors": metadata["authors"],
                "paper_id": paper_id,
            }

    print(f"\n{'='*80}")
    print(f"ALL {len(papers)} PAPERS IN COLLECTION")
    print(f"{'='*80}\n")
    for i, (paper_id, info) in enumerate(sorted(papers.items()), 1):
        print(f"{i}. [{paper_id}] {info['title']}")
        print(f"   Authors: {info['authors']}\n")


def main():
    collection = get_collection()

    print("Building BM25 index...")
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
    bm25_index = BM25Index(documents)
    print(f"BM25 index built: {len(documents)} chunks indexed.")

    print("Building agent graph...")
    agent = build_graph(collection, bm25_index)
    print("arXiv RAG ready. Type 'papers' to list all papers, or 'quit' to exit.")

    while True:
        user_input = input("Query: ").strip()

        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "papers":
            list_all_papers(collection)
            continue
        if not user_input:
            continue

        initial_state = {
            "original_query": user_input,
            "is_compound": False,
            "sub_queries": [],
            "all_sub_queries": [],
            "accumulated_context": [],
            "context_sufficient": False,
            "missing_elements": [],
            "retry_count": 0,
            "final_answer": "",
            "citations": [],
        }

        result = agent.invoke(initial_state)

        query_type = "compound" if result["is_compound"] else "simple"
        print(f"\nQuery classified as: {query_type}")
        print(f"Sub-queries: {result['sub_queries']}")

        if result["retry_count"] > 1:
            print(f"Retries performed: {result['retry_count'] - 1}")
            print(f"All sub-queries searched across all passes: {result['all_sub_queries']}")
            if result["missing_elements"]:
                print(f"Missing elements identified: {result['missing_elements']}")

        print("\n" + "="*60)
        print("RETRIEVED CHUNKS")
        print("="*60)
        for i, chunk in enumerate(result["accumulated_context"], 1):
            score = chunk.get("relevance_score", chunk.get("rrf_score", 0))
            print(f"\n[Chunk {i}] (score: {score:.3f}) [{chunk['title']}]")
            print(chunk["chunk_text"][:300] + ("..." if len(chunk["chunk_text"]) > 300 else ""))
        print("\n" + "="*60 + "\n")

        print(result["final_answer"])
        print()
        print("Sources:")
        for citation in result["citations"]:
            print(f"- {citation['title']} ({citation['paper_id']}) by {citation['authors']}")
        print()


if __name__ == "__main__":
    main()
