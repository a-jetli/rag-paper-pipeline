import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.embed_store import get_collection
from src.bm25 import BM25Index
from src.agent.graph import build_graph

agent = None
papers_index = []

BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"
MANIFEST_PATH = "data/corpus_manifest.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent

    collection = get_collection()

    if os.path.exists(BM25_CACHE_PATH):
        bm25_index = BM25Index.load(BM25_CACHE_PATH)
    else:
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

    agent = build_graph(collection, bm25_index)

    # Build a browsable paper list for the frontend (one entry per paper).
    global papers_index
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        papers_index = sorted(
            (
                {
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "authors": ", ".join(p["authors"]) if isinstance(p["authors"], list) else p.get("authors", ""),
                    "field": p.get("field", ""),
                    "year": (p.get("published") or "")[:4],
                    "arxiv_url": f"https://arxiv.org/abs/{p['paper_id']}",
                }
                for p in manifest
            ),
            key=lambda x: (x["field"], x["title"].lower()),
        )
    except FileNotFoundError:
        papers_index = []

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class ChunkResponse(BaseModel):
    chunk_id: str
    title: str
    authors: str
    paper_id: str
    chunk_text: str
    score: float


class CitationResponse(BaseModel):
    title: str
    authors: str
    paper_id: str


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sub_queries: list[str]
    retries: int
    chunks: list[ChunkResponse]
    citations: list[CitationResponse]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    result = agent.invoke({
        "original_query": req.query,
        "is_compound": False,
        "sub_queries": [],
        "accumulated_context": [],
        "context_sufficient": False,
        "missing_elements": [],
        "retry_count": 0,
        "final_answer": "",
        "citations": [],
    })

    chunks = []
    for chunk in result["accumulated_context"]:
        chunks.append(ChunkResponse(
            chunk_id=chunk.get("chunk_id", ""),
            title=chunk["title"],
            authors=chunk["authors"],
            paper_id=chunk["paper_id"],
            chunk_text=chunk["chunk_text"],
            score=chunk.get("relevance_score", chunk.get("rrf_score", 0)),
        ))

    citations = [
        CitationResponse(title=c["title"], authors=c["authors"], paper_id=c["paper_id"])
        for c in result["citations"]
    ]

    return QueryResponse(
        answer=result["final_answer"],
        query_type="compound" if result["is_compound"] else "simple",
        sub_queries=result["sub_queries"],
        retries=max(0, result["retry_count"] - 1),
        chunks=chunks,
        citations=citations,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/papers")
async def papers():
    """The full corpus as a browsable list: arXiv id, title, authors, topic, year, and link."""
    return {"count": len(papers_index), "papers": papers_index}
