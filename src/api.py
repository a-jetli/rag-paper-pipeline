import os
import json
import asyncio
import logging
from contextlib import aclosing, asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.embed_store import get_collection
from src.bm25 import BM25Index, documents_from_collection
from src.agent.graph import build_graph

logger = logging.getLogger(__name__)

agent = None
papers_index = []

BM25_CACHE_PATH = "chroma_db/bm25_index.pickle"
MANIFEST_PATH = "data/corpus_manifest.json"
STREAM_STAGES = frozenset({"planner", "retriever", "grader", "reformulator", "synthesizer"})


def _load_bm25_index(collection) -> BM25Index:
    """
    Load the prebuilt BM25 index, rebuilding in memory only if it can't be used.

    The staleness check is a chunk-count comparison, not a content hash. Hashing
    the corpus would mean reading all ~20k documents out of Chroma on every boot
    (~1s), which is the exact cost the cached index exists to avoid. Counting is
    a metadata lookup (~0.04s).

    Drift is mostly prevented upstream: scripts/build_index.py rebuilds this
    cache itself once it finishes writing Chroma, so the two cannot diverge
    through the normal path. This check only catches a restored backup or an
    interrupted build.
    """
    if os.path.exists(BM25_CACHE_PATH):
        bm25_index = BM25Index.load(BM25_CACHE_PATH)
        is_current, reason = bm25_index.matches_collection(collection)
        if is_current:
            logger.info("Loaded BM25 cache from %s (%s)", BM25_CACHE_PATH, reason)
            return bm25_index
        logger.warning(
            "BM25 cache at %s is stale (%s); rebuilding from Chroma in memory. "
            "Run scripts/build_bm25_cache.py to replace the cache.",
            BM25_CACHE_PATH,
            reason,
        )
    else:
        logger.warning(
            "BM25 cache not found at %s; rebuilding from Chroma in memory. "
            "Run scripts/build_bm25_cache.py to persist it.",
            BM25_CACHE_PATH,
        )

    return BM25Index(documents_from_collection(collection))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent

    collection = get_collection()
    bm25_index = _load_bm25_index(collection)

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
    all_sub_queries: list[str]
    retries: int
    chunks: list[ChunkResponse]
    citations: list[CitationResponse]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # agent.invoke() is synchronous and does the whole request's work — OpenAI
    # calls, Chroma queries, reranking. Calling it directly from an async route
    # would block the event loop, so concurrent requests would queue behind each
    # other. Handing it to a worker thread lets the loop serve others meanwhile.
    result = await asyncio.to_thread(agent.invoke, {
        "original_query": req.query,
        "is_compound": False,
        "sub_queries": [],
        "all_sub_queries": [],
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
        all_sub_queries=result["all_sub_queries"],
        retries=max(0, result["retry_count"] - 1),
        chunks=chunks,
        citations=citations,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    async def events():
        yield _sse_event("progress", {"stage": "started", "timestamp": _timestamp()})

        latest_state = None
        try:
            stream = agent.astream(
                {
                    "original_query": req.query,
                    "is_compound": False,
                    "sub_queries": [],
                    "all_sub_queries": [],
                    "accumulated_context": [],
                    "context_sufficient": False,
                    "missing_elements": [],
                    "retry_count": 0,
                    "final_answer": "",
                    "citations": [],
                },
                stream_mode=["updates", "values"],
            )
            async with aclosing(stream):
                async for mode, update in stream:
                    if await request.is_disconnected():
                        return
                    if mode == "values":
                        latest_state = update
                        continue
                    for stage in update:
                        if stage in STREAM_STAGES:
                            yield _sse_event(
                                "progress",
                                {"stage": stage, "timestamp": _timestamp()},
                            )

            if latest_state is None or await request.is_disconnected():
                return

            chunks = [
                ChunkResponse(
                    chunk_id=chunk.get("chunk_id", ""),
                    title=chunk["title"],
                    authors=chunk["authors"],
                    paper_id=chunk["paper_id"],
                    chunk_text=chunk["chunk_text"],
                    score=chunk.get("relevance_score", chunk.get("rrf_score", 0)),
                )
                for chunk in latest_state["accumulated_context"]
            ]
            citations = [
                CitationResponse(
                    title=item["title"],
                    authors=item["authors"],
                    paper_id=item["paper_id"],
                )
                for item in latest_state["citations"]
            ]
            response = QueryResponse(
                answer=latest_state["final_answer"],
                query_type="compound" if latest_state["is_compound"] else "simple",
                sub_queries=latest_state["sub_queries"],
                all_sub_queries=latest_state["all_sub_queries"],
                retries=max(0, latest_state["retry_count"] - 1),
                chunks=chunks,
                citations=citations,
            )
            yield _sse_event(
                "complete",
                {
                    "stage": "complete",
                    "timestamp": _timestamp(),
                    **jsonable_encoder(response),
                },
            )
        except Exception:
            logger.exception("Streaming query failed")
            if not await request.is_disconnected():
                yield _sse_event(
                    "error",
                    {
                        "stage": "error",
                        "timestamp": _timestamp(),
                        "message": "Query failed before completion.",
                    },
                )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/papers")
async def papers():
    """The full corpus as a browsable list: arXiv id, title, authors, topic, year, and link."""
    return {"count": len(papers_index), "papers": papers_index}
