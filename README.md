---
title: arXiv RAG Pipeline
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Live demo at [https://a-jetli.github.io/rag-paper-demo-live/](https://a-jetli.github.io/rag-paper-demo-live/)


---
# arXiv RAG Pipeline

This is a retrieval augmented generation system that answers questions about AI and ML research by pulling directly from a curated set of about 1000 arXiv papers. It started as the simplest possible RAG setup (embed, retrieve, generate) and evolved through several iterations, each one built to fix a specific failure mode I observed in the previous version. The final system decomposes complex queries, retrieves across multiple signals, reranks with a cross encoder, grades its own context for sufficiency, and retries with reformulated queries when it decides it hasn't found enough. It's deployed as a REST API on Hugging Face Spaces and also runs as a local CLI.


## Tech Stack

| Layer | What | Why |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | 1536 dim vectors, good quality at low cost |
| Generation | gpt-5.6-luna (synthesis + planning/grading/reformulation) | One model for every generation step: synthesis of the final answer as well as the structured planning, grading, and reformulation steps. The model only accepts the default temperature, so `reasoning_effort` replaces it as the tuning knob: `none` for the structured steps, `low` for synthesis. |
| Vector store | ChromaDB | Persistent local storage, simple API, supports metadata filtering |
| Keyword search | `rank_bm25` (BM25Okapi) | Catches named entities and rare terms that embeddings miss |
| Reranker | FlashRank `ms-marco-TinyBERT-L-2-v2` | Local cross encoder that reads query and chunk together for real relevance scoring, no API or rate limits, fast on CPU |
| Agent framework | LangGraph | Manages the planning/retrieval/grading/retry state machine |
| API | FastAPI + Uvicorn | Async HTTP layer with Pydantic validation, deployed on Hugging Face Spaces |
| PDF parsing | PyMuPDF | Fast, reliable text extraction from arXiv PDFs |


## The Corpus

I didn't want a toy dataset where every query trivially finds its answer. The corpus is about 1000 arXiv papers spanning 20 areas of AI and ML, from retrieval and language models to vision, generative models, reinforcement learning, and interpretability.

It's built to make retrieval genuinely hard. Around 30 percent are papers I hand-picked: the field-defining works in each area (the original Transformer, BERT, ResNet, GANs, CLIP, DPR, ColBERT, and so on), so the known-important papers are guaranteed present. The rest are filled in per topic from arXiv keyword searches, filtered to stay on topic and recent enough to matter. Because many of these papers share heavy vocabulary across overlapping topics, the retrieval system has to actually discriminate rather than just pattern match on surface similarity.

All papers are downloaded as PDFs, parsed into text, cleaned up (reference sections, page numbers, and layout junk get stripped), and split into chunks at sentence boundaries targeting ~500 tokens. That comes out to 24,278 chunks across the corpus (23,278 passages plus one abstract per paper). Each chunk carries up to the last two sentences of the one before it, around 17% overlap, so a fact split across a boundary is retrievable from either side. Abstracts are pulled out and indexed separately so they always show up in retrieval results.


## How It Evolved

The interesting part of this project is the progression. Each stage exists because the previous one failed in a specific, observable way.

### Where It Started: Baseline

Embed the query, find the top 5 chunks by cosine similarity, pass them to the LLM, get an answer. This works for straightforward factual questions about a single topic. It falls apart when terms are ambiguous. A query about "coverage" in RAG evaluation would pull chunks about facility location coverage, conformal prediction coverage, and actual RAG coverage metrics, because the embeddings don't discriminate between different usages of the same word. Compound queries like "compare X and Y" would return chunks entirely dominated by whichever topic the full query string was closest to in embedding space.

### Smarter Chunking

The first fix was about the chunks themselves. The original chunker used a fixed sliding window that would split mid sentence, producing incoherent passages that confused the LLM. I replaced it with a sentence boundary chunker that builds up a buffer and flushes when the next sentence would push past the token limit. It also handles academic text specifically, protecting abbreviations like "et al.," "Fig.," and author initials like "J. Smith" from being treated as sentence endings.

I also started extracting abstracts and indexing them as their own chunk type. Before this, abstracts would either get merged into the first passage chunk or ranked below noise. Now retrieval always includes abstract level context by querying the abstract and passage pools separately.

### Hybrid Retrieval

Smarter chunks helped with coherence, but the vocabulary mismatch problem was still there. Cosine similarity is fundamentally a bag of vectors operation. It doesn't know that "The Coverage Illusion" is a specific paper title rather than a generic phrase about coverage.

BM25 solves this because it weights terms by inverse document frequency. A rare term like "Coverage Illusion" gets a high IDF weight, so BM25 naturally discriminates on proper nouns and coined terms. I added a BM25 index over all chunks and merged its rankings with the semantic results using Reciprocal Rank Fusion. A chunk that scores well on both signals gets boosted; a chunk that only shows up in one signal gets a lower combined score. The fusion produces 25 candidates.

One subtle thing here: passage chunks are stored in ChromaDB with the paper's title and abstract prepended (for better embedding quality), but the BM25 index strips that enrichment prefix before indexing. Without the stripping, every chunk from the same paper would share identical title and abstract terms, which inflates BM25 term frequencies and defeats the whole purpose of IDF discrimination.

### Cross Encoder Reranking

Twenty-five RRF candidates is better than five cosine results, but a lot of them are marginally relevant. Both cosine similarity and BM25 score the query and each chunk independently. Neither one actually reads them together.

A cross encoder reranker does. It takes the query and a candidate chunk as a single input and produces a relevance score with full cross attention between them. I send all 25 RRF candidates through the reranker, which scores each one and keeps up to the top 8. There is a tiered relevance floor: a candidate the fast searches already ranked in the top 10 needs a moderate score, while one they barely surfaced needs a much higher one, on the reasoning that a high score with no upstream support is more likely a keyword coincidence than a discovery. Those few chunks become the tight, genuinely relevant context window.

This started out on a hosted reranker API (Jina) and was later moved to FlashRank running locally, mainly to save on time and cost. The agent reranks once per sub query per retrieval pass, so a single compound query was firing a lot of billed API calls and waiting on network round trips each time. FlashRank runs on CPU with no API key, no per call cost, no network call, and no rate limiting to work around. On the CPU constrained deployment the reranker model itself became the slowest step, so I run the small 2 layer `ms-marco-TinyBERT-L-2-v2` model, which is about 20 times faster per call than the larger 12 layer model. It gives up a little ranking precision, but the synthesizer answers from partial context so the final answers hold up.

### Agentic Query Processing

Even with the improved retrieval pipeline, the system still did everything in a single pass. If you asked "compare the training procedures of BERT, GPT-2, and T5," it would retrieve chunks dominated by whichever paper the combined query string was closest to. And if retrieval missed something, there was no mechanism to notice or recover.

I wrapped the retrieval pipeline in a LangGraph agent with five nodes: a planner, a retriever, a grader, a reformulator, and a synthesizer.

The planner decides whether a query is compound and breaks it into independent sub queries. Each sub query runs through the full retrieval pipeline separately, so "compare BERT and GPT-2" becomes two focused searches that each find the right paper.

After retrieval, the grader evaluates whether the accumulated context is actually sufficient to answer the original question. It gets a compressed summary of what's been found so far and identifies specific gaps. If it decides there isn't enough, the reformulator generates new queries targeting the missing pieces with different terminology, and those go back through retrieval. This can happen up to two times, meaning a single question can trigger up to three complete retrieval passes.

The critical design decision is that context accumulates across passes rather than being overwritten. Each pass appends its results to a shared pool (with deduplication), so by the time the synthesizer runs, it has access to everything the system found across all attempts.


## How a Query Flows Through the System

Here's what actually happens when you send a question:

1. The **planner** classifies it and generates sub queries (or just wraps a simple query as is).

2. For each sub query, the **retriever** fires three parallel searches: semantic over abstracts, semantic over passages, and BM25 keyword matching. The results merge through Reciprocal Rank Fusion into 25 candidates, then the FlashRank cross encoder reranks and filters down to the top few. New chunks get deduplicated against anything accumulated from earlier passes.

3. The **grader** reviews the total accumulated context and decides if it's sufficient. If not, and retries remain, the **reformulator** generates new queries and the retriever runs again.

4. Once the grader is satisfied (or the retry limit hits), the **synthesizer** generates a final answer from all accumulated context, constrained to only cite what was actually retrieved.

```
Planner --> Retriever --> Grader --> Synthesizer
                ^           |
                |       insufficient
                |           |
              Reformulator--+
```


## The API

The system is served through FastAPI with Uvicorn as the ASGI server, deployed on Hugging Face Spaces. On startup, the server loads the ChromaDB collection, loads a pre serialized BM25 index (built ahead of time to avoid a memory spike on boot), and compiles the LangGraph agent. All of this happens during the FastAPI lifespan, so no requests are served until everything is ready.

**POST /query** takes a JSON body with a `query` field and returns the answer, query classification (simple or compound), the sub queries used, retry count, all retrieved chunks with relevance scores, and deduplicated paper citations.

**POST /query/stream** runs the same query as Server Sent Events, emitting a progress event as each graph node finishes and then the full result. This is what lets a frontend show which stage the pipeline is on instead of a bare spinner.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does ColBERT handle late interaction?"}'
```

**GET /papers** returns the full corpus as a browsable list of 1000 entries, each with its arXiv id, title, authors, topic, year, and a link to the arXiv abstract page. The frontend uses this to render a filterable paper browser.

**GET /health** returns `{"status": "ok"}` for the platform's health check.


## Running Locally

The built index is not in this repository. It is roughly 800MB of vectors and a serialized
keyword index, which does not belong in Git. The two build scripts regenerate it from scratch,
which is the intended path: you get the same corpus, and you can change the selection criteria
and get a different one.

```bash
pip install -r requirements.txt

# Create a .env file with OPENAI_API_KEY

# Select the papers. Cheap, no PDFs downloaded, every arXiv response cached,
# so you can tune the filters and re-run for free.
python3 scripts/build_corpus_manifest.py

# Download, parse, chunk, embed, store. Then it rebuilds the BM25 index itself,
# so the keyword index can never be left behind by a corpus rebuild.
# ~20 minutes and ~$0.30 of embedding spend for 1,000 papers.
python3 scripts/build_index.py

# Interactive CLI
python3 -m src.cli

# Or start the API server
uvicorn src.api:app --reload
```

`scripts/build_bm25_cache.py` exists but is not part of this path. It rebuilds the keyword
index alone, for when Chroma is already correct and only the pickle needs replacing.

Two checks worth running after a build:

```bash
python3 scripts/eval_anchor_recall.py   # do the hand-picked papers retrieve themselves?
python3 scripts/bench_latency.py        # reranker, fan-out, embedding, cache, end-to-end
```


## What's Next

There's no evaluation framework yet. The grader makes sufficiency judgments, but I'm not systematically measuring retrieval quality, answer faithfulness, or context relevance. That's the next step: a RAGAS style evaluation pipeline that actually quantifies how well the system is doing rather than relying on spot checks.
