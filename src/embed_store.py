import time
import chromadb
import tiktoken
from openai import OpenAI, RateLimitError

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "arxiv_papers"


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI's embedding API.

    Batch in groups of 100 (API limit per request is 2048, but 100 is safe).
    Truncate any text exceeding 8000 tokens before sending to API.

    Returns: list of embedding vectors (list of floats)
    """
    client = OpenAI()
    embeddings = []
    encoding = tiktoken.get_encoding("cl100k_base")

    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]

        # Truncate texts exceeding 8000 tokens
        truncated_batch = []
        for text in batch:
            tokens = encoding.encode(text, disallowed_special=())
            if len(tokens) > 8000:
                truncated_tokens = tokens[:8000]
                text = encoding.decode(truncated_tokens)
            truncated_batch.append(text)

        # Retry with exponential backoff on the per-minute token rate limit (429).
        response = None
        for attempt in range(7):
            try:
                response = client.embeddings.create(input=truncated_batch, model=model)
                break
            except RateLimitError:
                if attempt == 6:
                    raise
                time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16, 32s — lets the TPM window reset
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    paper_metadata: dict,
    collection: chromadb.Collection,
    chunk_type: str = "passage"
) -> None:
    """
    Store chunks with embeddings and metadata in ChromaDB.

    For each chunk, store:
        - id: "{paper_id}_chunk_{i}" or "{paper_id}_abstract" for abstracts
        - document: the chunk text
        - embedding: the vector
        - metadata: {
            "paper_id": str,
            "title": str,
            "authors": str (comma-joined),
            "published": str,
            "categories": str (comma-joined),
            "chunk_index": int,
            "chunk_type": str ("passage" or "abstract")
        }

    ChromaDB metadata values must be str, int, float, or bool — no lists.

    Use collection.add() with ids, documents, embeddings, metadatas.
    Batch in groups of 100 to avoid memory issues on large corpora.
    """
    paper_id = paper_metadata["paper_id"]
    title = paper_metadata["title"]
    authors = ", ".join(paper_metadata["authors"])
    published = paper_metadata["published"]
    categories = ", ".join(paper_metadata["categories"])
    field = paper_metadata.get("field", "")

    ids = []
    documents = []
    embedding_list = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        if chunk_type == "abstract":
            ids.append(f"{paper_id}_abstract")
            chunk_index = -1
        else:
            ids.append(f"{paper_id}_chunk_{i}")
            chunk_index = i

        documents.append(chunk)
        embedding_list.append(embeddings[i])
        metadatas.append({
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "published": published,
            "categories": categories,
            "field": field,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type
        })

    for batch_idx in range(0, len(ids), 100):
        batch_ids = ids[batch_idx:batch_idx + 100]
        batch_documents = documents[batch_idx:batch_idx + 100]
        batch_embeddings = embedding_list[batch_idx:batch_idx + 100]
        batch_metadatas = metadatas[batch_idx:batch_idx + 100]

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )


def get_collection() -> chromadb.Collection:
    """Get or create ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection
