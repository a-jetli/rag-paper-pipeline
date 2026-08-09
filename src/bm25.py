import pickle
import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
	"""
	Simple whitespace tokenizer with normalization.

	Lowercase, strip non-alphanumeric characters, split on whitespace,
	remove tokens shorter than 2 characters.
	"""
	text = text.lower()
	text = re.sub(r'[^a-z0-9\s]', '', text)
	tokens = text.split()
	return [t for t in tokens if len(t) >= 2]


def documents_from_collection(collection) -> list[dict]:
	"""Build BM25 document dictionaries from every record in a Chroma collection."""
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
	return documents


def build_and_save(collection, filepath: str) -> "BM25Index":
	"""
	Build the BM25 index from a Chroma collection and persist it.

	Lives here rather than in a script so `build_index.py` can call it directly
	when it finishes writing Chroma. The two indexes only ever drift when that
	second step is a separate command someone has to remember.
	"""
	documents = documents_from_collection(collection)
	index = BM25Index(documents)
	index.save(filepath)
	return index


class BM25Index:
	"""
	Wrapper around BM25Okapi that maintains the same interface as the custom implementation.

	Each document is a dict with at minimum:
		- "chunk_text": str
		- "chunk_id": str (unique identifier)

	Additional keys are preserved and returned with results.
	"""

	def __init__(self, documents: list[dict]):
		"""
		Build the BM25 index from a list of document dicts.

		Tokenizes each document's chunk_text and builds a BM25Okapi index.
		Stores original documents for result retrieval.

		Instance variables:
		- self.documents: list[dict] — original document dicts
		- self.index: BM25Okapi — the rank_bm25 index
		"""
		self.documents = documents
		self.chunk_count = len(documents)
		tokenized_docs = [tokenize(doc["chunk_text"]) for doc in documents]
		self.index = BM25Okapi(tokenized_docs)

	def matches_collection(self, collection) -> tuple[bool, str]:
		"""
		Cheap staleness check: does this index cover the same number of chunks
		the collection currently holds?

		`collection.count()` is a metadata lookup (~0.04s). Reading every
		document back out to compare text would cost ~1s and defeat the point
		of caching the index at all.

		Every realistic way the two drift changes the chunk count — papers
		added or removed, re-chunking, re-indexing. The case this misses is an
		identical chunk count whose text changed, which needs a cleaning change
		that leaves every chunk boundary intact. `build_and_save()` being called
		from the indexer is what actually prevents drift; this is the backstop
		for a restored backup or a half-finished build.
		"""
		cached = getattr(self, "chunk_count", None)
		if cached is None:
			return False, "cache predates the chunk-count check"
		live = collection.count()
		if cached != live:
			return False, f"chunk count differs: cache={cached} collection={live}"
		return True, f"chunk count matches ({live})"

	def save(self, filepath: str):
		with open(filepath, "wb") as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filepath: str):
		with open(filepath, "rb") as f:
			return pickle.load(f)

	def query(self, query_text: str, n_results: int = 10) -> list[dict]:
		"""
		Score all documents against the query and return top-n.

		Args:
			query_text: the query string
			n_results: number of results to return

		Returns: list of dicts sorted by bm25_score descending,
		each with all original keys plus a "bm25_score" key.
		"""
		query_tokens = tokenize(query_text)
		scores = self.index.get_scores(query_tokens)

		scored_docs = [
			(idx, score) for idx, score in enumerate(scores) if score > 0
		]
		scored_docs.sort(key=lambda x: x[1], reverse=True)

		results = []
		for doc_idx, bm25_score in scored_docs[:n_results]:
			doc = self.documents[doc_idx].copy()
			doc["bm25_score"] = bm25_score
			results.append(doc)

		return results
