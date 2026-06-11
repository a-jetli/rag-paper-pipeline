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
		tokenized_docs = [tokenize(doc["chunk_text"]) for doc in documents]
		self.index = BM25Okapi(tokenized_docs)

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
