"""
Retrieval layer for the text-to-SQL pipeline. Two retrievers:

1. SchemaRetriever - given a natural language question, returns only the
   tables likely relevant to it, instead of dumping the entire schema into
   every prompt. This is "schema linking" - the retrieval step that matters
   most for text-to-SQL accuracy on databases with many tables.

2. FewShotRetriever - given a question, returns the most similar past
   (question, sql) pairs to include as in-context examples. Few-shot
   examples measurably improve SQL generation accuracy, especially for
   schema-specific conventions an LLM can't infer from column names alone.

Why TF-IDF instead of a neural embedding model: schema text (table/column
names) and short questions are keyword-heavy, low-vocabulary text where
TF-IDF + cosine similarity performs well, requires no model download, no
GPU, and is effectively free and instant. A drop-in swap to sentence
embeddings (e.g. sentence-transformers) is noted in the README for anyone
who wants to extend this.
"""
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.db import TableChunk


@dataclass
class FewShotExample:
    question: str
    sql: str


class SchemaRetriever:
    """Retrieves the top-k most relevant tables for a given question."""

    def __init__(self, chunks: list[TableChunk]):
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            ngram_range=(1, 2),
            lowercase=True,
        )
        corpus = [self._chunk_to_document(c) for c in chunks]
        if corpus:
            self._matrix = self._vectorizer.fit_transform(corpus)
        else:
            self._matrix = None

    @staticmethod
    def _chunk_to_document(chunk: TableChunk) -> str:
        # Split snake_case/camelCase column names into words so "customer_id"
        # also matches a question containing "customer" - this materially
        # improves retrieval quality on real-world schemas.
        expanded_cols = " ".join(
            name.replace("_", " ") for name in chunk.column_names
        )
        return f"{chunk.table_name.replace('_', ' ')} {expanded_cols} {chunk.text}"

    def retrieve(self, question: str, top_k: int = 5, min_score: float = 0.02) -> list[TableChunk]:
        """
        Returns up to top_k table chunks most relevant to the question.
        Always includes at least 1 table if any exist, even below min_score,
        so a poorly-matched question still gets something to work with.
        """
        if not self.chunks or self._matrix is None:
            return []

        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked_idx = np.argsort(scores)[::-1]

        selected = []
        for idx in ranked_idx[:top_k]:
            if scores[idx] >= min_score or len(selected) == 0:
                selected.append((self.chunks[idx], float(scores[idx])))

        return selected  # list of (TableChunk, score)


class FewShotRetriever:
    """Retrieves the top-k most similar past (question, sql) examples."""

    def __init__(self, examples: list[FewShotExample] | None = None):
        self.examples: list[FewShotExample] = examples or []
        self._vectorizer = None
        self._matrix = None
        self._rebuild_index()

    def _rebuild_index(self):
        if not self.examples:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([e.question for e in self.examples])

    def add_example(self, question: str, sql: str):
        self.examples.append(FewShotExample(question=question, sql=sql))
        self._rebuild_index()

    def retrieve(self, question: str, top_k: int = 3, min_score: float = 0.05) -> list[FewShotExample]:
        if not self.examples or self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [self.examples[i] for i in ranked_idx if scores[i] >= min_score]
