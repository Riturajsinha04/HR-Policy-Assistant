"""
Retriever module: vector search (+ optional BM25 hybrid) over the ChromaDB index.

Provides a simple interface to retrieve the most relevant chunks for a query,
with relevance scores and metadata.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    RETRIEVER_TOP_K,
)

logger = logging.getLogger(__name__)


def _load_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store() -> Chroma:
    """
    Load the persisted ChromaDB vector store.
    Raises FileNotFoundError if the index has not been built yet.
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DIR}. "
            "Run `python -m src.ingest --rebuild` first."
        )

    embeddings = _load_embeddings()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    count = store._collection.count()
    if count == 0:
        raise FileNotFoundError(
            f"ChromaDB collection '{COLLECTION_NAME}' is empty. "
            "Run `python -m src.ingest --rebuild` first."
        )

    logger.info("Loaded ChromaDB: %d documents in '%s'", count, COLLECTION_NAME)
    return store


def retrieve(
    query: str,
    top_k: int = RETRIEVER_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k relevant chunks for the given query.

    Returns:
        List of (Document, relevance_score) tuples, sorted by descending
        relevance.  Chunks below min_score are filtered out.

    ChromaDB returns distances (lower = more similar for cosine).
    We convert to a 0-1 relevance score: score = 1 - (distance / 2).
    """
    store = get_vector_store()

    # similarity_search_with_score returns (doc, distance)
    raw_results = store.similarity_search_with_score(query, k=top_k)

    # Convert distance to relevance score and filter
    results: List[Tuple[Document, float]] = []
    for doc, distance in raw_results:
        # Chroma cosine distance is in [0, 2]; convert to similarity in [0, 1]
        score = 1.0 - (distance / 2.0)
        if score >= min_score:
            results.append((doc, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "Query: '%s' → %d results (from %d raw, min_score=%.2f)",
        query[:60],
        len(results),
        len(raw_results),
        min_score,
    )
    return results


def retrieve_with_store(
    store: Chroma,
    query: str,
    top_k: int = RETRIEVER_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> List[Tuple[Document, float]]:
    """
    Same as retrieve() but accepts a pre-loaded vector store.
    Used by the Streamlit app to avoid reloading the store on every query.
    """
    raw_results = store.similarity_search_with_score(query, k=top_k)

    results: List[Tuple[Document, float]] = []
    for doc, distance in raw_results:
        score = 1.0 - (distance / 2.0)
        if score >= min_score:
            results.append((doc, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
