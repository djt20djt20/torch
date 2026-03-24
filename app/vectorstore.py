"""
Vector store — ChromaDB-backed document retrieval, ready to use.

The collection is initialised at app startup via the FastAPI lifespan hook in
main.py (which calls init()). Subsequent calls to retrieve() use the cached
in-process client — no re-loading on each request.

Usage:
    from app.vectorstore import retrieve

    results = retrieve("cyber technology account with prior claims", n_results=3)
    # Returns a list of dicts: {"record_id": ..., "document": ..., "distance": ...}
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
_DOCS_DIR = _REPO_ROOT / "data" / "documents"
_CHROMA_DIR = _REPO_ROOT / ".chroma"
_COLLECTION_NAME = "torch_records"

# ── Embedding function ────────────────────────────────────────────────────────
# Uses the sentence-transformers model locally — no API key required.
# Swap for chromadb.utils.embedding_functions.OpenAIEmbeddingFunction if preferred.

_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ── Collection (initialised at startup) ──────────────────────────────────────

_collection: chromadb.Collection | None = None


def init() -> None:
    """
    Initialise the vector store. Called once at app startup via main.py lifespan.
    Builds the ChromaDB index from data/documents/ if it doesn't exist yet,
    or loads the existing persistent index if it does.
    """
    _get_collection()


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initialising it if not already done."""
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

    existing = {c.name for c in client.list_collections()}
    if _COLLECTION_NAME in existing:
        _collection = client.get_collection(
            name=_COLLECTION_NAME,
            embedding_function=_EMBEDDING_FN,
        )
        return _collection

    # Build index from documents/
    collection = client.create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EMBEDDING_FN,
        metadata={"hnsw:space": "cosine"},
    )

    doc_paths = sorted(_DOCS_DIR.glob("*.txt"))
    if not doc_paths:
        raise FileNotFoundError(
            f"No documents found in {_DOCS_DIR}. "
            "Run generate_data.py first."
        )

    ids, documents, metadatas = [], [], []
    for path in doc_paths:
        record_id = path.stem
        text = path.read_text()
        ids.append(record_id)
        documents.append(text)
        metadatas.append({"record_id": record_id})

    # Add in batches to avoid memory spikes on large corpora
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[start : start + batch_size],
            documents=documents[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
        )

    _collection = collection
    return _collection


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve the top-n most similar historical record documents for a query.

    Args:
        query:     Natural-language description of the record to match against.
        n_results: Number of results to return.

    Returns:
        List of dicts with keys:
            record_id  — the matching record's ID
            document   — the full text of the document summary
            distance   — cosine distance (lower = more similar)
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "record_id": meta["record_id"],
                "document": doc,
                "distance": round(dist, 4),
            }
        )
    return output
