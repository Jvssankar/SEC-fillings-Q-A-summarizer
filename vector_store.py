"""Chroma vector database wrapper for SEC filing chunks."""
import os
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[dict]:
    """Translate {"cik": "320193", "year": "2024"} into Chroma's `where` syntax.

    Chroma needs {"key": {"$eq": value}} and a single {"$and": [...]} wrapper
    when there is more than one condition. Empty values are ignored.
    """
    clauses = [
        {key: {"$eq": str(value)}}
        for key, value in (filters or {}).items()
        if value not in (None, "")
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma accepts only str/int/float/bool values; drop None, stringify the rest."""
    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return cleaned


class SECVectorStore:
    def __init__(self, persist_directory: Optional[str] = None, embeddings=None,
                 collection_name: Optional[str] = None):
        self.persist_directory = str(persist_directory or config.VECTOR_DB_DIR)
        self.collection_name = collection_name or config.COLLECTION_NAME
        os.makedirs(self.persist_directory, exist_ok=True)

        if embeddings is None:
            # imported here so tests (and tools that only read the DB) do not need the model
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.embeddings = embeddings
        self.db = self._open()

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    # ------------------------------------------------------------ writes
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None,
                      batch_size: Optional[int] = None) -> int:
        """Store chunks in batches. With stable `ids`, re-adding a filing replaces it."""
        if not documents:
            return 0
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        size = batch_size or config.BATCH_SIZE
        for start in range(0, len(documents), size):
            batch = documents[start:start + size]
            for doc in batch:
                doc.metadata = _clean_metadata(doc.metadata)
            batch_ids = ids[start:start + size] if ids is not None else None
            self.db.add_documents(batch, ids=batch_ids)
        return len(documents)

    def reset(self) -> None:
        """Delete everything and start with an empty collection."""
        self.db.delete_collection()
        self.db = self._open()

    # ------------------------------------------------------------- reads
    def has_filing(self, source_url: str) -> bool:
        """True only if the filing is fully indexed.

        A filing whose indexing was interrupted has fewer stored chunks than the
        `chunk_count` recorded on its chunks, so it is reported as missing and the
        next run re-indexes it (stable chunk ids make that safe).
        """
        found = self.db.get(where={"source_url": source_url}, include=["metadatas"])
        if not found["ids"]:
            return False
        expected = max(int((m or {}).get("chunk_count") or 0) for m in found["metadatas"])
        return expected == 0 or len(found["ids"]) >= expected

    def retrieve(self, query: str, k: Optional[int] = None,
                 filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return self.db.similarity_search(
            query=query, k=k or config.TOP_K, filter=build_filter(filters)
        )

    def count(self) -> int:
        return self.db._collection.count()

    def list_filings(self) -> List[Dict[str, Any]]:
        """One entry per indexed filing, with its chunk count."""
        got = self.db.get(include=["metadatas"])
        filings: Dict[str, Dict[str, Any]] = {}
        for meta in got.get("metadatas") or []:
            url = (meta or {}).get("source_url")
            if not url:
                continue
            entry = filings.setdefault(
                url,
                {key: meta.get(key, "") for key in
                 ("company", "ticker", "cik", "form_type", "year", "filed_at", "source_url")}
                | {"chunks": 0},
            )
            entry["chunks"] += 1
        return sorted(filings.values(),
                      key=lambda f: (f["company"], f["form_type"], f["filed_at"]))

    def get_info(self) -> Dict[str, Any]:
        filings = self.list_filings()
        return {
            "collection": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunks": self.count(),
            "filings": len(filings),
            "companies": sorted({f["company"] for f in filings}),
            "filing_list": filings,
        }
