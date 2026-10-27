import hashlib
import os
import re
import sys

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SECVectorStore  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: similar words give similar vectors."""

    dim = 256

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def store(tmp_path):
    return SECVectorStore(persist_directory=str(tmp_path / "db"), embeddings=FakeEmbeddings(),
                          collection_name="test")


def make_doc(text, company="Apple Inc.", ticker="AAPL", cik="320193", form="10-K",
             year="2024", filed="2024-11-01", url="https://www.sec.gov/a/aapl-10k.htm", idx=0):
    return Document(page_content=text, metadata={
        "company": company, "ticker": ticker, "cik": cik, "form_type": form, "year": year,
        "filed_at": filed, "source_url": url, "chunk_index": idx,
    })


@pytest.fixture
def filled_store(store):
    docs = [
        make_doc("Total net sales were 391,035 million dollars in fiscal 2024.", idx=0),
        make_doc("Risk factors include supply chain disruption and currency exposure.", idx=1),
        make_doc("Net sales for fiscal 2023 were 383,285 million dollars.", year="2023",
                 filed="2023-11-03", url="https://www.sec.gov/a/aapl-10k-2023.htm"),
        make_doc("Quarterly revenue of Microsoft grew due to cloud services.",
                 company="MICROSOFT CORP", ticker="MSFT", cik="789019", form="10-Q",
                 year="2024", filed="2024-04-25", url="https://www.sec.gov/m/msft-10q.htm"),
    ]
    store.add_documents(docs, ids=[f"id{i}" for i in range(len(docs))])
    return store
