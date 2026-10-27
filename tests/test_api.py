import pytest
from fastapi.testclient import TestClient

import api
from errors import (
    AmbiguousCompanyError,
    IndexEmptyError,
    LLMError,
    LLMNotConfiguredError,
    NoFilingsFoundError,
)
from rag_chain import EvidenceQuote, LLMAnswer, SECRAGChain
from test_rag_chain import FakeLLM


@pytest.fixture
def client(filled_store):
    llm = FakeLLM(LLMAnswer(found=True, answer="391,035 million", summary="Sales.",
                            evidence=[EvidenceQuote(source=1, quote="Total net sales were 391,035 million dollars")]))
    chain = SECRAGChain(store=filled_store, llm=llm)
    api.app.dependency_overrides[api.get_chain] = lambda: chain
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_root_and_health(client):
    assert client.get("/").json()["message"].startswith("SEC Filing")
    assert client.get("/health").json() == {"status": "healthy"}


def test_info(client):
    body = client.get("/info").json()
    assert body["chunks"] == 4 and body["filings"] == 3


def test_ask_ok(client):
    r = client.post("/ask", json={"question": "What were total net sales?", "company": "AAPL",
                                  "form_type": "10-K", "year": 2024, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "satisfied" and body["claim_percentage"] == 100
    assert body["sources"][0]["url"].startswith("https://www.sec.gov/")


def test_validation_errors(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422          # too short
    assert client.post("/ask", json={"question": "What were sales?", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "What were sales?", "year": 1800}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


@pytest.mark.parametrize("exc,code", [
    (IndexEmptyError("empty"), 503),
    (LLMNotConfiguredError("no key"), 503),
    (NoFilingsFoundError("none"), 404),
    (AmbiguousCompanyError("many"), 400),
    (LLMError("bad"), 502),
])
def test_error_mapping(exc, code):
    class Boom:
        def ask(self, *a, **k):
            raise exc

    api.app.dependency_overrides[api.get_chain] = lambda: Boom()
    try:
        r = TestClient(api.app).post("/ask", json={"question": "What were sales?"})
    finally:
        api.app.dependency_overrides.clear()
    assert r.status_code == code and r.json()["detail"] == str(exc)
