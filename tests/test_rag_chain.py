import pytest

import config
from errors import (
    AmbiguousCompanyError,
    IndexEmptyError,
    LLMError,
    LLMNotConfiguredError,
    NoFilingsFoundError,
)
from rag_chain import NOT_FOUND_TEXT, EvidenceQuote, LLMAnswer, SECRAGChain, normalize_form_type, quote_in_text


class FakeLLM:
    def __init__(self, answer=None, exc=None):
        self.answer, self.exc, self.calls = answer, exc, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return self.answer


def good_answer(**kw):
    base = dict(found=True, answer="Total net sales were $391,035 million in fiscal 2024 [S1].",
                summary="Apple's yearly sales for 2024.",
                evidence=[EvidenceQuote(source=1, quote="Total net sales were 391,035 million dollars")])
    base.update(kw)
    return LLMAnswer(**base)


def chain_for(store, answer):
    return SECRAGChain(store=store, llm=FakeLLM(answer))


def test_fully_verified_answer(filled_store):
    result = chain_for(filled_store, good_answer()).ask(
        "What were total net sales?", company="AAPL", form_type="10-K", year=2024)
    assert result.status == "satisfied" and result.claim_percentage == 100
    assert result.evidence[0].verified
    assert result.filters == {"cik": "320193", "form_type": "10-K", "year": "2024"}
    assert result.sources[0].url == "https://www.sec.gov/a/aapl-10k.htm"
    assert result.sources[0].company == "Apple Inc."


def test_fabricated_quote_is_not_verified(filled_store):
    ans = good_answer(evidence=[EvidenceQuote(source=1, quote="Revenue was fifty billion on Mars")])
    r = chain_for(filled_store, ans).ask("What were total net sales?", company="AAPL", year=2024)
    assert r.claim_percentage == 0 and r.status == "not_satisfied"
    assert not r.evidence[0].verified and "None of the quoted evidence" in r.note


def test_half_verified_is_partial(filled_store):
    ans = good_answer(evidence=[
        EvidenceQuote(source=1, quote="Total net sales were 391,035 million dollars"),
        EvidenceQuote(source=1, quote="Profits tripled overnight everywhere"),
    ])
    r = chain_for(filled_store, ans).ask("What were total net sales?", company="AAPL", year=2024)
    assert r.claim_percentage == 50 and r.status == "partially_satisfied"


def test_found_without_evidence_is_not_verified(filled_store):
    r = chain_for(filled_store, good_answer(evidence=[])).ask("net sales?", company="AAPL")
    assert r.status == "not_satisfied" and r.claim_percentage == 0 and "no quotes" in r.note


def test_not_found_answer(filled_store):
    ans = LLMAnswer(found=False, answer="whatever", summary="", evidence=[])
    r = chain_for(filled_store, ans).ask("What is the CEO's favourite colour?", company="AAPL")
    assert r.answer == NOT_FOUND_TEXT and r.status == "not_satisfied" and r.claim_percentage == 0


def test_quote_with_wrong_source_number_is_searched_everywhere(filled_store):
    ans = good_answer(evidence=[EvidenceQuote(source=9, quote="Total net sales were 391,035 million dollars")])
    r = chain_for(filled_store, ans).ask("What were total net sales?", company="AAPL", year=2024)
    assert r.evidence[0].verified and r.evidence[0].source_id is None


def test_company_resolution_by_ticker_name_and_cik(filled_store):
    llm = FakeLLM(good_answer())
    chain = SECRAGChain(store=filled_store, llm=llm)
    assert chain.ask("net sales", company="aapl").filters == {"cik": "320193"}
    assert chain.ask("net sales", company="apple").filters == {"cik": "320193"}
    assert chain.ask("net sales", company="320193").filters == {"cik": "320193"}


def test_unknown_company_lists_what_is_indexed(filled_store):
    with pytest.raises(NoFilingsFoundError, match="Apple Inc."):
        chain_for(filled_store, good_answer()).ask("net sales", company="Nvidia")


def test_ambiguous_company(filled_store):
    filled_store.add_documents(
        [__import__("conftest").make_doc("another apple related text here", company="Apple Hospitality REIT",
                                         ticker="APLE", cik="1418121", url="https://x/y.htm")], ids=["amb"])
    with pytest.raises(AmbiguousCompanyError):
        chain_for(filled_store, good_answer()).ask("net sales", company="apple")


def test_no_filing_for_year(filled_store):
    with pytest.raises(NoFilingsFoundError, match="Indexed filings"):
        chain_for(filled_store, good_answer()).ask("net sales", company="AAPL", year=1999)


def test_form_type_normalisation():
    assert normalize_form_type("10k") == "10-K"
    assert normalize_form_type(" 10 q ") == "10-Q"
    assert normalize_form_type("") is None and normalize_form_type(None) is None


def test_empty_index_and_empty_question(store):
    with pytest.raises(IndexEmptyError):
        chain_for(store, good_answer()).ask("anything")


def test_empty_question(filled_store):
    with pytest.raises(ValueError):
        chain_for(filled_store, good_answer()).ask("   ")


def test_llm_failures_become_llm_error(filled_store):
    with pytest.raises(LLMError):
        SECRAGChain(store=filled_store, llm=FakeLLM(exc=RuntimeError("quota"))).ask("net sales", company="AAPL")
    with pytest.raises(LLMError):
        SECRAGChain(store=filled_store, llm=FakeLLM(answer=None)).ask("net sales", company="AAPL")


def test_dict_response_is_accepted(filled_store):
    raw = good_answer().model_dump()
    r = SECRAGChain(store=filled_store, llm=FakeLLM(raw)).ask("net sales", company="AAPL", year=2024)
    assert r.status == "satisfied"


def test_missing_api_key_is_reported_before_retrieval(filled_store, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config, "google_api_key", lambda: "")
    with pytest.raises(LLMNotConfiguredError):
        SECRAGChain(store=filled_store).ask("net sales", company="AAPL")


def test_context_budget_limits_sources(filled_store, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXT_CHARS", 150)
    llm = FakeLLM(good_answer())
    r = SECRAGChain(store=filled_store, llm=llm).ask("net sales dollars", top_k=4)
    sent = llm.calls[0][1][1]
    assert len(r.sources) < 4                                   # only what fits is cited ...
    assert all(f"[S{s.id}]" in sent for s in r.sources)         # ... and all of it was sent to the model


def test_prompt_contains_sources_and_question(filled_store):
    llm = FakeLLM(good_answer())
    SECRAGChain(store=filled_store, llm=llm).ask("What were total net sales?", company="AAPL", year=2024)
    system, human = llm.calls[0]
    assert system[0] == "system" and "ONLY" in system[1]
    assert "[S1] Apple Inc. (AAPL) 10-K for 2024" in human[1] and "What were total net sales?" in human[1]


def test_schema_sent_to_gemini_is_self_contained(monkeypatch):
    import json

    from rag_chain import inline_schema

    schema = inline_schema(LLMAnswer)
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema
    assert set(schema["properties"]) == {"found", "answer", "summary", "evidence"}
    assert set(schema["properties"]["evidence"]["items"]["properties"]) == {"source", "quote"}

    monkeypatch.setattr(config, "google_api_key", lambda: "fake-key-for-offline-test")
    runnable = SECRAGChain._build_llm()                       # builds the client, makes no network call
    sent = json.dumps(runnable.first.kwargs["response_json_schema"])
    assert "$ref" not in sent and "quote" in sent


def test_quote_matching_ignores_punctuation_and_case():
    text = "Total net sales were $391,035 million."
    assert quote_in_text("total NET sales were 391,035 million", text)
    assert not quote_in_text("net sales were 999", text)
    assert not quote_in_text("sales", text)                     # one-word quotes prove nothing
