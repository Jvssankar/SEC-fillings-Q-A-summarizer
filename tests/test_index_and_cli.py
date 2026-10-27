import index_documents
import query_engine
from errors import SECDownloadError
from rag_chain import EvidenceQuote, LLMAnswer, SECRAGChain
from sec_downloader import FilingRef
from test_rag_chain import FakeLLM

BODY = ("Total net sales were 391,035 million dollars in fiscal 2024. "
        "The company designs smartphones and services. ") * 60


def ref(ticker="AAPL", form="10-K", year="2024", n=1):
    return FilingRef(company="Apple Inc.", ticker=ticker, cik="320193", form_type=form,
                     filed_at=f"{year}-11-01", period_of_report=f"{year}-09-28",
                     accession_no=f"0000320193-{year[2:]}-00000{n}",
                     url=f"https://www.sec.gov/Archives/edgar/data/320193/{n}/doc{n}.htm")


class FakeClient:
    def __init__(self, refs_by_ticker):
        self.refs_by_ticker = refs_by_ticker

    def list_filings(self, ticker, forms, per_form):
        if ticker not in self.refs_by_ticker:
            raise SECDownloadError(f"Ticker '{ticker}' was not found in SEC's company list.")
        return self.refs_by_ticker[ticker]


class FakeProcessor:
    """Real chunking, fake downloads."""

    def __init__(self, texts):
        from document_processor import SECDocumentProcessor

        self.texts, self.real, self.fetched = texts, SECDocumentProcessor(), []

    def fetch_filing_text(self, url):
        self.fetched.append(url)
        value = self.texts[url]
        if isinstance(value, Exception):
            raise value
        return value

    def chunk_document(self, text, metadata):
        return self.real.chunk_document(text, metadata)


def test_indexing_is_idempotent_and_reports_failures(store, capsys):
    good, bad, short = ref(n=1), ref(form="10-Q", n=2), ref(year="2023", n=3)
    client = FakeClient({"AAPL": [good, bad, short]})
    proc = FakeProcessor({good.url: BODY, bad.url: SECDownloadError("HTTP 500"), short.url: "tiny"})

    code = index_documents.main(["--tickers", "AAPL"], client=client, processor=proc, store=store)
    out = capsys.readouterr().out
    assert code == 0
    assert store.has_filing(good.url) and not store.has_filing(bad.url) and not store.has_filing(short.url)
    assert "FAIL" in out and "HTTP 500" in out and "only 4 characters" in out
    first_count = store.count()
    assert first_count > 5

    proc.fetched.clear()
    index_documents.main(["--tickers", "AAPL"], client=client, processor=proc, store=store)
    assert store.count() == first_count            # no duplicates
    assert good.url not in proc.fetched            # not downloaded again


def test_interrupted_run_is_completed_without_duplicates(store, capsys):
    from document_processor import SECDocumentProcessor, chunk_id

    good = ref()
    real = SECDocumentProcessor()
    docs = real.chunk_document(BODY, good.metadata())
    half = docs[: len(docs) // 2]
    store.add_documents(half, ids=[chunk_id(good.url, d.metadata["chunk_index"]) for d in half])
    assert not store.has_filing(good.url)

    index_documents.main(["--tickers", "AAPL"], client=FakeClient({"AAPL": [good]}),
                         processor=FakeProcessor({good.url: BODY}), store=store)
    assert store.has_filing(good.url) and store.count() == len(docs)


def test_unknown_ticker_and_nothing_to_index(store, capsys):
    code = index_documents.main(["--tickers", "ZZZZ"], client=FakeClient({}), processor=FakeProcessor({}), store=store)
    assert code == 1 and "was not found" in capsys.readouterr().out


def test_list_and_reset(store, capsys):
    good = ref()
    client, proc = FakeClient({"AAPL": [good]}), FakeProcessor({good.url: BODY})
    index_documents.main(["--tickers", "AAPL"], client=client, processor=proc, store=store)
    capsys.readouterr()
    assert index_documents.main(["--list"], store=store) == 0
    assert "Apple Inc. (AAPL) 10-K 2024" in capsys.readouterr().out
    index_documents.main(["--tickers", "AAPL", "--reset"], client=client, processor=proc, store=store)
    assert store.count() > 0 and store.has_filing(good.url)


def test_missing_user_agent_is_a_clear_error(store, monkeypatch, capsys):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr("config.sec_user_agent", lambda: "")
    assert index_documents.main(["--tickers", "AAPL"], store=store) == 1
    assert "SEC_USER_AGENT" in capsys.readouterr().out


def test_cli_end_to_end_output(filled_store, capsys):
    llm = FakeLLM(LLMAnswer(found=True, answer="391,035 million", summary="Yearly sales.",
                            evidence=[EvidenceQuote(source=1, quote="Total net sales were 391,035 million dollars"),
                                      EvidenceQuote(source=1, quote="made up statement about nothing")]))
    chain = SECRAGChain(store=filled_store, llm=llm)
    result = query_engine.ask_question("AAPL", "10-K", "2024", "What were total net sales?", chain)
    text = query_engine.format_result(result)
    assert "391,035 million" in text and "[verified]" in text and "NOT FOUND in filing" in text
    assert "CLAIM PERCENTAGE: 50% (Partially satisfied)" in text
    assert "https://www.sec.gov/a/aapl-10k.htm" in text
