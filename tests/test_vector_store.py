import pytest

from conftest import make_doc
from vector_store import build_filter


def test_build_filter_shapes():
    assert build_filter(None) is None
    assert build_filter({}) is None
    assert build_filter({"cik": "1", "year": ""}) == {"cik": {"$eq": "1"}}
    assert build_filter({"cik": "1", "year": 2024}) == {
        "$and": [{"cik": {"$eq": "1"}}, {"year": {"$eq": "2024"}}]
    }


def test_filtered_retrieval_two_and_three_conditions(filled_store):
    two = filled_store.retrieve("net sales", k=5, filters={"cik": "320193", "form_type": "10-K"})
    assert two and all(d.metadata["cik"] == "320193" for d in two)
    three = filled_store.retrieve("net sales", k=5,
                                  filters={"cik": "320193", "form_type": "10-K", "year": "2023"})
    assert len(three) == 1 and three[0].metadata["year"] == "2023"


def test_single_filter_and_no_filter(filled_store):
    assert all(d.metadata["ticker"] == "MSFT"
               for d in filled_store.retrieve("revenue", k=5, filters={"ticker": "MSFT"}))
    assert len(filled_store.retrieve("revenue", k=10)) == 4


def test_filter_with_no_match_returns_empty(filled_store):
    assert filled_store.retrieve("net sales", filters={"year": "1999"}) == []


def test_same_ids_do_not_duplicate(store):
    docs = [make_doc("alpha beta gamma delta"), make_doc("epsilon zeta eta theta", idx=1)]
    store.add_documents(docs, ids=["a", "b"])
    store.add_documents([make_doc("alpha beta gamma delta"), make_doc("epsilon zeta eta theta", idx=1)],
                        ids=["a", "b"])
    assert store.count() == 2


def test_batching_writes_everything(store):
    docs = [make_doc(f"unique passage number {i}", idx=i) for i in range(25)]
    store.add_documents(docs, ids=[f"x{i}" for i in range(25)], batch_size=10)
    assert store.count() == 25


def test_ids_length_mismatch_raises(store):
    with pytest.raises(ValueError):
        store.add_documents([make_doc("one two three")], ids=["a", "b"])


def test_has_filing(filled_store):
    assert filled_store.has_filing("https://www.sec.gov/a/aapl-10k.htm")
    assert not filled_store.has_filing("https://www.sec.gov/nope.htm")


def test_has_filing_is_false_for_a_partially_indexed_filing(store):
    docs = [make_doc(f"passage {i} about revenue", idx=i) for i in range(6)]
    for d in docs:
        d.metadata["chunk_count"] = 6
    store.add_documents(docs[:3], ids=["p0", "p1", "p2"])          # run interrupted halfway
    assert not store.has_filing("https://www.sec.gov/a/aapl-10k.htm")
    store.add_documents(docs, ids=[f"p{i}" for i in range(6)])     # re-run finishes the job
    assert store.has_filing("https://www.sec.gov/a/aapl-10k.htm")
    assert store.count() == 6                                       # nothing duplicated


def test_list_filings_and_info(filled_store):
    filings = filled_store.list_filings()
    assert len(filings) == 3
    info = filled_store.get_info()
    assert info["chunks"] == 4 and info["filings"] == 3
    assert info["companies"] == ["Apple Inc.", "MICROSOFT CORP"]


def test_none_metadata_is_dropped(store):
    doc = make_doc("some text that is long enough")
    doc.metadata["ticker"] = None
    store.add_documents([doc], ids=["n"])
    assert store.count() == 1


def test_reset_empties_store(filled_store):
    filled_store.reset()
    assert filled_store.count() == 0
    filled_store.add_documents([make_doc("fresh start text here")], ids=["z"])
    assert filled_store.count() == 1
