import pytest
import requests

from errors import SECDownloadError
from sec_downloader import SECClient

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}
SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {"recent": {
        "accessionNumber": ["0000320193-24-000123", "0000320193-24-000100", "0000320193-24-000090",
                            "0000320193-23-000106", "0000320193-24-000050", "0000320193-24-000040"],
        "form": ["10-K", "10-Q", "8-K", "10-K", "10-K/A", "10-Q"],
        "filingDate": ["2024-11-01", "2024-08-02", "2024-07-01", "2023-11-03", "2024-12-01", "2024-05-03"],
        "reportDate": ["2024-09-28", "2024-06-29", "", "2023-09-30", "2024-09-28", "2024-03-30"],
        "primaryDocument": ["aapl-20240928.htm", "aapl-20240629.htm", "x.htm", "aapl-20230930.htm",
                            "amend.htm", "aapl-20240330.htm"],
    }},
}


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        value = self.routes[url]
        if isinstance(value, list):           # sequence of responses
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def make_client(routes, **kw):
    session = FakeSession(routes)
    client = SECClient(user_agent="Test User test.user@gmail.com", delay=0, session=session,
                       sleep=lambda s: None, **kw)
    return client, session


ROUTES = {
    "https://www.sec.gov/files/company_tickers.json": FakeResp(payload=TICKERS),
    "https://data.sec.gov/submissions/CIK0000320193.json": FakeResp(payload=SUBMISSIONS),
}


def test_user_agent_required():
    for bad in ("", "no-email", "Me me@example.com"):
        with pytest.raises(SECDownloadError):
            SECClient(user_agent=bad)


def test_user_agent_header_is_sent():
    client, session = make_client(ROUTES)
    assert session.headers["User-Agent"] == "Test User test.user@gmail.com"


def test_resolve_ticker():
    client, _ = make_client(ROUTES)
    assert client.resolve_ticker("aapl") == (320193, "Apple Inc.")
    with pytest.raises(SECDownloadError):
        client.resolve_ticker("ZZZZ")


def test_list_filings_newest_per_form_and_urls():
    client, _ = make_client(ROUTES)
    refs = client.list_filings("AAPL", forms=["10-K", "10-Q"], per_form=1)
    assert [(r.form_type, r.filed_at) for r in refs] == [("10-K", "2024-11-01"), ("10-Q", "2024-08-02")]
    k = refs[0]
    assert k.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    assert k.year == "2024" and k.cik == "320193" and k.company == "Apple Inc."
    assert k.metadata()["source_url"] == k.url


def test_list_filings_excludes_amendments_and_other_forms():
    client, _ = make_client(ROUTES)
    refs = client.list_filings("AAPL", forms=["10-K", "10-Q"], per_form=5)
    assert {r.form_type for r in refs} == {"10-K", "10-Q"}
    assert len(refs) == 4                         # 2 x 10-K, 2 x 10-Q (no 10-K/A, no 8-K)


def test_per_form_dict_limits():
    client, _ = make_client(ROUTES)
    refs = client.list_filings("AAPL", forms=["10-K", "10-Q"], per_form={"10-K": 2, "10-Q": 0})
    assert [r.form_type for r in refs] == ["10-K", "10-K"]


def test_retries_then_succeeds():
    routes = {"https://x/a": [FakeResp(429), FakeResp(503), FakeResp(200, {"ok": 1})]}
    client, session = make_client(routes)
    assert client.get("https://x/a").status_code == 200
    assert len(session.calls) == 3


def test_gives_up_after_retries():
    client, session = make_client({"https://x/a": [FakeResp(500)] * 10}, max_retries=2)
    with pytest.raises(SECDownloadError):
        client.get("https://x/a")
    assert len(session.calls) == 3


def test_network_error_is_retried_and_reported():
    err = requests.ConnectionError("down")
    client, session = make_client({"https://x/a": [err, err]}, max_retries=1)
    with pytest.raises(SECDownloadError, match="network error"):
        client.get("https://x/a")


def test_404_and_403_fail_fast_with_helpful_message():
    client, session = make_client({"https://x/404": FakeResp(404), "https://x/403": FakeResp(403)})
    with pytest.raises(SECDownloadError, match="404"):
        client.get("https://x/404")
    with pytest.raises(SECDownloadError, match="SEC_USER_AGENT"):
        client.get("https://x/403")
    assert len(session.calls) == 2                # no retries


def test_invalid_json_is_reported():
    client, _ = make_client({"https://x/j": FakeResp(200, None)})
    with pytest.raises(SECDownloadError, match="invalid JSON"):
        client.get_json("https://x/j")
