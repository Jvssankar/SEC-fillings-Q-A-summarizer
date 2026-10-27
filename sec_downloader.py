"""Talks to SEC EDGAR: finds the 10-K / 10-Q filings of a company and downloads them.

SEC fair-access rules (https://www.sec.gov/os/accessing-edgar-data):
  * every request must carry a User-Agent that names you and gives a contact email
  * at most 10 requests per second (this client stays at ~4 per second)
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import requests

import config
from errors import SECDownloadError

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


@dataclass(frozen=True)
class FilingRef:
    """Pointer to one filing's main document on EDGAR."""

    company: str
    ticker: str
    cik: str
    form_type: str
    filed_at: str
    period_of_report: str
    accession_no: str
    url: str

    @property
    def year(self) -> str:
        """Fiscal year the filing reports on (falls back to the filing year)."""
        return (self.period_of_report or self.filed_at or "")[:4]

    def metadata(self) -> Dict[str, str]:
        return {
            "company": self.company,
            "ticker": self.ticker,
            "cik": self.cik,
            "form_type": self.form_type,
            "filed_at": self.filed_at,
            "period_of_report": self.period_of_report,
            "year": self.year,
            "accession_no": self.accession_no,
            "source_url": self.url,
        }


class SECClient:
    """Small, polite HTTP client for EDGAR with throttling and retries."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        delay: Optional[float] = None,
        max_retries: Optional[int] = None,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        ua = (user_agent if user_agent is not None else config.sec_user_agent()).strip()
        if "@" not in ua or "example.com" in ua.lower():
            raise SECDownloadError(
                "SEC requires a real contact in the User-Agent. Set SEC_USER_AGENT in your "
                '.env file, for example: SEC_USER_AGENT="Jane Doe jane.doe@gmail.com"'
            )
        self.user_agent = ua
        self.delay = config.SEC_REQUEST_DELAY if delay is None else delay
        self.max_retries = config.SEC_MAX_RETRIES if max_retries is None else max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self._sleep = sleep
        self._last_request = 0.0
        self._tickers: Optional[Dict[str, Tuple[int, str]]] = None

    # ------------------------------------------------------------ HTTP
    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str, timeout: float = 30) -> requests.Response:
        """GET with throttling; retries 429 / 5xx / network errors with backoff."""
        last_problem = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                last_problem = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    raise SECDownloadError(f"Not found (404): {url}")
                if resp.status_code in (401, 403):
                    raise SECDownloadError(
                        f"SEC refused the request (HTTP {resp.status_code}) for {url}. "
                        "Check that SEC_USER_AGENT contains your name and a real email."
                    )
                last_problem = f"HTTP {resp.status_code}"
                if resp.status_code != 429 and resp.status_code < 500:
                    raise SECDownloadError(f"Unexpected {last_problem} for {url}")
            if attempt < self.max_retries:
                backoff = 2 ** attempt
                log.warning("%s for %s - retrying in %ss", last_problem, url, backoff)
                self._sleep(backoff)
        raise SECDownloadError(f"Giving up on {url} after {self.max_retries + 1} attempts ({last_problem})")

    def get_json(self, url: str) -> dict:
        try:
            return self.get(url).json()
        except ValueError as exc:
            raise SECDownloadError(f"SEC returned invalid JSON for {url}") from exc

    # ---------------------------------------------------------- lookups
    def resolve_ticker(self, ticker: str) -> Tuple[int, str]:
        """Return (cik, company name) for a ticker symbol."""
        if self._tickers is None:
            data = self.get_json(TICKERS_URL)
            self._tickers = {
                str(row["ticker"]).upper(): (int(row["cik_str"]), str(row["title"]))
                for row in data.values()
                if "ticker" in row and "cik_str" in row
            }
        key = ticker.strip().upper()
        if key not in self._tickers:
            raise SECDownloadError(f"Ticker '{ticker}' was not found in SEC's company list.")
        return self._tickers[key]

    def list_filings(
        self,
        ticker: str,
        forms: Iterable[str] = config.FORMS,
        per_form: Union[int, Dict[str, int]] = config.FILINGS_PER_FORM,
    ) -> List[FilingRef]:
        """Newest filings of the requested forms for one company."""
        forms = list(forms)
        cik, title = self.resolve_ticker(ticker)
        data = self.get_json(SUBMISSIONS_URL.format(cik=cik))
        recent = (data.get("filings") or {}).get("recent") or {}
        rows = zip(
            recent.get("accessionNumber", []),
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            recent.get("primaryDocument", []),
        )
        wanted = [r for r in rows if r[1] in forms and r[4]]
        wanted.sort(key=lambda r: r[2], reverse=True)

        name = data.get("name") or title
        taken: Dict[str, int] = {}
        refs: List[FilingRef] = []
        for accession, form, filed, period, document in wanted:
            limit = per_form.get(form, 0) if isinstance(per_form, dict) else per_form
            if taken.get(form, 0) >= limit:
                continue
            taken[form] = taken.get(form, 0) + 1
            refs.append(
                FilingRef(
                    company=name,
                    ticker=ticker.strip().upper(),
                    cik=str(cik),
                    form_type=form,
                    filed_at=filed,
                    period_of_report=period or "",
                    accession_no=accession,
                    url=ARCHIVE_URL.format(
                        cik=cik, accession_nodash=accession.replace("-", ""), document=document
                    ),
                )
            )
        return refs
