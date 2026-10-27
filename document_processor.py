"""Turns an SEC filing (HTML or PDF) into clean text and splits it into chunks."""
import hashlib
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Union

from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
from errors import SECDownloadError
from sec_downloader import SECClient

log = logging.getLogger(__name__)

_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_BLOCK_TAGS = [
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "center",
    "h1", "h2", "h3", "h4", "h5", "h6",
]
_TRAILING_CELLS = {")", "%", ")%"}


def _drop(tags) -> None:
    for tag in tags:
        if getattr(tag, "decomposed", False):
            continue
        tag.decompose()


def _clean_cells(cells: List[str]) -> List[str]:
    """Tidy SEC table cells: "$", "391,035", ")" become "391,035)" etc."""
    out: List[str] = []
    for cell in cells:
        cell = cell.strip()
        if not cell or cell == "$":
            continue
        if cell in _TRAILING_CELLS and out:
            out[-1] += cell
        else:
            out.append(cell)
    return out


def html_to_text(html: Union[str, bytes]) -> str:
    """Readable text from a filing page. Table rows are kept (as 'a | b | c')
    because financial figures live in tables."""
    soup = BeautifulSoup(html, "lxml")

    _drop(soup.find_all(["script", "style", "noscript", "head"]))
    _drop(soup.find_all(re.compile(r"^ix:header$")))  # hidden inline-XBRL facts
    _drop(soup.find_all(style=_HIDDEN_STYLE))

    # innermost tables first so nested tables flatten correctly
    for table in reversed(soup.find_all("table")):
        if table.parent is None:
            continue
        rows = []
        for tr in table.find_all("tr"):
            cells = _clean_cells([c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])])
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append(extracted)
    return "\n".join(pages).strip()


def chunk_id(source_url: str, index: int) -> str:
    """Stable id so re-indexing the same filing overwrites instead of duplicating."""
    return hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:12] + f"-{index:05d}"


class SECDocumentProcessor:
    def __init__(self, client: Optional[SECClient] = None):
        self._client = client

    @property
    def client(self) -> SECClient:
        if self._client is None:
            self._client = SECClient()
        return self._client

    def fetch_filing_text(self, url: str) -> str:
        """Download one filing and return its text.

        Raises SECDownloadError when the download fails, so problems are visible
        instead of silently producing empty documents.
        """
        resp = self.client.get(url)
        content_type = resp.headers.get("Content-Type", "").lower()
        try:
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                return pdf_to_text(resp.content)
            return html_to_text(resp.content)
        except Exception as exc:  # parsing problems are reported, not hidden
            raise SECDownloadError(f"Could not read the document at {url}: {exc}") from exc

    def chunk_document(self, text: str, metadata: Dict[str, str]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = [p.strip() for p in splitter.split_text(text)]
        pieces = [p for p in pieces if len(p) >= config.MIN_CHUNK_CHARS]
        # chunk_count lets the store tell a complete filing from one whose indexing was interrupted
        return [
            Document(page_content=piece,
                     metadata={**metadata, "chunk_index": i, "chunk_count": len(pieces)})
            for i, piece in enumerate(pieces)
        ]
