import pytest

import document_processor as dp
from errors import SECDownloadError

HTML = """
<html><head><title>x</title></head><body>
<div style="display:none"><ix:header><ix:hidden>HIDDEN-XBRL-FACT 12345</ix:hidden></ix:header></div>
<p>Item 7. Management's Discussion</p>
<p><span>Total</span> <span>net</span> <span>sales</span> increased.</p>
<table>
 <tr><td>Net sales</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>
 <tr><td>Net loss</td><td>(1,200</td><td>)</td><td>5.2</td><td>%</td></tr>
</table>
<script>var secret = 1;</script>
<p>Page one<br/>second line</p>
<a href="https://www.sec.gov/Archives/edgar/data/1/other.htm">exhibit</a>
</body></html>
"""


def test_html_to_text_keeps_tables_and_drops_noise():
    text = dp.html_to_text(HTML)
    assert "Net sales | 391,035 | 383,285" in text          # table kept, "$" cells merged away
    assert "Net loss | (1,200) | 5.2%" in text                # ")" and "%" glued back
    assert "Total net sales increased." in text               # inline spans joined with spaces
    assert "HIDDEN-XBRL-FACT" not in text                     # hidden inline XBRL removed
    assert "secret" not in text and "var " not in text        # script removed
    assert "Page one\nsecond line" in text                    # <br> becomes a newline


def test_ix_header_removed_even_without_hidden_wrapper():
    text = dp.html_to_text("<body><ix:header><ix:hidden>BARE-FACT 999</ix:hidden></ix:header><p>Real text here.</p></body>")
    assert "BARE-FACT" not in text and "Real text here." in text


def test_nested_table_does_not_crash():
    html = "<table><tr><td>Outer <table><tr><td>Inner</td><td>7</td></tr></table></td><td>9</td></tr></table>"
    text = dp.html_to_text(html)
    assert "Inner" in text and "9" in text


def test_chunking_metadata_and_ids():
    proc = dp.SECDocumentProcessor()
    text = ("Revenue grew strongly this year. " * 200).strip()
    docs = proc.chunk_document(text, {"company": "X", "source_url": "u"})
    assert len(docs) > 3
    assert all(len(d.page_content) <= dp.config.CHUNK_SIZE for d in docs)
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    assert all(d.metadata["chunk_count"] == len(docs) for d in docs)
    assert docs[0].metadata["company"] == "X"
    ids = [dp.chunk_id("u", d.metadata["chunk_index"]) for d in docs]
    assert len(set(ids)) == len(ids)
    assert ids == [dp.chunk_id("u", i) for i in range(len(docs))]   # stable


def test_chunk_document_empty_text():
    assert dp.SECDocumentProcessor().chunk_document("", {}) == []


class _Resp:
    def __init__(self, content, ctype="text/html"):
        self.content = content
        self.headers = {"Content-Type": ctype}


class _Client:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc = resp, exc

    def get(self, url):
        if self.exc:
            raise self.exc
        return self.resp


def test_fetch_filing_text_html_does_not_follow_links():
    proc = dp.SECDocumentProcessor(_Client(_Resp(HTML.encode())))
    text = proc.fetch_filing_text("https://www.sec.gov/Archives/edgar/data/1/x.htm")
    assert "Total net sales increased." in text        # the page itself, not the linked exhibit


def test_fetch_filing_text_download_error_is_raised():
    proc = dp.SECDocumentProcessor(_Client(exc=SECDownloadError("boom")))
    with pytest.raises(SECDownloadError):
        proc.fetch_filing_text("https://www.sec.gov/x.htm")


def test_fetch_bad_pdf_raises_readable_error():
    proc = dp.SECDocumentProcessor(_Client(_Resp(b"not a pdf", "application/pdf")))
    with pytest.raises(SECDownloadError):
        proc.fetch_filing_text("https://www.sec.gov/x.pdf")
