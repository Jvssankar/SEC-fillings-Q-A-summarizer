"""Download real 10-K / 10-Q filings from SEC EDGAR and index them for questions.

    python index_documents.py                       # companies from TICKERS in .env
    python index_documents.py --tickers AAPL NVDA   # choose companies
    python index_documents.py --forms 10-K --per-form 3
    python index_documents.py --reset               # wipe the database first
    python index_documents.py --list                # show what is indexed

Re-running is safe: filings that are already indexed are skipped.
"""
import argparse
import sys
from typing import Dict, List, Optional

import config
from document_processor import SECDocumentProcessor, chunk_id
from errors import SECProjectError
from sec_downloader import FilingRef, SECClient


def index_filings(refs: List[FilingRef], processor, store) -> Dict[str, int]:
    """Download, chunk and store each filing. Returns counts for the final report."""
    stats = {"indexed": 0, "skipped_existing": 0, "failed": 0, "chunks": 0}
    for ref in refs:
        label = f"{ref.ticker} {ref.form_type} {ref.year} (filed {ref.filed_at})"
        if store.has_filing(ref.url):
            print(f"  skip   {label}: already indexed")
            stats["skipped_existing"] += 1
            continue
        try:
            text = processor.fetch_filing_text(ref.url)
        except SECProjectError as exc:
            print(f"  FAIL   {label}: {exc}")
            stats["failed"] += 1
            continue
        if len(text) < config.MIN_TEXT_LENGTH:
            print(f"  FAIL   {label}: only {len(text)} characters of text were extracted")
            stats["failed"] += 1
            continue
        documents = processor.chunk_document(text, ref.metadata())
        ids = [chunk_id(ref.url, d.metadata["chunk_index"]) for d in documents]
        store.add_documents(documents, ids=ids)
        stats["indexed"] += 1
        stats["chunks"] += len(documents)
        print(f"  ok     {label}: {len(documents)} chunks")
    return stats


def main(argv=None, client: Optional[SECClient] = None, processor=None, store=None) -> int:
    parser = argparse.ArgumentParser(description="Download and index SEC 10-K / 10-Q filings.")
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS,
                        help="ticker symbols (default: TICKERS from .env)")
    parser.add_argument("--forms", nargs="+", default=list(config.FORMS), help="form types")
    parser.add_argument("--per-form", type=int, default=None,
                        help="newest filings to take per form (default: FILINGS_10K / FILINGS_10Q)")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    parser.add_argument("--list", action="store_true", help="show indexed filings and exit")
    args = parser.parse_args(argv)

    def open_store():
        if store is not None:
            return store
        from vector_store import SECVectorStore  # loads the embedding model, so import lazily

        return SECVectorStore()

    if args.list:
        info = open_store().get_info()
        print(f"{info['filings']} filings, {info['chunks']} chunks")
        for f in info["filing_list"]:
            print(f"  {f['company']} ({f['ticker']}) {f['form_type']} {f['year']} - {f['chunks']} chunks")
        return 0

    # Check the cheap requirement (SEC contact details) before loading the embedding model.
    try:
        client = client or SECClient()
    except SECProjectError as exc:
        print(f"ERROR: {exc}")
        return 1
    store = open_store()
    if args.reset:
        store.reset()
        print("Index cleared.")

    print("Starting SEC filings indexing...\n")
    processor = processor or SECDocumentProcessor(client)
    forms = [f.upper() for f in args.forms]
    per_form = args.per_form if args.per_form is not None else config.FILINGS_PER_FORM

    refs: List[FilingRef] = []
    for ticker in args.tickers:
        try:
            found = client.list_filings(ticker, forms=forms, per_form=per_form)
        except SECProjectError as exc:
            print(f"{ticker}: {exc}")
            continue
        print(f"{ticker}: {len(found)} filings found")
        refs.extend(found)

    if not refs:
        print("\nNo filings to index. Check the tickers, SEC_USER_AGENT and your internet connection.")
        return 1

    print()
    stats = index_filings(refs, processor, store)

    print("\nIndexing finished.")
    print(f"  indexed now:        {stats['indexed']} filings ({stats['chunks']} chunks)")
    print(f"  already indexed:    {stats['skipped_existing']}")
    print(f"  failed:             {stats['failed']}")
    print(f"  total in database:  {store.count()} chunks")
    if store.count() == 0:
        print("\nNothing is indexed. See the FAIL lines above for the reason.")
        return 1
    print("\nReady. Ask questions with: python query_engine.py   or   uvicorn api:app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
