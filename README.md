# SEC Filing Summarizer & Verified Q&A System (RAG)

Ask questions about real SEC 10-K and 10-Q filings and get a cited answer, a plain-English
summary, and a **claim percentage** that measures how much of the answer's evidence was
actually found word for word in the filing.

## How it works

```
index_documents.py                          query_engine.py / api.py
SEC EDGAR  ->  download 10-K / 10-Q   ->    question
   |           (sec_downloader.py)             |
   v                                           v
clean text (tables kept)  ->  chunks   ->   retrieve best chunks (Chroma, filtered by
(document_processor.py)    ->  embeddings    company / form / year)
                              (MiniLM)          |
                                 |              v
                                 v          Gemini answers ONLY from those chunks and
                            Chroma database  must quote its evidence (rag_chain.py)
                            (chroma_db/)        |
                                                v
                                          every quote is checked against the chunk text
                                          -> claim percentage + sources
```

**Claim percentage** = quoted evidence found in the retrieved filing text / quotes given.
100% = Satisfied, 1-99% = Partially satisfied, 0% = Not satisfied (including "not found").
It checks that the quotes are real; it cannot prove the model reasoned about them correctly,
so still read the sources.

## Setup (Python 3.10+)

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set:

* `GOOGLE_API_KEY` - free Gemini key from https://aistudio.google.com/apikey
* `SEC_USER_AGENT` - your real name and email, e.g. `"Jane Doe jane.doe@gmail.com"`.
  SEC blocks anonymous scripts, so this is required.

## 1. Index filings (needs internet, run once)

```bash
python index_documents.py                          # AAPL, MSFT, TSLA: 2 newest 10-K + 2 newest 10-Q each
python index_documents.py --tickers NVDA AMZN      # pick companies
python index_documents.py --forms 10-K --per-form 3
python index_documents.py --list                   # what is indexed
python index_documents.py --reset                  # start over
```

The first run downloads the embedding model (about 90 MB). Re-running is safe: finished
filings are skipped, interrupted ones are completed, nothing is duplicated.

## 2. Ask questions

Command line:

```bash
python query_engine.py                                              # interactive
python query_engine.py -c AAPL -f 10-K -y 2024 -q "What were total net sales?"
```

API:

```bash
uvicorn api:app --reload        # interactive docs at http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What were total net sales?", "company": "AAPL", "form_type": "10-K", "year": 2024}'
```

Company can be a ticker (`AAPL`), part of the name (`apple`) or a CIK. Company, form type and
year are all optional. `year` is the fiscal year the filing reports on.

Endpoints: `GET /` , `GET /health`, `GET /info` (what is indexed), `POST /ask`.
Errors: 400 ambiguous company, 404 no matching filing, 422 bad input, 502 model failure,
503 nothing indexed or no API key.

## Example output

```
ANSWER:
Total net sales were $391,035 million in fiscal 2024 [S1].

SUMMARY:
Apple's revenue for the year ended September 2024, mostly from iPhone and services.

EVIDENCE CHECK:
  [verified] (S1) "Total net sales 391,035 383,285 394,328"

SOURCES USED:
  [S1] Apple Inc. 10-K 2024 (filed 2024-11-01) - https://www.sec.gov/Archives/edgar/...

CLAIM PERCENTAGE: 100% (Satisfied) - share of quoted evidence found word for word in the filing text
```

(Illustrative layout; real figures come from the filings you index.)

## Tests

```bash
pytest
```

The tests run offline with stand-ins for the SEC, the embedding model and Gemini, so they need
no keys or internet.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Settings and `.env` loading |
| `sec_downloader.py` | Finds and downloads filings from EDGAR politely (throttled, retries) |
| `document_processor.py` | HTML/PDF to clean text (tables kept), chunking |
| `vector_store.py` | Chroma database, filters, duplicate-safe writes |
| `rag_chain.py` | Retrieval, Gemini call, evidence verification, claim percentage |
| `index_documents.py` | Indexing command |
| `query_engine.py` | Question command |
| `api.py` | FastAPI service |
| `errors.py` | Error types |
| `tests/` | Offline test suite |

## Limits

* The downloader reads EDGAR's "recent filings" list, so very old filings of a company may not be found.
* Answers come from the top retrieved passages only, so broad questions ("summarize the whole
  filing") see just a few passages. Ask specific questions.
* Not financial advice; check figures against the filing.

## Data

Real filings are downloaded from SEC EDGAR. The earlier Kaggle CSV contained mostly contract
exhibits, repeated rows and no 10-K or 10-Q text, so it is no longer used.
