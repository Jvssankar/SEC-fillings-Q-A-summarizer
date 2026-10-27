"""FastAPI service.  Run with:  uvicorn api:app --reload   (docs at /docs)"""
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from errors import (
    AmbiguousCompanyError,
    IndexEmptyError,
    LLMError,
    LLMNotConfiguredError,
    NoFilingsFoundError,
)
from rag_chain import AskResult, SECRAGChain

app = FastAPI(
    title="SEC Filing Q&A System",
    description="Ask questions about 10-K / 10-Q filings and get verified, cited answers.",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, description="Your question about the filings")
    company: Optional[str] = Field(default=None, description="Ticker or company name, e.g. AAPL")
    form_type: Optional[str] = Field(default=None, description="10-K or 10-Q")
    year: Optional[int] = Field(default=None, ge=1990, le=2100, description="Fiscal year, e.g. 2024")
    top_k: int = Field(default=config.TOP_K, ge=1, le=20, description="Passages to retrieve")


@lru_cache(maxsize=1)
def get_chain() -> SECRAGChain:
    """One shared chain (loads the embedding model once, on first use)."""
    return SECRAGChain()


@app.get("/")
def root():
    return {"message": "SEC Filing Q&A System running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/info")
def info(chain: SECRAGChain = Depends(get_chain)):
    return chain.store.get_info()


@app.post("/ask", response_model=AskResult)
def ask(req: QuestionRequest, chain: SECRAGChain = Depends(get_chain)):
    try:
        return chain.ask(
            req.question,
            company=req.company,
            form_type=req.form_type,
            year=req.year,
            top_k=req.top_k,
        )
    except (IndexEmptyError, LLMNotConfiguredError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except NoFilingsFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except AmbiguousCompanyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
