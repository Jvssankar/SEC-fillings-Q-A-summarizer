"""Retrieval-augmented question answering over indexed SEC filings.

Flow: resolve company/form/year -> retrieve chunks -> ask Gemini for a structured
answer -> VERIFY every quote the model gives against the retrieved text.
The claim percentage is the share of quoted evidence that was found word for word
in the filing text, so it measures grounding instead of guessing from answer length.
"""
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field

import config
from errors import (
    AmbiguousCompanyError,
    IndexEmptyError,
    LLMError,
    LLMNotConfiguredError,
    NoFilingsFoundError,
)

NOT_FOUND_TEXT = "Not found in the retrieved filing excerpts."

SYSTEM_PROMPT = """You are a careful financial analyst answering questions about SEC filings (10-K and 10-Q).
Use ONLY the numbered sources provided. Never use outside knowledge.

Rules:
- If the sources do not contain the answer, set found=false and answer exactly: "Not found in the retrieved filing excerpts."
- Otherwise set found=true and answer concisely and precisely. Include figures with units and the period they cover, and mention the source like [S2].
- summary: 2-3 plain-English sentences that explain the relevant excerpts to a non-expert. Do not just repeat the answer.
- evidence: for each key fact in the answer give the source number and a SHORT quote (at most 25 words) copied EXACTLY, character for character, from that source."""


# ------------------------------------------------------------ data models
class EvidenceQuote(BaseModel):
    source: int = Field(description="Number N of the [SN] source the quote comes from")
    quote: str = Field(description="Short exact quote from that source (max 25 words)")


class LLMAnswer(BaseModel):
    """Structured output requested from the language model."""

    found: bool = Field(description="True only if the sources contain the answer")
    answer: str
    summary: str
    evidence: List[EvidenceQuote] = Field(default_factory=list)


class SourceInfo(BaseModel):
    id: int
    company: str
    ticker: str
    form_type: str
    year: str
    filed_at: str
    url: str
    snippet: str


class EvidenceItem(BaseModel):
    source_id: Optional[int]
    quote: str
    verified: bool


class AskResult(BaseModel):
    question: str
    answer: str
    summary: str
    found: bool
    status: Literal["satisfied", "partially_satisfied", "not_satisfied"]
    claim_percentage: int
    evidence: List[EvidenceItem]
    sources: List[SourceInfo]
    filters: Dict[str, str]
    note: Optional[str] = None


def inline_schema(model: type) -> Dict[str, Any]:
    """JSON schema of a pydantic model with every $ref/$defs expanded in place.

    Some model back-ends handle self-contained schemas more reliably than schemas
    that point at '#/$defs/...', so we send the flattened form.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


STATUS_LABELS = {
    "satisfied": "Satisfied",
    "partially_satisfied": "Partially satisfied",
    "not_satisfied": "Not satisfied",
}


# ---------------------------------------------------------------- helpers
def normalize_form_type(form: Optional[str]) -> Optional[str]:
    if not form or not form.strip():
        return None
    compact = re.sub(r"[\s_]+", "", form.strip().upper())
    if re.fullmatch(r"10-?K", compact):
        return "10-K"
    if re.fullmatch(r"10-?Q", compact):
        return "10-Q"
    return compact


def _tokens(text: str) -> str:
    """Lower-case alphanumeric tokens joined by spaces (ignores punctuation/spacing)."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def quote_in_text(quote: str, text: str, min_tokens: int = 2) -> bool:
    q = _tokens(quote)
    if len(q.split()) < min_tokens:
        return False
    return q in _tokens(text)


def _score(evidence: List[EvidenceItem], found: bool) -> Tuple[int, str, Optional[str]]:
    if not found:
        return 0, "not_satisfied", None
    if not evidence:
        return 0, "not_satisfied", "The model gave an answer but no quotes to verify it against the filing."
    verified = sum(1 for e in evidence if e.verified)
    pct = round(100 * verified / len(evidence))
    if pct == 100:
        return pct, "satisfied", None
    if pct > 0:
        return pct, "partially_satisfied", "Some quoted evidence could not be found in the filing text."
    return 0, "not_satisfied", "None of the quoted evidence could be found in the filing text."


# ------------------------------------------------------------------ chain
class SECRAGChain:
    def __init__(self, store=None, llm=None):
        self._store = store
        self._llm = llm

    @property
    def store(self):
        if self._store is None:
            from vector_store import SECVectorStore

            self._store = SECVectorStore()
        return self._store

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    @staticmethod
    def _build_llm():
        key = config.google_api_key()
        if not key:
            raise LLMNotConfiguredError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=key,
            temperature=config.LLM_TEMPERATURE,
            max_retries=config.LLM_MAX_RETRIES,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        # A dict comes back from the model; _invoke validates it into LLMAnswer.
        return model.with_structured_output(inline_schema(LLMAnswer))

    # ------------------------------------------------- company resolution
    @staticmethod
    def _resolve_company(text: str, filings: List[Dict[str, Any]]) -> str:
        """Map 'AAPL', 'apple' or a CIK to the CIK stored in the index."""
        wanted = text.strip()
        up, low = wanted.upper(), wanted.lower()
        for f in filings:
            if f["ticker"].upper() == up or str(f["cik"]) == wanted.lstrip("0"):
                return str(f["cik"])
        matches = {str(f["cik"]): f["company"] for f in filings if low in f["company"].lower()}
        if len(matches) == 1:
            return next(iter(matches))
        if not matches:
            known = sorted({f"{f['company']} ({f['ticker'] or f['cik']})" for f in filings})
            raise NoFilingsFoundError(
                f"No indexed company matches '{text}'. Indexed companies: {', '.join(known)}."
            )
        raise AmbiguousCompanyError(
            f"'{text}' matches several companies: {', '.join(sorted(matches.values()))}. Use a ticker."
        )

    # ------------------------------------------------------------- ask
    def ask(self, question: str, company: Optional[str] = None, form_type: Optional[str] = None,
            year: Optional[Any] = None, top_k: Optional[int] = None) -> AskResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("The question must not be empty.")

        # Check the cheap requirement (API key) before loading the embedding model.
        llm = self.llm

        store = self.store
        if store.count() == 0:
            raise IndexEmptyError("No filings are indexed yet. Run: python index_documents.py")

        filings = store.list_filings()
        filters: Dict[str, str] = {}
        if company and company.strip():
            filters["cik"] = self._resolve_company(company, filings)
        form = normalize_form_type(form_type)
        if form:
            filters["form_type"] = form
        if year not in (None, "") and str(year).strip():
            filters["year"] = str(year).strip()

        docs = store.retrieve(question, k=top_k or config.TOP_K, filters=filters)
        docs = self._dedupe(docs)
        if not docs:
            raise NoFilingsFoundError(self._no_match_message(filters, filings))

        context, used = self._build_context(docs, config.MAX_CONTEXT_CHARS)
        raw = self._invoke(llm, question, context)
        return self._assemble(question, raw, used, filters)

    # --------------------------------------------------------- internals
    @staticmethod
    def _dedupe(docs: List[Document]) -> List[Document]:
        seen, unique = set(), []
        for doc in docs:
            key = _tokens(doc.page_content)
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    @staticmethod
    def _no_match_message(filters: Dict[str, str], filings: List[Dict[str, Any]]) -> str:
        indexed = sorted({
            f"{f['company']} {f['form_type']} {f['year']}" for f in filings
        })
        shown = "; ".join(indexed[:25]) + (" ..." if len(indexed) > 25 else "")
        return f"No indexed filing matches {filters}. Indexed filings: {shown}"

    @staticmethod
    def _build_context(docs: List[Document], budget: int) -> Tuple[str, List[Document]]:
        parts: List[str] = []
        used: List[Document] = []
        total = 0
        for doc in docs:
            m = doc.metadata
            header = (f"[S{len(used) + 1}] {m.get('company', '?')} ({m.get('ticker') or 'no ticker'}) "
                      f"{m.get('form_type', '')} for {m.get('year', '')}, filed {m.get('filed_at', '')}")
            block = f"{header}\n{doc.page_content.strip()}"
            if used and total + len(block) > budget:
                break
            parts.append(block[:budget])
            used.append(doc)
            total += len(block)
        return "\n\n".join(parts), used

    @staticmethod
    def _invoke(llm, question: str, context: str) -> LLMAnswer:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Sources:\n{context}\n\nQuestion: {question}"),
        ]
        try:
            result = llm.invoke(messages)
        except Exception as exc:
            raise LLMError(f"The language model request failed: {exc}") from exc
        if isinstance(result, dict):
            try:
                result = LLMAnswer(**result)
            except Exception as exc:
                raise LLMError(f"The language model returned an unexpected format: {exc}") from exc
        if not isinstance(result, LLMAnswer):
            raise LLMError("The language model returned an empty or unreadable response.")
        return result

    @staticmethod
    def _assemble(question: str, raw: LLMAnswer, used: List[Document],
                  filters: Dict[str, str]) -> AskResult:
        found = bool(raw.found and raw.answer.strip())

        evidence: List[EvidenceItem] = []
        if found:
            for item in raw.evidence:
                idx = item.source - 1
                if 0 <= idx < len(used):
                    candidates = [used[idx]]
                else:  # model cited a source number that does not exist: search all of them
                    candidates = used
                ok = any(quote_in_text(item.quote, d.page_content) for d in candidates)
                evidence.append(EvidenceItem(
                    source_id=item.source if 0 < item.source <= len(used) else None,
                    quote=item.quote.strip(),
                    verified=ok,
                ))

        pct, status, note = _score(evidence, found)
        sources = []
        for i, doc in enumerate(used, 1):
            m = doc.metadata
            sources.append(SourceInfo(
                id=i,
                company=str(m.get("company", "")),
                ticker=str(m.get("ticker", "")),
                form_type=str(m.get("form_type", "")),
                year=str(m.get("year", "")),
                filed_at=str(m.get("filed_at", "")),
                url=str(m.get("source_url", "")),
                snippet=doc.page_content.strip()[:300],
            ))
        return AskResult(
            question=question,
            answer=raw.answer.strip() if found else NOT_FOUND_TEXT,
            summary=raw.summary.strip(),
            found=found,
            status=status,
            claim_percentage=pct,
            evidence=evidence,
            sources=sources,
            filters=filters,
            note=note,
        )
