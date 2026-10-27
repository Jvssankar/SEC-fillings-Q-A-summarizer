"""Exceptions shared across the project so callers (CLI, API) can react precisely."""


class SECProjectError(Exception):
    """Base class for every error raised on purpose by this project."""


class SECDownloadError(SECProjectError):
    """A request to SEC EDGAR failed or returned unusable data."""


class IndexEmptyError(SECProjectError):
    """The vector database has no filings yet; run index_documents.py first."""


class NoFilingsFoundError(SECProjectError):
    """No indexed filing matches the company / form / year the user asked for."""


class AmbiguousCompanyError(SECProjectError):
    """The company text matches more than one indexed company."""


class LLMNotConfiguredError(SECProjectError):
    """GOOGLE_API_KEY is missing."""


class LLMError(SECProjectError):
    """The language model call failed or returned nothing usable."""
