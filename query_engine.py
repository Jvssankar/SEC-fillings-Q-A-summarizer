"""Command-line question answering over the indexed SEC filings.

Interactive:   python query_engine.py
One question:  python query_engine.py -c AAPL -f 10-K -y 2024 -q "What were total net sales?"
"""
import argparse
import sys
from typing import Optional

from errors import SECProjectError
from rag_chain import STATUS_LABELS, AskResult, SECRAGChain


def format_result(result: AskResult) -> str:
    lines = [
        "",
        "ANSWER:",
        result.answer,
        "",
        "SUMMARY:",
        result.summary or "(none)",
        "",
        "EVIDENCE CHECK:",
    ]
    if result.evidence:
        for item in result.evidence:
            mark = "verified" if item.verified else "NOT FOUND in filing"
            src = f"S{item.source_id}" if item.source_id else "S?"
            lines.append(f'  [{mark}] ({src}) "{item.quote}"')
    else:
        lines.append("  (no quotes to check)")
    lines += ["", "SOURCES USED:"]
    for s in result.sources:
        lines.append(f"  [S{s.id}] {s.company} {s.form_type} {s.year} (filed {s.filed_at}) - {s.url}")
    lines += [
        "",
        f"CLAIM PERCENTAGE: {result.claim_percentage}% ({STATUS_LABELS[result.status]})"
        " - share of quoted evidence found word for word in the filing text",
    ]
    if result.note:
        lines.append(f"NOTE: {result.note}")
    lines.append("-" * 60)
    return "\n".join(lines)


def ask_question(company: Optional[str], form_type: Optional[str], year: Optional[str],
                 question: str, chain: Optional[SECRAGChain] = None) -> AskResult:
    """Ask one question and return the result (printing is done by the caller)."""
    chain = chain or SECRAGChain()
    return chain.ask(question, company=company, form_type=form_type, year=year)


def _interactive(chain: SECRAGChain) -> int:
    print("SEC Filing Q&A ready. Leave company / form / year empty to search every indexed filing.\n")
    while True:
        company = input("Company ticker or name (or 'exit'): ").strip()
        if company.lower() == "exit":
            return 0
        form_type = input("Form type (10-K / 10-Q, optional): ").strip()
        year = input("Fiscal year YYYY (optional): ").strip()
        question = input("Question: ").strip()
        try:
            print(format_result(ask_question(company, form_type, year, question, chain)))
        except SECProjectError as exc:
            print(f"\nCould not answer: {exc}\n")
        except ValueError as exc:
            print(f"\n{exc}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about indexed SEC filings.")
    parser.add_argument("-c", "--company", default="", help="ticker or company name")
    parser.add_argument("-f", "--form", default="", help="10-K or 10-Q")
    parser.add_argument("-y", "--year", default="", help="fiscal year, e.g. 2024")
    parser.add_argument("-q", "--question", default="", help="question (omit for interactive mode)")
    args = parser.parse_args(argv)

    chain = SECRAGChain()
    if not args.question:
        return _interactive(chain)
    try:
        print(format_result(ask_question(args.company, args.form, args.year, args.question, chain)))
        return 0
    except (SECProjectError, ValueError) as exc:
        print(f"Could not answer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
