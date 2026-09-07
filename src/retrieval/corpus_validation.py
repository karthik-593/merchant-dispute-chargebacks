"""Validation gate for the circulars corpus. Ingestion fails rather than shipping a blank rule.

The failure this exists to prevent is quiet: a scanned circular that extracts to nothing still
produces a record, and a corpus with an empty OC 208 in it will confidently retrieve nothing about
the evidence table while looking complete. Every document must yield real text, and the method
that produced it must be recorded.

Some findings are flagged for a human rather than raised. OCR misreads digits, so a parsed
circular reference can be wrong in ways only a person can adjudicate — those are surfaced, never
silently corrected from the filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.retrieval.corpus_ingest import (
    EXTRACTION_BLANK,
    EXTRACTION_MIXED,
    EXTRACTION_OCR,
    EXTRACTION_TEXT,
    LOW_YIELD_DOC_CHARS,
    PAGE_MARKER_PATTERN,
    DocumentRecord,
    source_pdfs,
)

VALID_METHODS = {EXTRACTION_TEXT, EXTRACTION_OCR, EXTRACTION_MIXED, EXTRACTION_BLANK}
# Methods a whole document may be rolled up to. A document that is nothing but blank pages
# would mean every page failed to yield anything, which is a problem rather than a document.
VALID_DOCUMENT_METHODS = {EXTRACTION_TEXT, EXTRACTION_OCR, EXTRACTION_MIXED}

# Only used to cross-check a parsed reference. Never used as the reference itself.
FILENAME_HINT = re.compile(r"OC[_\- ]*(?:No[_\-. ]*)*([0-9]{2,3})[_\- ]*([A-C])?", re.I)


class CorpusValidationError(ValueError):
    """Raised when the ingested corpus violates an invariant."""


@dataclass
class CorpusReport:
    """What validation found: hard problems, and things a human should look at."""

    problems: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found."""
        return not self.problems


def filename_hint(path: str) -> str | None:
    """The circular number suggested by a filename, for cross-checking only."""
    match = FILENAME_HINT.search(Path(path).name)
    if not match:
        return None
    suffix = (match.group(2) or "").upper()
    return f"OC {match.group(1)}{suffix}"


def validate_corpus(
    records: list[DocumentRecord], source_directory: Path | None = None
) -> CorpusReport:
    """Check the corpus is complete, non-empty and traceable."""
    report = CorpusReport()
    sources = source_pdfs(source_directory)

    # Every circular must produce exactly one record.
    by_source = {Path(record.source_path).name: record for record in records}
    for path in sources:
        if path.name not in by_source:
            report.problems.append(f"{path.name}: no record was produced")
    extra = set(by_source) - {path.name for path in sources}
    for name in sorted(extra):
        report.problems.append(f"{name}: record has no matching source PDF")

    seen_ids: set[str] = set()
    for record in records:
        doc = record.doc_id
        if doc in seen_ids:
            report.problems.append(f"{doc}: duplicate doc_id")
        seen_ids.add(doc)

        if not record.text.strip():
            report.problems.append(f"{doc}: empty text")
        if record.char_count == 0:
            report.problems.append(f"{doc}: zero characters extracted")
        if record.extraction_method not in VALID_DOCUMENT_METHODS:
            report.problems.append(
                f"{doc}: extraction_method {record.extraction_method!r} is not one of "
                f"{sorted(VALID_DOCUMENT_METHODS)}"
            )
        if record.page_count == 0:
            report.problems.append(f"{doc}: no pages")

        # Page markers must be present and complete, or a later chunk cannot cite a page.
        markers = [int(n) for n in PAGE_MARKER_PATTERN.findall(record.text)]
        if markers != list(range(1, record.page_count + 1)):
            report.problems.append(
                f"{doc}: page markers {markers[:5]}... do not cover pages 1..{record.page_count}"
            )

        for page in record.pages:
            if page.extraction_method not in VALID_METHODS:
                report.problems.append(
                    f"{doc} p{page.page}: extraction_method {page.extraction_method!r} is invalid"
                )
            # An empty page is fine only when it was recognised as genuinely blank. Empty
            # text on a page that has ink on it means content was lost.
            if not page.text.strip() and page.extraction_method != EXTRACTION_BLANK:
                report.problems.append(
                    f"{doc} p{page.page}: empty text from a page marked "
                    f"{page.extraction_method!r}, which should have yielded content"
                )

        # Findings for a human, not blockers.
        blank_pages = [p.page for p in record.pages if p.extraction_method == EXTRACTION_BLANK]
        if blank_pages:
            report.review_flags.append(
                f"{doc}: page(s) {blank_pages} are blank in the source document (no ink), so "
                f"they carry no text by design"
            )
        if record.is_low_yield:
            report.review_flags.append(
                f"{doc}: only {record.char_count} chars over {record.page_count} page(s) "
                f"(< {LOW_YIELD_DOC_CHARS}); the scan may have OCR'd to near-nothing"
            )
        if record.circular_ref is None:
            report.review_flags.append(
                f"{doc}: no circular reference could be parsed from the document text"
            )
        else:
            hint = filename_hint(record.source_path)
            if hint and hint.replace(" ", "") != record.circular_ref.replace(" ", ""):
                report.review_flags.append(
                    f"{doc}: parsed reference {record.circular_ref!r} disagrees with the "
                    f"filename hint {hint!r}; OCR misreads digits, so confirm by eye"
                )
    return report


def validate_or_raise(
    records: list[DocumentRecord], source_directory: Path | None = None
) -> CorpusReport:
    """Validate and raise on any blocking problem."""
    report = validate_corpus(records, source_directory)
    if not report.ok:
        listed = "\n  - ".join(report.problems[:25])
        more = "" if len(report.problems) <= 25 else f"\n  ... and {len(report.problems) - 25} more"
        raise CorpusValidationError(
            f"{len(report.problems)} corpus problem(s):\n  - {listed}{more}"
        )
    return report
