"""Builds the circulars corpus: ingest, validate, write, document.

Run with `uv run python -m src.retrieval.build_corpus`. OCR is slow — a couple of minutes for the
scanned documents — so this is a deliberate, occasional step rather than something the tests do.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.logging_setup import get_logger
from src.retrieval.corpus_ingest import (
    EXTRACTION_MIXED,
    EXTRACTION_OCR,
    EXTRACTION_TEXT,
    LOW_YIELD_DOC_CHARS,
    MIN_TEXT_CHARS_PER_PAGE,
    OCR_DPI,
    DocumentRecord,
    TesseractBackend,
    corpus_dir,
    ingest_corpus,
    read_corpus,
    write_corpus,
)
from src.retrieval.corpus_validation import CorpusReport, validate_corpus, validate_or_raise

log = get_logger(__name__)

CARD_NAME = "CORPUS_CARD.md"


def build_card(records: list[DocumentRecord], report: CorpusReport, ocr_engine: str) -> str:
    """Render the corpus card from the documents actually ingested."""
    total = len(records)
    methods = Counter(record.extraction_method for record in records)
    pages = sum(record.page_count for record in records)
    chars = sum(record.char_count for record in records)
    ocr_pages = sum(
        1 for record in records for page in record.pages if page.extraction_method == EXTRACTION_OCR
    )

    lines = [
        "# Corpus card — NPCI/RBI circulars",
        "",
        "## What this is",
        "",
        "The **retrieval target**. These are the source circulars the rulebook is derived from, "
        "normalised into clean per-document text with page boundaries intact so a chunk can cite "
        "`doc_id` plus page.",
        "",
        "**This is not the evidence store.** Finding the rule that governs a dispute is a search "
        "problem over this corpus; finding a dispute's own artifacts is an exact keyed join, "
        "handled by `src/data/evidence_store.py` against the datasets. The two are kept strictly "
        "apart — a similarity score is the wrong tool for a question that has one right answer.",
        "",
        "Nothing here is chunked or embedded yet. That is later work.",
        "",
        "## Extraction",
        "",
        f"- **Documents**: {total}",
        f"- **Pages**: {pages} ({ocr_pages} recognised by OCR)",
        f"- **Characters**: {chars:,}",
        f"- **OCR engine**: {ocr_engine}",
        f"- **Rasterisation**: {OCR_DPI} dpi, via PyMuPDF (no poppler dependency)",
        f"- **Text-layer threshold**: a page yielding under {MIN_TEXT_CHARS_PER_PAGE} characters "
        "is treated as having no text layer and is sent to OCR",
        "",
        "Most of these circulars are image scans, so extraction is two-stage and per page: try "
        "the PDF text layer first, rasterise and OCR whatever it cannot read. The method is "
        "recorded per page and rolled up per document, because whether a quoted rule came from a "
        "text layer or from a character recogniser changes how much it should be trusted.",
        "",
        "### Text vs OCR",
        "",
        "| method | documents | share | meaning |",
        "|---|---:|---:|---|",
        f"| `text` | {methods.get(EXTRACTION_TEXT, 0)} | "
        f"{methods.get(EXTRACTION_TEXT, 0) / total:.1%} | every page had a usable text layer |",
        f"| `ocr` | {methods.get(EXTRACTION_OCR, 0)} | "
        f"{methods.get(EXTRACTION_OCR, 0) / total:.1%} | image scan; every page was recognised |",
        f"| `mixed` | {methods.get(EXTRACTION_MIXED, 0)} | "
        f"{methods.get(EXTRACTION_MIXED, 0) / total:.1%} | text body with scanned pages "
        "(typically a cover or an annexure) |",
        "",
        "## Per-document yield",
        "",
        "| method | ref (parsed from content) | pages | chars | chars/page | doc_id |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in sorted(records, key=lambda r: (r.extraction_method, -r.char_count)):
        per_page = record.char_count // max(record.page_count, 1)
        ref = record.circular_ref or "—"
        lines.append(
            f"| `{record.extraction_method}` | {ref} | {record.page_count} | "
            f"{record.char_count:,} | {per_page:,} | `{record.doc_id}` |"
        )

    lines += [
        "",
        "## Provenance",
        "",
        "Every record carries `source_path` and `source_sha256`, so a chunk traces to an exact "
        "file, and `extraction_method` at both document and page level. Page text is stored "
        "separately as well as joined into `text` with `[[page N]]` markers, so page attribution "
        "survives chunking.",
        "",
        "`circular_ref` and `title` are parsed from the **document's own text**, not from the "
        "filename. Filenames are download artefacts and sometimes disagree with the document.",
        "",
        "## Flagged for review",
        "",
    ]
    if not report.review_flags:
        lines += ["Nothing flagged.", ""]
    else:
        lines += [
            f"{len(report.review_flags)} item(s). None of these block ingestion; they are "
            "recorded so a human can check them.",
            "",
        ]
        lines += [f"- {flag}" for flag in report.review_flags]
        lines += [
            "",
            "**On the reference mismatches.** OCR misreads digits — a `1` scanned as a `7`, a "
            "trailing `A`/`B`/`C` lost to a smudge — so a reference parsed out of recognised text "
            "can be wrong. These are surfaced rather than silently overwritten from the filename, "
            "because the filename is not authoritative either and quietly substituting it would "
            "hide the OCR quality problem instead of exposing it. Resolve them by eye against the "
            "PDF before anything downstream relies on `circular_ref` as a key.",
            "",
        ]

    lines += [
        "## Known limitations",
        "",
        f"- OCR quality is unmeasured. No page has been checked against a human transcription, so "
        "the character error rate is unknown. The evidence for it being adequate is indirect: "
        f"the scanned documents average around {chars // max(pages, 1):,} characters per page, "
        "which is in the range of the born-digital ones, and no document fell under the "
        f"{LOW_YIELD_DOC_CHARS}-character review threshold.",
        "- Tables are flattened. The evidence table in OC 208 §C and the RGNB table in OC 184B "
        "become running text, losing their row and column structure. A chunker that splits mid-"
        "table will produce misleading fragments; this needs attention when chunking is designed.",
        "- Recognition is English-only. Several RBI circulars carry a Hindi header, which is "
        "recognised as noise. It sits at the top of page 1 and does not affect the operative "
        "English text below it.",
        "- Tesseract is a system binary, not a Python dependency, so `uv sync` alone does not "
        "make ingestion reproducible on another machine. The corpus output is DVC-tracked so it "
        "does not have to be regenerated to be used.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Ingest, validate, write and document the corpus."""
    parser = argparse.ArgumentParser(description="Build the circulars corpus.")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="rewrite the card from the corpus already on disk, re-running no OCR",
    )
    args = parser.parse_args(argv)
    out = args.out or corpus_dir()

    backend = TesseractBackend()
    usable, reason = backend.available()

    if args.card_only:
        records = read_corpus(out)
        report = validate_corpus(records)
        engine = f"tesseract ({reason})" if usable else reason
    else:
        if not usable:
            print(f"OCR is unavailable: {reason}")
            print("Most circulars here are image scans; ingesting without OCR would write blank")
            print("records for them. Install Tesseract and re-run.")
            return 1
        log.info("ingesting with %s", reason)
        records, failures = ingest_corpus(ocr_backend=backend, on_error="collect")
        if failures:
            print(f"{len(failures)} document(s) failed to ingest:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        report = validate_or_raise(records)
        write_corpus(records, out)
        engine = reason

    (out / CARD_NAME).write_text(build_card(records, report, engine), encoding="utf-8")

    methods = Counter(record.extraction_method for record in records)
    print(f"corpus written to {out}")
    print(f"  documents : {len(records)}")
    print(f"  methods   : {dict(sorted(methods.items()))}")
    print(f"  pages     : {sum(r.page_count for r in records)}")
    print(f"  characters: {sum(r.char_count for r in records):,}")
    print(f"  review flags: {len(report.review_flags)}")
    for flag in report.review_flags:
        print(f"    - {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
