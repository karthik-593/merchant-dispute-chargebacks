"""Tests for the two stores: the keyed evidence store and the circulars corpus.

The corpus tests read what ingestion already produced rather than re-running OCR, which takes
minutes. They skip when the DVC-tracked file is not present, so a fresh clone without a `dvc
pull` still gets a green suite instead of a misleading failure.
"""

from __future__ import annotations

import pytest

from src.data.evidence_store import (
    SEED_SOURCE,
    SYNTHETIC_SOURCE,
    EvidenceStore,
)
from src.data.schemas import SeedCase
from src.data.seed_loader import load_seed_cases
from src.data.synthetic_dataset import read_jsonl, synthetic_dir
from src.decision.sufficiency import assess
from src.decision.verifier import StubVerifier
from src.retrieval.corpus_ingest import (
    CORPUS_FILE,
    EXTRACTION_BLANK,
    EXTRACTION_MIXED,
    EXTRACTION_OCR,
    EXTRACTION_TEXT,
    PAGE_MARKER_PATTERN,
    corpus_dir,
    extract_document,
    read_corpus,
    source_pdfs,
)
from src.retrieval.corpus_validation import filename_hint, validate_corpus

SYNTHETIC_TRAIN = synthetic_dir() / "train.jsonl"
CORPUS_PATH = corpus_dir() / CORPUS_FILE

needs_synthetic = pytest.mark.skipif(
    not SYNTHETIC_TRAIN.is_file(), reason="synthetic split not on disk (dvc pull)"
)
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="ingested corpus not on disk (dvc pull)"
)


# --- evidence store: a keyed lookup, not a search ---------------------------------------------


@pytest.fixture(scope="module")
def seed_cases() -> list[SeedCase]:
    return load_seed_cases()


@pytest.fixture(scope="module")
def seed_store(seed_cases) -> EvidenceStore:
    return EvidenceStore.from_cases(seed_cases, source=SEED_SOURCE)


def test_seed_dispute_returns_its_own_artifacts(seed_store, seed_cases):
    """A known seed dispute must come back with exactly the artifacts filed against it."""
    case = next(c for c in seed_cases if c.case_id == "seed_001")
    dispute_id = case.dispute.dispute_id

    assert seed_store.get_dispute(dispute_id) == case.dispute
    artifacts = seed_store.get_evidence(dispute_id)
    assert [a.evidence_id for a in artifacts] == [a.evidence_id for a in case.evidence]
    assert all(a.dispute_id == dispute_id for a in artifacts)


def test_every_seed_dispute_round_trips(seed_store, seed_cases):
    for case in seed_cases:
        dispute_id = case.dispute.dispute_id
        assert seed_store.get_dispute(dispute_id) == case.dispute
        assert seed_store.get_evidence(dispute_id) == list(case.evidence)


def test_a_case_with_no_evidence_returns_an_empty_list(seed_store, seed_cases):
    empty = next(c for c in seed_cases if not c.evidence)
    assert seed_store.get_evidence(empty.dispute.dispute_id) == []


def test_unknown_dispute_raises(seed_store):
    with pytest.raises(KeyError, match="not in the evidence store"):
        seed_store.get_evidence("NOT-A-DISPUTE")
    with pytest.raises(KeyError):
        seed_store.get_dispute("NOT-A-DISPUTE")


def test_duplicate_dispute_ids_are_refused(seed_cases):
    store = EvidenceStore.from_cases(seed_cases[:1], source=SEED_SOURCE)
    with pytest.raises(ValueError, match="duplicate dispute_id"):
        store.add_cases(seed_cases[:1], source=SYNTHETIC_SOURCE)


def test_store_feeds_the_engine_the_same_verdict(seed_store, seed_cases):
    """The engine reading through the store must reach what it reaches reading the case."""
    verifier = StubVerifier()
    for case in seed_cases:
        direct, _ = assess(case.dispute, case.evidence, verifier)
        through_store, _ = assess(
            seed_store.get_dispute(case.dispute.dispute_id),
            seed_store.get_evidence(case.dispute.dispute_id),
            verifier,
        )
        assert direct.level is through_store.level, case.case_id
        assert direct.accepted_evidence_ids == through_store.accepted_evidence_ids


def test_evidence_by_type_is_an_exact_filter(seed_store):
    artifacts = seed_store.evidence_by_type("acquirer_declaration_letter")
    assert artifacts
    assert {a.type for a in artifacts} == {"acquirer_declaration_letter"}


@needs_synthetic
def test_synthetic_dispute_returns_its_own_artifacts():
    """A known synthetic dispute must resolve the same way as a seed one."""
    cases = read_jsonl(SYNTHETIC_TRAIN)
    store = EvidenceStore.from_cases(cases, source=SYNTHETIC_SOURCE, split="train")
    case = cases[0]
    dispute_id = case.dispute.dispute_id

    assert store.get_dispute(dispute_id) == case.dispute
    assert store.get_evidence(dispute_id) == list(case.evidence)
    assert store.get(dispute_id).source == SYNTHETIC_SOURCE
    assert store.get(dispute_id).split == "train"


@needs_synthetic
def test_loading_both_datasets_keeps_them_distinguishable():
    store = EvidenceStore.load()
    stats = store.stats()
    assert stats["by_source"][SEED_SOURCE] == 40
    assert stats["by_source"][SYNTHETIC_SOURCE] > 0
    assert len(store.dispute_ids(source=SEED_SOURCE)) == 40
    assert set(store.dispute_ids(source=SEED_SOURCE)).isdisjoint(
        store.dispute_ids(source=SYNTHETIC_SOURCE)
    )


def test_store_holds_no_ranking_or_embedding_surface():
    """The API is a lookup. If a scoring method appears here, the stores have been conflated."""
    surface = {name for name in dir(EvidenceStore) if not name.startswith("_")}
    assert not surface & {"search", "query", "rank", "similarity", "embed", "nearest", "top_k"}


# --- circulars corpus: the retrieval target ----------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return read_corpus()


def test_extracting_a_born_digital_circular_needs_no_ocr():
    """Runs without OCR, so it exercises the text path on every machine."""
    path = next(p for p in source_pdfs() if p.name.startswith("UPI_OC_No_213"))
    record = extract_document(path)
    assert record.extraction_method == EXTRACTION_TEXT
    assert record.char_count > 500
    assert record.page_count == len(record.pages)
    assert record.source_sha256
    assert PAGE_MARKER_PATTERN.search(record.text)


@needs_corpus
def test_one_non_empty_record_per_circular(corpus):
    assert len(corpus) == len(source_pdfs())
    for record in corpus:
        assert record.text.strip(), record.doc_id
        assert record.char_count > 0, record.doc_id
        assert record.page_count > 0, record.doc_id
        assert record.extraction_method in {EXTRACTION_TEXT, EXTRACTION_OCR, EXTRACTION_MIXED}


@needs_corpus
def test_every_page_carries_a_marker_and_a_method(corpus):
    for record in corpus:
        markers = [int(n) for n in PAGE_MARKER_PATTERN.findall(record.text)]
        assert markers == list(range(1, record.page_count + 1)), record.doc_id
        for page in record.pages:
            assert page.extraction_method, f"{record.doc_id} p{page.page}"
            # Only a page recognised as blank in the source may be empty.
            if page.extraction_method == EXTRACTION_BLANK:
                assert page.char_count == 0, f"{record.doc_id} p{page.page}"
            else:
                assert page.text.strip(), f"{record.doc_id} p{page.page}"


@needs_corpus
def test_oc_208_was_ocred_and_yields_substantial_text(corpus):
    """OC 208 is the core dispute circular and is a pure image scan.

    If this ever comes back as a text extraction, or thin, the corpus has silently lost the
    evidence table the whole sufficiency engine is derived from.
    """
    record = next(r for r in corpus if r.doc_id.startswith("upi_oc_no_208_fy_24_25"))
    assert record.extraction_method == EXTRACTION_OCR
    assert all(p.extraction_method == EXTRACTION_OCR for p in record.pages)
    assert record.page_count == 8
    assert record.char_count > 10000, "OC 208 OCR'd to far less text than its 8 pages should hold"
    assert "evidence" in record.text.lower()


@needs_corpus
def test_scanned_documents_yield_comparable_density_to_born_digital_ones(corpus):
    """A crude OCR sanity check: recognised pages should not be far thinner than typed ones."""

    def density(method: str) -> float:
        pages = [p for r in corpus if r.extraction_method == method for p in r.pages]
        return sum(p.char_count for p in pages) / len(pages)

    assert density(EXTRACTION_OCR) > 0.5 * density(EXTRACTION_TEXT)


@needs_corpus
def test_corpus_validation_reports_no_blocking_problems(corpus):
    report = validate_corpus(corpus)
    assert report.ok, report.problems


@needs_corpus
def test_no_document_is_low_yield(corpus):
    assert [r.doc_id for r in corpus if r.is_low_yield] == []


@needs_corpus
def test_provenance_is_complete(corpus):
    for record in corpus:
        assert record.source_path.startswith("refs/circulars/"), record.doc_id
        assert len(record.source_sha256) == 64, record.doc_id


def test_filename_hint_is_only_a_cross_check():
    assert filename_hint("refs/circulars/UPI_OC_No_208_A_FY_2025_26.pdf") == "OC 208A"
    assert filename_hint("refs/circulars/no_number_here.pdf") is None


@needs_corpus
def test_blank_pages_are_marked_blank_not_treated_as_ocr_failures(corpus):
    """The Ombudsman Scheme booklet has genuinely blank pages.

    Marking them `blank` is what separates "this page has no ink" from "OCR read nothing off a
    page that has content" — the second is a lost rule and must stay a hard failure.
    """
    record = next(r for r in corpus if r.doc_id == "osdt31012019")
    blank = [p.page for p in record.pages if p.extraction_method == EXTRACTION_BLANK]
    assert blank == [4, 6]
    assert record.extraction_method == EXTRACTION_MIXED
    assert all(p.char_count == 0 for p in record.pages if p.page in blank)


@needs_corpus
def test_no_document_rolls_up_to_blank(corpus):
    """A document that is blank end to end would mean extraction produced nothing at all."""
    assert [r.doc_id for r in corpus if r.extraction_method == EXTRACTION_BLANK] == []
