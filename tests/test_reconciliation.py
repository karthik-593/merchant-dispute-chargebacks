"""Tests for the OC 208A OCR reconciliation.

A correction log is only worth having if it is checkable. These tests make it so, in both
directions:

- every `as_scanned` value must really be in the scanned record, so no correction can be invented
  for damage that was not there;
- every `corrected_to` value must really be in the curated record, so no correction can be claimed
  and then not applied;
- every value listed as `clean` must be intact in *both*, so the log is a statement about all 28
  codes rather than only the interesting ones.

Together those mean the log cannot drift from what the corpus actually holds without a test going
red, which is the only reason to trust a provenance trail at all.
"""

from __future__ import annotations

import re

import pytest

from src.data.rulebook_vocab import all_nrp_annexure_codes, load_reject_taxonomy
from src.evaluation.retrieval_metrics import normalise
from src.retrieval.corpus_ingest import CORPUS_FILE, corpus_dir, read_corpus
from src.retrieval.curated_records import (
    OC_208A_DOC_ID,
    build_oc_208a_reject_taxonomy,
    load_reconciliation,
)

CORPUS_PATH = corpus_dir() / CORPUS_FILE
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="ingested corpus not on disk (dvc pull)"
)


@pytest.fixture(scope="module")
def reconciliation():
    return load_reconciliation()


@pytest.fixture(scope="module")
def curated_text():
    return normalise(build_oc_208a_reject_taxonomy().text)


@pytest.fixture(scope="module")
def scanned_text():
    """The Annexure A page of the scan, which is what the reconciliation is about.

    Deliberately one page rather than the whole document. Page 3 holds Annexure B's arbitration
    lifecycle table, which cites this block as the range "1132 to 1157"; matching a code there
    would count a cross-reference as a table row and mask the fact that 1132's row is gone.
    """
    meta = load_reconciliation()["meta"]
    record = next((r for r in read_corpus() if r.doc_id == meta["superseded_doc_id"]), None)
    if record is None:
        pytest.skip(f"superseded record {meta['superseded_doc_id']} not in the corpus")
    page = next(p for p in record.pages if p.page == meta["annexure_page"])
    return normalise(page.text)


# --- the log covers the whole annexure ----------------------------------------------------------


def test_every_code_is_accounted_for_exactly_once(reconciliation):
    """28 codes, each either corrected or declared clean. No silent third category."""
    corrected = [c["code"] for c in reconciliation["corrections"]]
    clean = list(reconciliation["clean"])
    assert len(corrected) == len(set(corrected)), "a code is corrected twice"
    assert not set(corrected) & set(clean), "a code cannot be both corrected and clean"
    assert set(corrected) | set(clean) == set(all_nrp_annexure_codes())


def test_the_log_names_its_authority_and_both_records(reconciliation):
    meta = reconciliation["meta"]
    assert meta["authority"] == "configs/rulebook/reject_taxonomy.yaml"
    assert meta["curated_doc_id"] == OC_208A_DOC_ID
    assert meta["superseded_doc_id"]
    assert meta["circular"]


def test_every_correction_cites_a_source(reconciliation):
    for correction in reconciliation["corrections"]:
        assert correction["source"], correction["code"]
        assert "reject_taxonomy.yaml" in correction["source"], correction["code"]
        assert correction["damage"], correction["code"]
        assert correction["change"], correction["code"]


# --- the corrections describe damage that was really there --------------------------------------


@needs_corpus
def test_every_as_scanned_value_is_actually_in_the_scan(reconciliation, scanned_text):
    """A correction with no matching damage in the scan would be an invented one."""
    for correction in reconciliation["corrections"]:
        as_scanned = correction["as_scanned"]
        if as_scanned is None:
            continue
        assert normalise(as_scanned) in scanned_text, (
            f"{correction['code']}: as_scanned {as_scanned!r} is not in the scanned record"
        )


@needs_corpus
def test_what_the_log_calls_missing_really_is_missing(reconciliation, scanned_text):
    """A [missing] entry claims the scan lost a specific part. Check it lost exactly that part.

    The distinction is the point: 1145 kept its description and lost its code, 1136 and 1148 kept
    their code and lost their description, 1132 lost both. Checking only "something is missing"
    would pass on a log that got the repair backwards.
    """
    book = all_nrp_annexure_codes()
    for correction in reconciliation["corrections"]:
        if "missing" not in correction["damage"]:
            assert "absent" not in correction, f"{correction['code']}: absent on a non-missing row"
            continue
        code = correction["code"]
        absent = correction["absent"]
        assert absent in {"row", "description", "code"}, code
        verified = normalise(book[code]["description"])
        if absent in {"row", "description"}:
            assert verified not in scanned_text, (
                f"{code}: {absent} claimed missing, but the description is in the scan"
            )
        if absent in {"row", "code"}:
            assert not re.search(rf"(?<!\d){code}(?!\d)", scanned_text), (
                f"{code}: {absent} claimed missing, but the code is in the scan"
            )
        if absent == "description" and "code" not in correction["damage"]:
            # 1136 and 1148 kept a correct code and lost only their text. 1154 is excluded here
            # because its code was *also* misread (1184), so the right code is not in the scan
            # either - which `as_scanned: "1184"` already pins down.
            assert re.search(rf"(?<!\d){code}(?!\d)", scanned_text), (
                f"{code}: only the description was lost, so the code should still be there"
            )
        if absent == "code":
            assert verified in scanned_text, (
                f"{code}: only the code was lost, so the description should still be there"
            )


# --- the corrections were actually applied ------------------------------------------------------


def test_every_corrected_value_is_in_the_curated_record(reconciliation, curated_text):
    for correction in reconciliation["corrections"]:
        corrected = normalise(" ".join(correction["corrected_to"].split()))
        # The log writes a restored row as "<code> - <description>"; the record renders it as
        # "Reason code <code> - <description>". Compare on the part that carries the meaning.
        needle = corrected.split(" - ", 1)[-1] if corrected[:4].isdigit() else corrected
        assert needle in curated_text, (
            f"{correction['code']}: corrected_to {needle!r} is not in the curated record"
        )


def test_no_corrupted_value_survives_into_the_curated_record(reconciliation, curated_text):
    """The whole point. If `Ilagible` is still there, nothing was fixed."""
    corrupted = ["ilagible", "goods/serces", "tan details", "catego:", "4146", "4147", "4156"]
    for token in corrupted:
        assert token not in curated_text, f"{token!r} survived into the curated record"
    # 1184 was a misread of 1154; neither the wrong code nor a bare 1184 belongs in the record.
    assert "1184" not in curated_text


def test_the_curated_record_carries_every_code_with_its_verified_description(curated_text):
    for code, entry in all_nrp_annexure_codes().items():
        assert f"reason code {code} -" in curated_text, code
        assert normalise(" ".join(entry["description"].split())) in curated_text, code


def test_rows_restored_from_the_rulebook_are_present(curated_text):
    """The four rows the scan lost entirely are the reason this record exists."""
    for code, fragment in [
        ("1132", "rejection document says merchant accepting dispute"),
        ("1136", "cbs screenshot uploaded without txn details"),
        ("1148", "others"),
        ("1154", "the refund has failed as per the refund details"),
    ]:
        assert f"reason code {code} -" in curated_text, code
        assert fragment in curated_text, code


# --- clean rows were left alone -----------------------------------------------------------------


@needs_corpus
def test_clean_rows_are_intact_in_both_records(reconciliation, scanned_text, curated_text):
    """A row declared clean must read the same in the scan and in the curated record."""
    book = all_nrp_annexure_codes()
    for code in reconciliation["clean"]:
        verified = normalise(" ".join(book[code]["description"].split()))
        assert verified in scanned_text, f"{code}: declared clean but not intact in the scan"
        assert verified in curated_text, f"{code}: declared clean but missing from the record"


# --- the record supersedes rather than replaces --------------------------------------------------


def test_the_curated_record_declares_what_it_supersedes(reconciliation):
    record = build_oc_208a_reject_taxonomy()
    assert record.doc_id == OC_208A_DOC_ID
    assert record.supersedes_doc_id == reconciliation["meta"]["superseded_doc_id"]
    assert record.extraction_method == "curated"
    assert record.source_path.endswith(".pdf"), "provenance must point at the real circular"


@needs_corpus
def test_the_scanned_record_is_kept_not_overwritten(reconciliation):
    """Superseded, never deleted. The damaged original stays auditable."""
    doc_ids = {r.doc_id for r in read_corpus()}
    assert reconciliation["meta"]["superseded_doc_id"] in doc_ids
    assert OC_208A_DOC_ID in doc_ids


def test_the_rulebook_is_the_authority_not_this_module():
    """Change the rulebook and the record changes with it; nothing here is hand-keyed."""
    taxonomy = load_reject_taxonomy()
    text = build_oc_208a_reject_taxonomy().text
    entries = taxonomy["nrp_verdict_for_codes"] + taxonomy["nrp_verdict_against_codes"]
    for entry in entries:
        assert " ".join(entry["description"].split()) in " ".join(text.split())
