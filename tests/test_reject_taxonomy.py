"""Tests for the NRP verdict reason codes transcribed from OC 208A Annexure A."""

from __future__ import annotations

import pytest

from src.data.rulebook_vocab import load_reject_taxonomy, nrp_verdict_reason_codes

# The codes the verifier's hard gates are built around: a submission tripping one of these loses.
CORE_INVALID_REASONS = {
    "1133": "blurry",
    "1138": "blank",
    "1140": "illegible",
    "1141": "does not address",
    "1142": "generic statements",
    "1144": "refund details",
    "1146": "door-delivery",
    "1147": "merchant details",
    "1149": "balance",
    "1150": "closed/freeze/lien",
    "1151": "debit consent",
    "1152": "not in contact",
    "1153": "black-listed",
    "1155": "invoice",
    "1156": "cancelled/pending/failed",
    "1157": "does not pertain",
}


@pytest.fixture(scope="module")
def taxonomy():
    return load_reject_taxonomy()


def test_taxonomy_loads_and_is_no_longer_a_stub(taxonomy):
    assert taxonomy["meta"]["status"] == "active"
    assert taxonomy["nrp_verdict_reason_codes"]


def test_every_entry_has_a_code_description_and_source(taxonomy):
    for entry in taxonomy["nrp_verdict_reason_codes"]:
        assert entry["code"].isdigit(), entry
        assert entry["description"].strip(), entry["code"]
        assert entry["source"].strip(), entry["code"]
        assert "OC 208A" in entry["source"], entry["code"]


def test_codes_are_unique_and_contiguous(taxonomy):
    codes = [entry["code"] for entry in taxonomy["nrp_verdict_reason_codes"]]
    assert len(codes) == len(set(codes))
    assert codes == [str(n) for n in range(1130, 1158)]


def test_the_core_invalid_reasons_are_present_and_say_what_they_should(taxonomy):
    """Each code the verifier keys on must exist and describe the condition it is meant to catch."""
    by_code = nrp_verdict_reason_codes()
    for code, fragment in CORE_INVALID_REASONS.items():
        assert code in by_code, f"{code} is missing from the taxonomy"
        description = " ".join(by_code[code]["description"].split()).lower()
        assert fragment.lower() in description, f"{code}: {description!r} lacks {fragment!r}"


def test_transcription_provenance_is_recorded(taxonomy):
    """OCR corrections must be written down, not made silently."""
    transcription = taxonomy["meta"]["transcription"]
    assert transcription["corrections"], "digit corrections should be recorded"
    assert transcription["not_transcribed"], "codes left out should be named"
    assert "OCR" in transcription["method"]


def test_uncertain_wording_is_flagged_rather_than_smoothed_over(taxonomy):
    """An entry OCR could not read cleanly carries a note saying so."""
    flagged = [e for e in taxonomy["nrp_verdict_reason_codes"] if "transcription_note" in e]
    assert flagged, "expected at least one entry to record its uncertainty"
    for entry in flagged:
        assert entry["transcription_note"].strip(), entry["code"]
