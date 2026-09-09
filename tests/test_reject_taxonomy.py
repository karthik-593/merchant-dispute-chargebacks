"""Tests for the NRP verdict reason codes transcribed from OC 208A Annexure A."""

from __future__ import annotations

import pytest

from src.data.rulebook_vocab import load_reject_taxonomy, nrp_verdict_against_codes

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
    assert taxonomy["nrp_verdict_against_codes"]
    assert taxonomy["nrp_verdict_for_codes"]


def test_every_entry_has_a_code_description_and_source(taxonomy):
    entries = taxonomy["nrp_verdict_for_codes"] + taxonomy["nrp_verdict_against_codes"]
    for entry in entries:
        assert entry["code"].isdigit(), entry
        assert entry["description"].strip(), entry["code"]
        assert entry["source"].strip(), entry["code"]
        assert "OC 208A" in entry["source"], entry["code"]


def test_codes_are_unique_and_contiguous(taxonomy):
    """Both bands together still cover 1130-1157 with no gap and no duplicate."""
    codes = [
        entry["code"]
        for entry in taxonomy["nrp_verdict_for_codes"] + taxonomy["nrp_verdict_against_codes"]
    ]
    assert len(codes) == len(set(codes))
    assert sorted(codes) == [str(n) for n in range(1130, 1158)]


def test_the_core_invalid_reasons_are_present_and_say_what_they_should(taxonomy):
    """Each code the verifier keys on must exist and describe the condition it is meant to catch."""
    by_code = nrp_verdict_against_codes()
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
    entries = taxonomy["nrp_verdict_for_codes"] + taxonomy["nrp_verdict_against_codes"]
    flagged = [e for e in entries if "transcription_note" in e]
    assert flagged, "expected at least one entry to record its uncertainty"
    for entry in flagged:
        assert entry["transcription_note"].strip(), entry["code"]


# --- NVB / NVR category boundary --------------------------------------------------------------
#
# This is the one behavioural error the source-verification pass found, and the guard that keeps
# it fixed. Annexure A's Adj Flag column has three categories; an earlier revision scoped the
# catalog "1130-1157" and treated every code in it as a fail. 1130 and 1131 are NVB - they record
# a representment that WON - so that scoping would have made the verifier reject valid
# representments on the strength of the codes that say they succeeded.


def test_the_fail_catalog_is_nvr_1132_to_1157_only():
    """The NVR band, and nothing outside it, is the auto-loss catalog."""
    from src.data.rulebook_vocab import nrp_fail_codes

    codes = nrp_fail_codes()
    assert min(codes) == "1132"
    assert max(codes) == "1157"
    assert all("1132" <= c <= "1157" for c in codes), sorted(codes)


def test_no_downstream_component_treats_nvb_codes_as_a_fail():
    """1126-1131 are verdict-FOR. If this fails, the verifier will reject valid representments."""
    from src.data.rulebook_vocab import (
        nrp_fail_codes,
        nrp_verdict_against_codes,
        nrp_verdict_for_codes,
    )

    nvb = {f"{c}" for c in range(1126, 1132)}
    fails = nrp_fail_codes()
    assert not (nvb & fails), f"NVB codes folded into the fail catalog: {sorted(nvb & fails)}"
    assert not (nvb & set(nrp_verdict_against_codes()))
    assert set(nrp_verdict_for_codes()) <= nvb
    assert "1130" in nrp_verdict_for_codes()
    assert "1131" in nrp_verdict_for_codes()


def test_the_two_bands_never_overlap():
    from src.data.rulebook_vocab import nrp_verdict_against_codes, nrp_verdict_for_codes

    assert not set(nrp_verdict_for_codes()) & set(nrp_verdict_against_codes())


def test_verdict_for_codes_are_labelled_and_sourced():
    from src.data.rulebook_vocab import nrp_verdict_for_codes

    for code, entry in nrp_verdict_for_codes().items():
        assert entry["verdict"] == "for_beneficiary", code
        assert "NVB" in entry["source"], code
        assert "verified against source PDF" in entry["source"], code


def test_the_ranges_are_pinned_in_the_config_itself():
    """A future edit that widens the fail band has to change this too, and explain why."""
    from src.data.rulebook_vocab import load_reject_taxonomy

    meta = load_reject_taxonomy()["meta"]
    assert meta["nvr_range"] == ["1132", "1157"]
    assert meta["nvb_range"] == ["1126", "1131"]
    assert "verified against source PDF" in meta["category_boundary_source"]


def test_1102_is_recorded_as_removed_by_oc_208a_not_merely_absent():
    """Absent by intent. Without this, a later pass would 'restore' it from OC 208."""
    from src.data.rulebook_vocab import load_reject_taxonomy

    removed = {r["code"]: r for r in load_reject_taxonomy()["meta"]["removed_codes"]}
    assert "1102" in removed
    assert removed["1102"]["removed_by"] == "OC 208A"
    assert "OC 208" in removed["1102"]["previously"]
    from src.data.rulebook_vocab import nrp_fail_codes

    assert "1102" not in nrp_fail_codes()


def test_1132_uses_the_source_verified_wording():
    """'says', not 'of'. The earlier reading was a guess the source pass resolved."""
    from src.data.rulebook_vocab import nrp_verdict_against_codes

    entry = nrp_verdict_against_codes()["1132"]
    assert entry["description"] == "Rejection document says merchant accepting dispute"
    assert "verified against source PDF" in entry["source"]
