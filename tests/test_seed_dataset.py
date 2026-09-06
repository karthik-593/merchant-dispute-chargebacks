"""Structural tests for the hand-built seed-v1 fixture set."""

from __future__ import annotations

import pytest

from src.data.rulebook_vocab import (
    evidence_type_ids,
    merchant_type_branch_ids,
    reason_code_entries,
    reason_code_ids,
    txn_sub_type_ids,
)
from src.data.schemas import (
    SEED_VERSION,
    Decision,
    DisputeRecord,
    HardCaseClass,
    MerchantType,
    Sufficiency,
    TxnSubType,
)
from src.data.seed_loader import load_seed_cases, seed_files

# The record-level merchant_type values map onto the rulebook's declaration-rule branches.
MERCHANT_TYPE_TO_RULEBOOK_BRANCH = {
    MerchantType.SMALL_OFFLINE: "small_or_offline",
    MerchantType.LARGE: "large",
}


@pytest.fixture(scope="module")
def cases():
    return load_seed_cases()


def resolve_rulebook_entry(reason_code: str, txn_sub_type: TxnSubType) -> str:
    """Find the single rulebook entry governing a (reason code, sub-type) pair."""
    matches = [
        key
        for key, entry in reason_code_entries().items()
        if entry.get("code", key) == reason_code
        and txn_sub_type.value in entry.get("txn_sub_type", [])
    ]
    assert len(matches) == 1, f"{reason_code}/{txn_sub_type.value} resolved to {matches}"
    return matches[0]


def test_seed_files_present():
    files = seed_files()
    assert 30 <= len(files) <= 40, f"seed set should hold 30-40 cases, found {len(files)}"


def test_all_files_parse_into_models(cases):
    assert len(cases) == len(seed_files())
    assert all(case.version == SEED_VERSION for case in cases)


def test_identifiers_are_unique(cases):
    for label, values in (
        ("case_id", [c.case_id for c in cases]),
        ("dispute_id", [c.dispute.dispute_id for c in cases]),
        ("rrn", [c.dispute.rrn for c in cases]),
        ("evidence_id", [a.evidence_id for c in cases for a in c.evidence]),
    ):
        assert len(values) == len(set(values)), f"duplicate {label} in the seed set"


def test_every_reason_code_exists_in_the_rulebook(cases):
    known = reason_code_ids()
    for case in cases:
        assert case.dispute.reason_code in known, case.case_id
        assert case.ground_truth.expected_reason_code in known, case.case_id


def test_every_case_resolves_to_exactly_one_rulebook_entry(cases):
    for case in cases:
        resolve_rulebook_entry(case.dispute.reason_code, case.dispute.txn_sub_type)


def test_every_rulebook_entry_is_exercised(cases):
    exercised = {
        resolve_rulebook_entry(case.dispute.reason_code, case.dispute.txn_sub_type)
        for case in cases
    }
    missing = sorted(set(reason_code_entries()) - exercised)
    assert not missing, f"rulebook entries with no seed case: {missing}"


def test_every_evidence_type_is_in_the_controlled_vocabulary(cases):
    known = evidence_type_ids()
    for case in cases:
        for artifact in case.evidence:
            assert artifact.type in known, f"{case.case_id}/{artifact.evidence_id}"


def test_every_hard_case_class_is_represented(cases):
    seen = {case.hard_case_class for case in cases}
    missing = sorted(c.value for c in HardCaseClass if c not in seen)
    assert not missing, f"hard-case classes with no seed case: {missing}"


def test_ground_truth_fields_are_populated(cases):
    for case in cases:
        truth = case.ground_truth
        assert isinstance(truth.expected_sufficiency, Sufficiency), case.case_id
        assert isinstance(truth.expected_decision, Decision), case.case_id
        assert truth.rationale.strip(), case.case_id
        assert truth.source_rule.strip(), case.case_id
        assert "OC" in truth.source_rule, f"{case.case_id}: source_rule cites no circular"


def test_evidence_artifacts_are_hand_labelled(cases):
    for case in cases:
        for artifact in case.evidence:
            assert artifact.validity_note.strip(), f"{case.case_id}/{artifact.evidence_id}"
            assert artifact.description.strip(), f"{case.case_id}/{artifact.evidence_id}"


def test_narratives_are_present(cases):
    for case in cases:
        assert len(case.narrative.split()) >= 8, f"{case.case_id}: narrative too thin"


def test_enums_match_the_rulebook_vocabulary():
    assert {t.value for t in TxnSubType} == set(txn_sub_type_ids())
    branches = merchant_type_branch_ids()
    for merchant_type, branch in MERCHANT_TYPE_TO_RULEBOOK_BRANCH.items():
        assert branch in branches, f"{merchant_type.value} maps to unknown branch {branch!r}"


def test_absent_cases_carry_no_valid_accepted_evidence(cases):
    """A case labelled ABSENT must not also be labelled as holding valid required evidence."""
    for case in cases:
        if case.ground_truth.expected_sufficiency is not Sufficiency.ABSENT:
            continue
        entry_key = resolve_rulebook_entry(case.dispute.reason_code, case.dispute.txn_sub_type)
        accepted = set(reason_code_entries()[entry_key]["required_evidence_any_of"])
        offending = [a.evidence_id for a in case.evidence if a.type in accepted and a.is_valid]
        assert not offending, (
            f"{case.case_id} is labelled ABSENT but holds valid accepted evidence {offending}"
        )


# Cases where the acquiring PSP and the merchant bank are one institution, so the
# deemed-approval presumption of delivery applies.
DEEMED_APPROVAL_MATCHING = {"seed_036", "seed_037"}
# The contrast: same P2M shape, two different institutions, so no presumption arises.
DEEMED_APPROVAL_CONTRAST = {"seed_038"}


def test_deemed_approval_condition_is_true_where_institutions_match(cases):
    by_id = {c.case_id: c for c in cases}
    assert set(by_id) >= DEEMED_APPROVAL_MATCHING, "expected deemed-approval cases are missing"
    for case_id in sorted(DEEMED_APPROVAL_MATCHING):
        dispute = by_id[case_id].dispute
        assert dispute.txn_sub_type is TxnSubType.U2, case_id
        assert dispute.acquiring_psp == dispute.beneficiary_bank, case_id
        assert dispute.acquiring_psp_is_merchant_bank is True, case_id


def test_deemed_approval_condition_is_false_for_the_contrast_case(cases):
    by_id = {c.case_id: c for c in cases}
    assert set(by_id) >= DEEMED_APPROVAL_CONTRAST, "expected contrast case is missing"
    for case_id in sorted(DEEMED_APPROVAL_CONTRAST):
        dispute = by_id[case_id].dispute
        assert dispute.txn_sub_type is TxnSubType.U2, case_id
        assert dispute.acquiring_psp is not None and dispute.beneficiary_bank is not None, case_id
        assert dispute.acquiring_psp != dispute.beneficiary_bank, case_id
        assert dispute.acquiring_psp_is_merchant_bank is False, case_id


def test_deemed_approval_condition_is_false_when_fields_are_unset(cases):
    unset = [
        case
        for case in cases
        if case.dispute.acquiring_psp is None and case.dispute.beneficiary_bank is None
    ]
    assert unset, "expected most cases to leave the institution fields unset"
    for case in unset:
        assert case.dispute.acquiring_psp_is_merchant_bank is False, case.case_id


def test_deemed_approval_condition_is_false_for_p2p_even_if_institutions_match(cases):
    p2p = next(c for c in cases if c.dispute.txn_sub_type is not TxnSubType.U2)
    same_bank = p2p.dispute.model_copy(
        update={"acquiring_psp": "Same Bank Ltd", "beneficiary_bank": "Same Bank Ltd"}
    )
    assert same_bank.acquiring_psp_is_merchant_bank is False


def test_deemed_approval_condition_is_derived_not_stored():
    assert "acquiring_psp_is_merchant_bank" not in DisputeRecord.model_fields
    for field in ("acquiring_psp", "beneficiary_bank"):
        assert field in DisputeRecord.model_fields
