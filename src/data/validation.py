"""Validation gate for the synthetic dataset. Generation fails rather than shipping bad data.

Every check here is a violation that would quietly poison downstream work: an evidence type
outside the vocabulary, a reason code the rulebook does not declare, an artifact pointing at the
wrong dispute, a case whose transaction sub-type cannot host its reason code, or the same
transaction appearing in both train and test.
"""

from __future__ import annotations

from collections import Counter

from src.data.rulebook_vocab import (
    evidence_type_ids,
    reason_code_ids,
    resolve_reason_code_entry,
)
from src.data.schemas import HardCaseClass, SeedCase, TxnSubType
from src.data.synthetic_dataset import DatasetSplits


class DatasetValidationError(ValueError):
    """Raised when generated data violates an invariant."""


def _check(problems: list[str], condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def validate_cases(cases: list[SeedCase]) -> list[str]:
    """Check every case in isolation. Returns the problems found."""
    problems: list[str] = []
    known_types = evidence_type_ids()
    known_codes = reason_code_ids()

    seen_case_ids: set[str] = set()
    seen_dispute_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()

    for case in cases:
        cid = case.case_id
        _check(problems, cid not in seen_case_ids, f"{cid}: duplicate case_id")
        seen_case_ids.add(cid)

        dispute = case.dispute
        _check(problems, dispute.dispute_id not in seen_dispute_ids, f"{cid}: duplicate dispute_id")
        seen_dispute_ids.add(dispute.dispute_id)

        _check(problems, dispute.reason_code in known_codes, f"{cid}: unknown reason code")
        try:
            _, entry = resolve_reason_code_entry(
                dispute.reason_code, dispute.txn_sub_type.value
            )
        except KeyError as error:
            problems.append(f"{cid}: {error}")
            continue

        for artifact in case.evidence:
            _check(
                problems,
                artifact.dispute_id == dispute.dispute_id,
                f"{cid}/{artifact.evidence_id}: artifact is tied to a different dispute",
            )
            _check(
                problems,
                artifact.type in known_types,
                f"{cid}/{artifact.evidence_id}: evidence type outside the vocabulary",
            )
            _check(
                problems,
                artifact.evidence_id not in seen_evidence_ids,
                f"{cid}/{artifact.evidence_id}: duplicate evidence_id",
            )
            seen_evidence_ids.add(artifact.evidence_id)

        truth = case.ground_truth
        _check(problems, bool(truth.rationale.strip()), f"{cid}: empty rationale")
        _check(problems, bool(truth.source_rule.strip()), f"{cid}: empty source_rule")
        _check(
            problems,
            truth.expected_reason_code == dispute.reason_code,
            f"{cid}: ground-truth reason code disagrees with the record",
        )
        _check(problems, truth.realized_outcome is not None, f"{cid}: missing realized_outcome")
        _check(problems, truth.win_probability is not None, f"{cid}: missing win_probability")

        # Impossible states.
        _check(problems, dispute.amount > 0, f"{cid}: non-positive amount")
        _check(
            problems,
            dispute.chargeback_count_ifsc_acct >= 0 and dispute.chargeback_count_vpa_pair >= 0,
            f"{cid}: negative chargeback counter",
        )
        if dispute.txn_sub_type is not TxnSubType.U2:
            _check(
                problems,
                dispute.payee_merchant_id is None and dispute.mcc is None,
                f"{cid}: P2P transaction carries merchant identifiers",
            )
            _check(
                problems,
                not dispute.acquiring_psp_is_merchant_bank,
                f"{cid}: deemed approval claimed on a P2P transaction",
            )
        if case.hard_case_class is HardCaseClass.SMALL_OFFLINE:
            _check(
                problems,
                dispute.txn_sub_type is TxnSubType.U2,
                f"{cid}: small/offline merchant on a P2P transaction",
            )
    return problems


def validate_distribution(
    cases: list[SeedCase], expected_shares: dict[str, float], tolerance: float
) -> list[str]:
    """Check the realised hard-case mix is within tolerance of the configured mix."""
    problems: list[str] = []
    total = len(cases)
    counts = Counter(case.hard_case_class.value for case in cases)
    for name, expected in expected_shares.items():
        actual = counts.get(name, 0) / total
        if abs(actual - expected) > tolerance:
            problems.append(
                f"distribution: {name} is {actual:.3f}, expected {expected:.3f} "
                f"(tolerance {tolerance})"
            )
    return problems


def validate_no_leakage(splits: DatasetSplits) -> list[str]:
    """No dispute or transaction may appear in more than one split."""
    problems: list[str] = []
    named = {"train": splits.train, "val": splits.val, "test": splits.test}
    for field in ("dispute_id", "txn_id"):
        seen: dict[str, str] = {}
        for split_name, cases in named.items():
            for case in cases:
                value = getattr(case.dispute, field)
                if value in seen and seen[value] != split_name:
                    problems.append(
                        f"leakage: {field} {value} is in both {seen[value]} and {split_name}"
                    )
                seen[value] = split_name

    train_ids = {case.case_id for case in splits.train}
    stray = sorted({case.case_id for case in splits.dev} - train_ids)
    if stray:
        problems.append(f"dev subset must come from train; stray cases: {stray[:5]}")

    if splits.train and splits.test:
        latest_train = max(case.dispute.txn_timestamp for case in splits.train)
        earliest_test = min(case.dispute.txn_timestamp for case in splits.test)
        if earliest_test < latest_train:
            problems.append(
                f"temporal split is not clean: test starts {earliest_test} but train runs to "
                f"{latest_train}"
            )
    return problems


def validate_dataset(
    splits: DatasetSplits, expected_shares: dict[str, float], tolerance: float
) -> None:
    """Run every check and raise on the first set of violations found."""
    everything = splits.train + splits.val + splits.test
    problems = (
        validate_cases(everything)
        + validate_distribution(everything, expected_shares, tolerance)
        + validate_no_leakage(splits)
    )
    if problems:
        listed = "\n  - ".join(problems[:25])
        more = "" if len(problems) <= 25 else f"\n  ... and {len(problems) - 25} more"
        raise DatasetValidationError(
            f"{len(problems)} validation problem(s):\n  - {listed}{more}"
        )
