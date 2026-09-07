"""Tests for the synthetic generator, its validation gate, and generator/engine sync."""

from __future__ import annotations

from collections import Counter

import pytest

from src.data.generator import (
    DATASET_VERSION,
    OracleVerifier,
    SyntheticGenerator,
    load_generator_config,
)
from src.data.rulebook_vocab import evidence_type_ids, reason_code_ids, resolve_reason_code_entry
from src.data.schemas import (
    Decision,
    HardCaseClass,
    MerchantType,
    RealizedOutcome,
    Sufficiency,
)
from src.data.seed_loader import load_seed_cases
from src.data.synthetic_dataset import temporal_split
from src.data.validation import (
    DatasetValidationError,
    validate_cases,
    validate_dataset,
    validate_no_leakage,
)
from src.decision.engine import DecisionEngine
from src.evaluation.generator_sync import check_engine_agreement, check_seed_reproduction

SAMPLE_SIZE = 400


@pytest.fixture(scope="module")
def config():
    return load_generator_config()


@pytest.fixture(scope="module")
def generated():
    return SyntheticGenerator().generate(SAMPLE_SIZE)


@pytest.fixture(scope="module")
def cases(generated):
    return [item.case for item in generated]


# --- reproducibility -------------------------------------------------------------------------


def test_same_seed_reproduces_the_same_cases():
    first = SyntheticGenerator(seed=7).generate(60)
    second = SyntheticGenerator(seed=7).generate(60)
    assert [c.case.model_dump_json() for c in first] == [c.case.model_dump_json() for c in second]


def test_different_seeds_produce_different_cases():
    first = SyntheticGenerator(seed=7).generate(60)
    second = SyntheticGenerator(seed=8).generate(60)
    assert [c.case.model_dump_json() for c in first] != [c.case.model_dump_json() for c in second]


# --- the generator invents nothing ------------------------------------------------------------


def test_every_evidence_type_comes_from_the_controlled_vocabulary(cases):
    vocabulary = evidence_type_ids()
    for case in cases:
        for artifact in case.evidence:
            assert artifact.type in vocabulary, f"{case.case_id}: {artifact.type}"


def test_every_reason_code_is_declared_by_the_rulebook(cases):
    known = reason_code_ids()
    for case in cases:
        assert case.dispute.reason_code in known, case.case_id
        resolve_reason_code_entry(case.dispute.reason_code, case.dispute.txn_sub_type.value)


def test_config_rejects_evidence_outside_the_vocabulary(config):
    broken = {**config, "evidence": {**config["evidence"], "refund_types": ["not_a_real_type"]}}
    with pytest.raises(ValueError, match="outside the vocabulary"):
        SyntheticGenerator(config=broken)


# --- the sync points -------------------------------------------------------------------------


def test_capped_cases_respect_the_cap_convention(cases):
    """Counters at or past a limit trigger; one below does not; fraud is exempt."""
    from src.data.rulebook_vocab import load_caps

    limits = {c["counter_field"]: c["limit"] for c in load_caps()["caps"].values()}
    for case in cases:
        dispute = case.dispute
        breached = any(getattr(dispute, field) >= limit for field, limit in limits.items())
        expect_rgnb = breached and not dispute.fraud_flag
        assert (case.ground_truth.expected_decision is Decision.RGNB) == expect_rgnb, case.case_id


def test_cap_boundary_cases_are_present(cases):
    """The distribution must contain cases sitting exactly one below each limit."""
    from src.data.rulebook_vocab import load_caps

    caps = load_caps()["caps"]
    cd1 = caps["CD1"]["limit"]
    cd2 = caps["CD2"]["limit"]
    boundary = [
        case
        for case in cases
        if case.dispute.chargeback_count_ifsc_acct == cd1 - 1
        and case.dispute.chargeback_count_vpa_pair == cd2 - 1
    ]
    assert boundary, "no cap-boundary cases were generated"
    for case in boundary:
        assert case.ground_truth.expected_decision is not Decision.RGNB, case.case_id


def test_declaration_substitutes_only_for_small_offline_merchants(cases):
    from src.data.rulebook_vocab import merchant_type_rule

    declaration = merchant_type_rule()["small_or_offline"]["substitute_evidence"]
    for case in cases:
        only_declaration = bool(case.evidence) and all(a.type == declaration for a in case.evidence)
        if not only_declaration:
            continue
        expected = (
            Sufficiency.STRONG
            if case.dispute.merchant_type is MerchantType.SMALL_OFFLINE
            else Sufficiency.ABSENT
        )
        assert case.ground_truth.expected_sufficiency is expected, case.case_id


def test_deemed_approval_never_changes_the_outcome(cases):
    """The presumption is metadata; plain rules decide. Removing it must change nothing."""
    engine = DecisionEngine()
    flagged = [c for c in cases if c.dispute.acquiring_psp_is_merchant_bank]
    assert flagged, "no deemed-approval cases were generated"
    for case in flagged:
        with_flag = engine.decide(case.dispute, case.evidence)
        stripped = case.dispute.model_copy(update={"acquiring_psp": None, "beneficiary_bank": None})
        without = engine.decide(stripped, case.evidence)
        assert with_flag.decision is without.decision, case.case_id
        assert with_flag.sufficiency.level is without.sufficiency.level, case.case_id


def test_contradictory_cases_are_two_valid_artifacts_labelled_escalate(generated):
    contradictory = [g for g in generated if g.contradictions]
    assert contradictory, "no contradiction cases were generated"
    for item in contradictory:
        assert item.case.hard_case_class is HardCaseClass.CONTRADICTORY
        assert len(item.case.evidence) >= 2
        assert all(a.is_valid for a in item.case.evidence), item.case.case_id
        assert item.case.ground_truth.expected_decision is Decision.ESCALATE


# --- the stochastic label --------------------------------------------------------------------


def test_realized_outcome_is_populated_and_probabilistic(cases):
    outcomes = Counter(c.ground_truth.realized_outcome for c in cases)
    assert set(outcomes) == {RealizedOutcome.WON, RealizedOutcome.LOST}
    assert min(outcomes.values()) > 0.05 * len(cases), "outcome is nearly degenerate"


def test_win_probability_never_reaches_certainty(cases, config):
    floor = config["noise_model"]["floor"]
    ceiling = config["noise_model"]["ceiling"]
    for case in cases:
        p = case.ground_truth.win_probability
        assert floor <= p <= ceiling, f"{case.case_id}: {p}"


def test_outcome_is_not_a_deterministic_function_of_sufficiency(cases):
    """Both outcomes must occur within each sufficiency level, or the target is trivial."""
    by_level: dict[Sufficiency, set[RealizedOutcome]] = {}
    for case in cases:
        by_level.setdefault(case.ground_truth.expected_sufficiency, set()).add(
            case.ground_truth.realized_outcome
        )
    for level, seen in by_level.items():
        assert len(seen) == 2, f"{level.value} always resolves to {seen}"


def test_stronger_evidence_wins_more_often(cases):
    """The noise must not wash out the signal it is layered onto."""
    rate = {}
    for level in (Sufficiency.STRONG, Sufficiency.WEAK, Sufficiency.ABSENT):
        subset = [c for c in cases if c.ground_truth.expected_sufficiency is level]
        won = sum(1 for c in subset if c.ground_truth.realized_outcome is RealizedOutcome.WON)
        rate[level] = won / len(subset)
    assert rate[Sufficiency.STRONG] > rate[Sufficiency.WEAK] > rate[Sufficiency.ABSENT]


def test_outcome_does_not_change_the_deterministic_decision(cases):
    """A WON and a LOST case with the same sufficiency get the same decision."""
    engine = DecisionEngine()
    for case in cases:
        outcome = engine.decide(case.dispute, case.evidence)
        if case.ground_truth.expected_decision is Decision.ESCALATE:
            continue  # B0 cannot see contradictions; covered by the sync check
        assert outcome.decision is case.ground_truth.expected_decision, case.case_id


# --- validation gate -------------------------------------------------------------------------


def test_generated_data_passes_validation(cases, config):
    splits = temporal_split(cases, config["split"], dev_size=20, dev_seed=1)
    validate_dataset(
        splits,
        expected_shares=config["hard_case_class_shares"],
        tolerance=0.06,  # a 400-case sample is noisier than the full 2000
    )


def test_validation_catches_a_mismatched_evidence_dispute_id(cases):
    case = cases[next(i for i, c in enumerate(cases) if c.evidence)]
    broken = case.model_copy(
        update={
            "evidence": [case.evidence[0].model_copy(update={"dispute_id": "SYN-999999"})],
        }
    )
    problems = validate_cases([broken])
    assert any("tied to a different dispute" in p for p in problems)


def test_validation_catches_leakage(cases, config):
    splits = temporal_split(cases, config["split"], dev_size=20, dev_seed=1)
    leaked = splits.__class__(
        train=splits.train,
        val=splits.val,
        test=splits.test + [splits.train[0]],
        dev=splits.dev,
    )
    problems = validate_no_leakage(leaked)
    assert any("leakage" in p for p in problems)


def test_validation_raises_on_a_distribution_that_is_off(cases, config):
    splits = temporal_split(cases, config["split"], dev_size=20, dev_seed=1)
    with pytest.raises(DatasetValidationError, match="distribution"):
        validate_dataset(splits, expected_shares=config["hard_case_class_shares"], tolerance=0.0001)


# --- splits ----------------------------------------------------------------------------------


def test_temporal_split_is_ordered_and_disjoint(cases, config):
    splits = temporal_split(cases, config["split"], dev_size=20, dev_seed=1)
    assert splits.sizes["train"] + splits.sizes["val"] + splits.sizes["test"] == len(cases)
    latest_train = max(c.dispute.txn_timestamp for c in splits.train)
    earliest_test = min(c.dispute.txn_timestamp for c in splits.test)
    assert earliest_test >= latest_train
    assert {c.case_id for c in splits.dev} <= {c.case_id for c in splits.train}


# --- the two consistency checks ----------------------------------------------------------------


def test_check_a_engine_agrees_with_the_generated_labels(generated):
    result = check_engine_agreement(generated)
    assert result.sufficiency_matches == result.total, result.summary()
    assert not result.unexpected, result.summary()
    assert result.expected_decision_gap > 0, "expected some contradiction cases in the sample"
    assert result.decision_matches + result.expected_decision_gap == result.total
    assert result.passed


def test_check_b_generator_reproduces_the_hand_written_seed_labels():
    result = check_seed_reproduction()
    assert result.total == len(load_seed_cases())
    assert result.sufficiency_matches == result.total, result.summary()
    assert result.decision_matches == result.total, result.summary()
    assert result.passed


def test_oracle_verifier_differs_from_the_stub_only_on_contradictions(generated):
    item = next(g for g in generated if g.contradictions)
    oracle = OracleVerifier({item.case.dispute.dispute_id: item.contradictions})
    stub_engine = DecisionEngine()
    oracle_engine = DecisionEngine(verifier=oracle)
    stub = stub_engine.decide(item.case.dispute, item.case.evidence)
    with_oracle = oracle_engine.decide(item.case.dispute, item.case.evidence)
    assert stub.sufficiency.level is with_oracle.sufficiency.level
    assert stub.decision is Decision.FILE
    assert with_oracle.decision is Decision.ESCALATE


def test_dataset_version_is_distinct_from_the_seed_set(cases):
    assert DATASET_VERSION == "dataset-v1.0"
    assert all(case.version == DATASET_VERSION for case in cases)
    assert all(case.version == "seed-v1" for case in load_seed_cases())
