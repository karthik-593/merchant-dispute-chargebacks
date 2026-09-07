"""End-to-end tests for the B0 vertical slice over the seed-v1 fixtures."""

from __future__ import annotations

import inspect

import pytest

from src.data.schemas import Decision, Sufficiency
from src.data.seed_loader import load_seed_cases
from src.decision.caps import evaluate_caps
from src.decision.engine import DecisionEngine
from src.decision.verifier import StubVerifier
from src.evaluation.run_seed import format_report, run
from src.llm.classifier import PassthroughClassifier

# Representative case per behaviour the deterministic thread must get right.
EXPECTED_OUTCOMES = [
    ("seed_001", Sufficiency.STRONG, Decision.FILE, "valid POD satisfies RC 1064"),
    ("seed_007", Sufficiency.ABSENT, Decision.CONCEDE, "no evidence at all"),
    ("seed_012", Sufficiency.WEAK, Decision.CONCEDE, "invoice does not match the transaction"),
    ("seed_020", Sufficiency.WEAK, Decision.CONCEDE, "refund initiated, never completed"),
    ("seed_030", Sufficiency.STRONG, Decision.FILE, "small/offline declaration substitutes"),
    ("seed_032", Sufficiency.ABSENT, Decision.CONCEDE, "declaration does not substitute, large"),
    ("seed_033", Sufficiency.STRONG, Decision.RGNB, "CD1 cap reached"),
    ("seed_034", Sufficiency.STRONG, Decision.RGNB, "CD2 cap reached"),
    ("seed_035", Sufficiency.STRONG, Decision.FILE, "over both caps but fraud-exempt"),
    ("seed_037", Sufficiency.ABSENT, Decision.CONCEDE, "deemed approval cannot be rebutted"),
    ("seed_039", Sufficiency.STRONG, Decision.FILE, "CD1 boundary, last allowed chargeback"),
    ("seed_040", Sufficiency.STRONG, Decision.FILE, "CD2 boundary, last allowed chargeback"),
]

CONTRADICTORY_CASES = {"seed_016", "seed_017", "seed_018", "seed_019"}


@pytest.fixture(scope="module")
def cases_by_id():
    return {case.case_id: case for case in load_seed_cases()}


@pytest.fixture(scope="module")
def engine():
    return DecisionEngine()


@pytest.fixture(scope="module")
def scores():
    return run()


@pytest.mark.parametrize(
    ("case_id", "expected_sufficiency", "expected_decision", "why"),
    EXPECTED_OUTCOMES,
    ids=[row[0] for row in EXPECTED_OUTCOMES],
)
def test_representative_outcomes(
    cases_by_id, engine, case_id, expected_sufficiency, expected_decision, why
):
    case = cases_by_id[case_id]
    outcome = engine.decide(case.dispute, case.evidence)
    assert outcome.sufficiency.level is expected_sufficiency, f"{case_id}: {why}"
    assert outcome.decision is expected_decision, f"{case_id}: {why}"


def test_declaration_substitutes_only_for_small_offline_merchants(cases_by_id, engine):
    """The same artifact type carries seed_030 and fails seed_032; only merchant type differs."""
    small = engine.decide(cases_by_id["seed_030"].dispute, cases_by_id["seed_030"].evidence)
    large = engine.decide(cases_by_id["seed_032"].dispute, cases_by_id["seed_032"].evidence)
    assert small.sufficiency.declaration_substituted is True
    assert small.sufficiency.matched_requirement == "acquirer_declaration_letter"
    assert large.sufficiency.declaration_substituted is False
    assert "acquirer_declaration_letter" not in large.sufficiency.required_any_of


def test_caps_gate_short_circuits_before_the_evidence_question(cases_by_id, engine):
    """seed_033 holds STRONG evidence and still routes to RGNB because the cap fired first."""
    outcome = engine.decide(cases_by_id["seed_033"].dispute, cases_by_id["seed_033"].evidence)
    assert outcome.sufficiency.level is Sufficiency.STRONG
    assert outcome.decision is Decision.RGNB
    assert outcome.caps.triggered is True
    assert [b.decline_code for b in outcome.caps.breaches] == ["CD1"]


def test_cap_boundary_pair_differs_only_by_the_counter(cases_by_id):
    """One below the limit is allowed; at the limit is declined."""
    allowed = evaluate_caps(cases_by_id["seed_039"].dispute)
    declined = evaluate_caps(cases_by_id["seed_033"].dispute)
    assert allowed.triggered is False
    assert declined.triggered is True


def test_fraud_transactions_are_exempt_from_the_caps(cases_by_id):
    caps = evaluate_caps(cases_by_id["seed_035"].dispute)
    assert caps.fraud_exempt is True
    assert caps.triggered is False
    assert caps.breaches, "seed_035 is above both limits; the breaches should still be recorded"


# --- known gap -------------------------------------------------------------------------------


def test_known_gap_contradictory_cases_resolve_to_file_not_escalate(cases_by_id, engine):
    """B0 cannot reach ESCALATE, and this documents exactly why.

    Contradiction detection is the verifier's job. StubVerifier reads back a per-artifact label
    and never compares artifacts, so it reports no contradictions and the ESCALATE branch stays
    unreachable. Each of these four cases holds individually valid evidence of an accepted type,
    so the thread grades them STRONG and files. The expected ground truth is ESCALATE.

    This test should start failing when a real verifier lands: that is the signal to update it.
    """
    for case_id in sorted(CONTRADICTORY_CASES):
        case = cases_by_id[case_id]
        outcome = engine.decide(case.dispute, case.evidence)
        assert outcome.contradictions == [], f"{case_id}: StubVerifier should detect nothing"
        assert outcome.sufficiency.level is Sufficiency.STRONG, case_id
        assert outcome.decision is Decision.FILE, case_id
        assert case.ground_truth.expected_decision is Decision.ESCALATE, case_id


def test_stub_verifier_never_reports_contradictions(cases_by_id):
    verifier = StubVerifier()
    for case in cases_by_id.values():
        assert verifier.find_contradictions(case.dispute, case.evidence) == []


# --- scoring ---------------------------------------------------------------------------------


def test_sufficiency_is_perfect_on_the_seed_set(scores):
    misses = [s.case_id for s in scores if not s.sufficiency_match]
    assert not misses, f"sufficiency mismatches: {misses}"
    assert len(scores) == 40


def test_decision_misses_are_exactly_the_contradictory_cases(scores):
    misses = {s.case_id for s in scores if not s.decision_match}
    assert misses == CONTRADICTORY_CASES
    assert sum(s.decision_match for s in scores) == 36


def test_report_lists_the_mismatches(scores):
    report = format_report(scores)
    assert "sufficiency accuracy : 40/40" in report
    assert "decision accuracy    : 36/40" in report
    assert "MISMATCHES (4)" in report
    for case_id in CONTRADICTORY_CASES:
        assert case_id in report


# --- the engine must not see test metadata ---------------------------------------------------


def test_engine_takes_only_a_record_and_artifacts():
    """The signature is the guardrail: no seed case, so no hard_case_class and no ground truth."""
    params = list(inspect.signature(DecisionEngine.decide).parameters)
    assert params == ["self", "dispute", "evidence"]


def test_decision_is_unchanged_when_ground_truth_is_altered(cases_by_id, engine):
    """Rewriting the label a case carries must not move the decision."""
    case = cases_by_id["seed_001"]
    before = engine.decide(case.dispute, case.evidence)
    tampered = case.model_copy(
        update={
            "ground_truth": case.ground_truth.model_copy(
                update={"expected_decision": Decision.CONCEDE}
            )
        }
    )
    after = engine.decide(tampered.dispute, tampered.evidence)
    assert before.decision is after.decision is Decision.FILE


def test_passthrough_classifier_reports_no_confidence(cases_by_id):
    result = PassthroughClassifier().classify(cases_by_id["seed_001"].dispute)
    assert result.reason_code == "RC_1064"
    assert result.confidence is None
    assert result.method == "passthrough"


# --- audit trail -----------------------------------------------------------------------------


def test_every_fired_rule_carries_a_source(cases_by_id, engine):
    for case in cases_by_id.values():
        outcome = engine.decide(case.dispute, case.evidence)
        for rule in outcome.rules_fired:
            assert rule.source.strip(), f"{case.case_id}/{rule.rule_id} has no source"
            assert "OC" in rule.source, f"{case.case_id}/{rule.rule_id} cites no circular"


def test_audit_record_reconstructs_the_decision(cases_by_id, engine):
    case = cases_by_id["seed_033"]
    audit = engine.decide(case.dispute, case.evidence).audit
    assert audit.dispute_id == case.dispute.dispute_id
    assert audit.decision is Decision.RGNB
    assert audit.engine_version == "B0"
    assert audit.inputs.chargeback_count_ifsc_acct == 10
    assert audit.caps.triggered is True
    assert {r.rule_id for r in audit.rules_fired} >= {"caps_gate", "policy:caps_short_circuit"}
    assert audit.model_dump_json()


def test_audit_inputs_exclude_test_metadata():
    from src.decision.audit import AuditInputs

    forbidden = {"hard_case_class", "ground_truth", "expected_decision", "expected_sufficiency"}
    assert forbidden.isdisjoint(AuditInputs.model_fields)
