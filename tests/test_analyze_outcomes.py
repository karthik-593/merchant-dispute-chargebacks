"""Tests for the outcome diagnostics.

Built on a freshly generated sample rather than the dataset on disk, so the suite does not depend
on a DVC pull.
"""

from __future__ import annotations

import pytest

from src.data.analyze_outcomes import (
    MIN_INTERQUARTILE_SPREAD,
    WIN_RATE_CEILING,
    WIN_RATE_FLOOR,
    bucket_stats,
    build_report,
    collect_flags,
    format_p2p_coverage,
    markdown_bucket_table,
    stats_by_decision,
    stats_by_sufficiency,
)
from src.data.generator import SyntheticGenerator
from src.data.schemas import RealizedOutcome, Sufficiency, TxnSubType
from src.evaluation.generator_sync import check_engine_agreement, contradictions_from_class


@pytest.fixture(scope="module")
def generated():
    return SyntheticGenerator().generate(600)


@pytest.fixture(scope="module")
def cases(generated):
    return [item.case for item in generated]


def test_bucket_stats_are_internally_consistent(cases):
    for row in stats_by_sufficiency(cases) + stats_by_decision(cases):
        if row.n == 0:
            continue
        assert 0 <= row.won <= row.n
        assert row.minimum <= row.p25 <= row.median <= row.p75 <= row.maximum
        assert row.interquartile_spread == pytest.approx(row.p75 - row.p25)
        assert row.win_rate == pytest.approx(row.won / row.n)


def test_empty_bucket_is_reported_not_crashed():
    row = bucket_stats("nothing", [])
    assert row.n == 0
    assert row.flags() == ["empty bucket"]


def test_buckets_cover_every_case(cases):
    assert sum(row.n for row in stats_by_sufficiency(cases)) == len(cases)
    assert sum(row.n for row in stats_by_decision(cases)) == len(cases)


def test_flags_fire_on_a_degenerate_bucket(cases):
    """A bucket where everything wins must be flagged, or the diagnostic is useless."""
    winners = [
        c
        for c in cases
        if c.ground_truth.realized_outcome is RealizedOutcome.WON
        and c.ground_truth.win_probability > 0.8
    ]
    row = bucket_stats("all-winners", winners)
    assert row.win_rate > WIN_RATE_CEILING
    assert any("win rate" in flag for flag in row.flags())


def test_flag_thresholds_are_the_documented_ones():
    assert WIN_RATE_FLOOR == 0.05
    assert WIN_RATE_CEILING == 0.95
    assert MIN_INTERQUARTILE_SPREAD == 0.10


def test_report_renders_and_returns_its_flags(cases):
    report, flags = build_report(cases)
    assert "by expected_sufficiency" in report
    assert "by expected_decision" in report
    assert "P2P coverage" in report
    assert isinstance(flags, list)
    for flag in flags:
        assert flag.split(":")[0] in {s.value for s in Sufficiency} | {
            "FILE",
            "CONCEDE",
            "ESCALATE",
            "RGNB",
        }


def test_report_does_not_mutate_the_cases(cases):
    before = [c.model_dump_json() for c in cases]
    build_report(cases)
    assert [c.model_dump_json() for c in cases] == before


def test_p2p_coverage_counts_match_the_data(cases):
    text = format_p2p_coverage(cases)
    p2p = [c for c in cases if c.dispute.txn_sub_type in (TxnSubType.U3, TxnSubType.UC)]
    assert "combined U3+UC" in text
    assert str(len(p2p)) in text
    # P2P must only ever carry the two reason codes the rulebook gives it.
    assert {c.dispute.reason_code for c in p2p} <= {"RC_108", "RC_121"}


def test_markdown_table_has_a_row_per_bucket(cases):
    rows = stats_by_sufficiency(cases)
    lines = markdown_bucket_table(rows, "sufficiency")
    assert lines[0].startswith("| sufficiency |")
    body = [line for line in lines if line.startswith("| `")]
    assert len(body) == len(rows)


def test_collect_flags_names_the_bucket(cases):
    flags = collect_flags(stats_by_sufficiency(cases))
    for flag in flags:
        assert ":" in flag


def test_contradictions_can_be_rebuilt_from_a_case_read_back(generated):
    """Guards the card-only path: rebuilding from disk must not lose the planted contradictions.

    Without this, Check A recomputed off a loaded dataset reports every contradiction case as an
    unexplained mismatch and the card claims the generator and engine disagree.
    """
    planted = [g for g in generated if g.contradictions]
    assert planted, "expected contradiction cases in the sample"
    for item in planted:
        assert contradictions_from_class(item.case), item.case.case_id

    rebuilt = [
        type(item)(case=item.case, contradictions=contradictions_from_class(item.case))
        for item in generated
    ]
    result = check_engine_agreement(rebuilt)
    assert result.passed, result.summary()
    assert result.expected_decision_gap == len(planted)
