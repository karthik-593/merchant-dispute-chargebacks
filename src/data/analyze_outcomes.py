"""Diagnostics on the synthetic outcome label and on P2P coverage. Measures, never mutates.

Two questions this answers.

**Is the outcome target learnable but not trivial?** If a bucket wins essentially always or never,
or if the modelled probabilities inside it are bunched into a narrow band, then a scorer can
recover the label from the bucket alone and its reported skill is an artefact of the generator
rather than a measurement of anything.

**How thin are the P2P cells?** Person-to-person disputes are a minority by design. Knowing how
few cases sit in each P2P reason code tells you how much weight a per-type metric on them can
carry.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.schemas import Decision, RealizedOutcome, SeedCase, Sufficiency, TxnSubType
from src.data.synthetic_dataset import load_splits, synthetic_dir

# A bucket whose win rate sits outside this band is effectively decided by its label alone.
WIN_RATE_FLOOR = 0.05
WIN_RATE_CEILING = 0.95
# Modelled probabilities bunched tighter than this leave a scorer nothing to separate.
MIN_INTERQUARTILE_SPREAD = 0.10

P2P_SUB_TYPES = (TxnSubType.U3, TxnSubType.UC)


@dataclass(frozen=True)
class BucketStats:
    """Realised and modelled outcome statistics for one bucket of cases."""

    name: str
    n: int
    won: int
    mean_probability: float
    minimum: float
    p25: float
    median: float
    p75: float
    maximum: float

    @property
    def win_rate(self) -> float:
        """Share of cases whose counterfactual outcome came out WON."""
        return self.won / self.n if self.n else 0.0

    @property
    def interquartile_spread(self) -> float:
        """Width of the middle half of the modelled probabilities."""
        return self.p75 - self.p25

    def flags(self) -> list[str]:
        """Reasons this bucket would make the learning target too easy."""
        problems = []
        if self.n == 0:
            return ["empty bucket"]
        if self.win_rate > WIN_RATE_CEILING:
            problems.append(f"win rate {self.win_rate:.1%} > {WIN_RATE_CEILING:.0%}")
        if self.win_rate < WIN_RATE_FLOOR:
            problems.append(f"win rate {self.win_rate:.1%} < {WIN_RATE_FLOOR:.0%}")
        if self.interquartile_spread < MIN_INTERQUARTILE_SPREAD:
            problems.append(
                f"p75-p25 spread {self.interquartile_spread:.3f} < {MIN_INTERQUARTILE_SPREAD}"
            )
        return problems


def bucket_stats(name: str, cases: list[SeedCase]) -> BucketStats:
    """Summarise the outcome label for one bucket."""
    if not cases:
        return BucketStats(name, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    probabilities = np.array(
        [c.ground_truth.win_probability for c in cases], dtype=float
    )
    won = sum(1 for c in cases if c.ground_truth.realized_outcome is RealizedOutcome.WON)
    q25, q50, q75 = (float(v) for v in np.percentile(probabilities, [25, 50, 75]))
    return BucketStats(
        name=name,
        n=len(cases),
        won=won,
        mean_probability=float(probabilities.mean()),
        minimum=float(probabilities.min()),
        p25=q25,
        median=q50,
        p75=q75,
        maximum=float(probabilities.max()),
    )


def stats_by_sufficiency(cases: list[SeedCase]) -> list[BucketStats]:
    """Outcome statistics per sufficiency level."""
    return [
        bucket_stats(
            level.value,
            [c for c in cases if c.ground_truth.expected_sufficiency is level],
        )
        for level in Sufficiency
    ]


def stats_by_decision(cases: list[SeedCase]) -> list[BucketStats]:
    """Outcome statistics per decision."""
    return [
        bucket_stats(
            decision.value,
            [c for c in cases if c.ground_truth.expected_decision is decision],
        )
        for decision in Decision
    ]


def format_bucket_table(title: str, rows: list[BucketStats]) -> str:
    """Render one bucket table with its spread columns."""
    lines = [
        title,
        f"  {'bucket':10} {'n':>5} {'won':>5} {'win%':>7} {'mean p':>7} "
        f"{'min':>6} {'p25':>6} {'med':>6} {'p75':>6} {'max':>6} {'IQR':>6}  flags",
        "  " + "-" * 100,
    ]
    for row in rows:
        if row.n == 0:
            lines.append(f"  {row.name:10} {0:>5}   (no cases)")
            continue
        flags = "; ".join(row.flags())
        lines.append(
            f"  {row.name:10} {row.n:>5} {row.won:>5} {row.win_rate:>6.1%} "
            f"{row.mean_probability:>7.3f} {row.minimum:>6.3f} {row.p25:>6.3f} "
            f"{row.median:>6.3f} {row.p75:>6.3f} {row.maximum:>6.3f} "
            f"{row.interquartile_spread:>6.3f}  {'WARN ' + flags if flags else 'ok'}"
        )
    return "\n".join(lines)


def format_p2p_coverage(cases: list[SeedCase]) -> str:
    """Render the sub-type counts and the P2P breakdown by reason code and decision."""
    total = len(cases)
    by_sub_type = Counter(c.dispute.txn_sub_type.value for c in cases)
    p2p = [c for c in cases if c.dispute.txn_sub_type in P2P_SUB_TYPES]

    lines = ["P2P coverage", "  by transaction sub-type", "  " + "-" * 40]
    for sub_type in TxnSubType:
        count = by_sub_type.get(sub_type.value, 0)
        marker = "  (P2P)" if sub_type in P2P_SUB_TYPES else ""
        lines.append(f"  {sub_type.value:5} {count:>6} {count / total:>7.1%}{marker}")
    lines.append(f"  {'P2P':5} {len(p2p):>6} {len(p2p) / total:>7.1%}  combined U3+UC")

    lines += ["", "  P2P cells by reason code", "  " + "-" * 40]
    for code, count in sorted(Counter(c.dispute.reason_code for c in p2p).items()):
        by_type = Counter(c.dispute.txn_sub_type.value for c in p2p if c.dispute.reason_code == code)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        lines.append(f"  {code:10} {count:>6}   ({detail})")

    lines += ["", "  P2P cells by decision", "  " + "-" * 40]
    for decision in Decision:
        count = sum(1 for c in p2p if c.ground_truth.expected_decision is decision)
        share = count / len(p2p) if p2p else 0.0
        lines.append(f"  {decision.value:10} {count:>6} {share:>7.1%}")

    lines += ["", "  thinnest P2P cell (reason code x decision)", "  " + "-" * 40]
    cells = Counter(
        (c.dispute.reason_code, c.ground_truth.expected_decision.value) for c in p2p
    )
    for (code, decision), count in sorted(cells.items(), key=lambda kv: kv[1])[:6]:
        lines.append(f"  {code:10} {decision:10} {count:>5}")
    return "\n".join(lines)


def collect_flags(rows: list[BucketStats]) -> list[str]:
    """Flatten every flag raised across a set of buckets."""
    return [f"{row.name}: {problem}" for row in rows for problem in row.flags()]


def build_report(cases: list[SeedCase]) -> tuple[str, list[str]]:
    """Render the whole diagnostic report and return it with any flags raised."""
    by_sufficiency = stats_by_sufficiency(cases)
    by_decision = stats_by_decision(cases)
    flags = collect_flags(by_sufficiency) + collect_flags(by_decision)

    sections = [
        f"Outcome diagnostics over {len(cases)} cases (train + val + test)",
        "",
        format_bucket_table("Win rate and win_probability by expected_sufficiency", by_sufficiency),
        "",
        format_bucket_table("Win rate and win_probability by expected_decision", by_decision),
        "",
        format_p2p_coverage(cases),
        "",
        "Calibration flags",
        "  " + "-" * 40,
    ]
    if flags:
        sections += [f"  WARN {flag}" for flag in flags]
        sections.append(
            "\n  A flagged bucket means the target may be recoverable from its label alone, "
            "which would inflate any score the M11 scorer reports."
        )
    else:
        sections.append(
            "  none - every bucket sits inside the win-rate band and keeps a usable spread"
        )
    return "\n".join(sections), flags


def load_full_dataset(directory: Path | None = None) -> list[SeedCase]:
    """Load train, validation and test. Dev is excluded: it is a subset of train."""
    splits = load_splits(directory or synthetic_dir())
    return splits.train + splits.val + splits.test


def main(argv: list[str] | None = None) -> int:
    """Print the diagnostics. Flags are reported, not fatal."""
    parser = argparse.ArgumentParser(description="Diagnostics for the synthetic outcome label.")
    parser.add_argument("--dir", type=Path, default=None, help="dataset directory")
    args = parser.parse_args(argv)

    report, _ = build_report(load_full_dataset(args.dir))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def markdown_bucket_table(rows: list[BucketStats], header: str) -> list[str]:
    """Render bucket statistics as a markdown table for the dataset card."""
    lines = [
        f"| {header} | cases | won | win rate | mean p | min | p25 | median | p75 | max | p75-p25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.n == 0:
            lines.append(f"| `{row.name}` | 0 | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{row.name}` | {row.n} | {row.won} | {row.win_rate:.1%} | "
            f"{row.mean_probability:.3f} | {row.minimum:.3f} | {row.p25:.3f} | "
            f"{row.median:.3f} | {row.p75:.3f} | {row.maximum:.3f} | "
            f"{row.interquartile_spread:.3f} |"
        )
    lines.append("")
    return lines
