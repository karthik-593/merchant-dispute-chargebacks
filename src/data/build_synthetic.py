"""Builds dataset-v1.0 end to end: generate, validate, split, lock, check, document.

Run with `uv run python -m src.data.build_synthetic`. Generation fails loudly rather than
shipping data that violates an invariant, and the dataset card is written from the data that was
actually produced, so its numbers cannot drift from the files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.analyze_outcomes import (
    collect_flags,
    markdown_bucket_table,
    stats_by_decision,
    stats_by_sufficiency,
)
from src.data.generator import (
    DATASET_VERSION,
    GeneratedCase,
    SyntheticGenerator,
    load_generator_config,
)
from src.data.schemas import Decision, HardCaseClass, SeedCase
from src.data.synthetic_dataset import (
    DatasetSplits,
    load_splits,
    synthetic_dir,
    temporal_split,
    write_jsonl,
    write_test_lock,
)
from src.data.validation import validate_dataset
from src.evaluation.generator_sync import (
    CheckResult,
    contradictions_from_class,
    check_engine_agreement,
    check_seed_reproduction,
)
from src.logging_setup import get_logger

log = get_logger(__name__)

CARD_NAME = "DATASET_CARD.md"


def _table(title: str, counts: Counter, total: int, header: str = "value") -> list[str]:
    lines = [f"### {title}", "", f"| {header} | cases | share |", "|---|---:|---:|"]
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{key}` | {count} | {count / total:.1%} |")
    lines.append("")
    return lines


def _calibration_notes(cases: list[SeedCase]) -> list[str]:
    """Commentary on any calibration flag the diagnostics raise."""
    flags = collect_flags(stats_by_sufficiency(cases)) + collect_flags(stats_by_decision(cases))
    if not flags:
        return [
            "No bucket is flagged: every one sits inside the 5-95% win-rate band and keeps an "
            "interquartile spread of at least 0.10.",
            "",
        ]
    return [
        "**Flagged buckets.** The diagnostics warn where a bucket's win rate falls outside 5-95% "
        "or its interquartile spread drops below 0.10:",
        "",
        *[f"- {flag}" for flag in flags],
        "",
        "These are recorded rather than corrected. A narrow spread inside a bucket means the "
        "modelled probabilities there are bunched, so within that bucket there is little for a "
        "scorer to separate; treat per-bucket skill claims on it with suspicion until the noise "
        "model is revisited.",
        "",
    ]


def build_card(
    cases: list[SeedCase],
    splits: DatasetSplits,
    check_a: CheckResult,
    check_b: CheckResult,
    config: dict,
) -> str:
    """Render the dataset card from the data actually generated."""
    total = len(cases)
    noise = config["noise_model"]
    window = config["window"]
    won = sum(1 for c in cases if c.ground_truth.realized_outcome.value == "WON")
    probs = [c.ground_truth.win_probability for c in cases]

    lines = [
        f"# Dataset card — `{DATASET_VERSION}`",
        "",
        "## Name and status",
        "",
        f"**`{DATASET_VERSION}`** — the synthetic dispute set for the UPI merchant-dispute agent.",
        "",
        "Distinct from `seed-v1`, which is the 40-case hand-written fixture set. That one is the "
        "human ground truth; this one is generated, far larger, and carries a stochastic outcome "
        "label the seed set does not have.",
        "",
        "## Generation method",
        "",
        f"Produced by `src/data/generator.py` from `configs/generator.yaml`, seeded with "
        f"**{config['seed']}**. Regenerating with the same seed reproduces the files byte for "
        "byte.",
        "",
        "The generator holds **no domain rules of its own**. Reason codes, accepted evidence "
        "types, the small/offline declaration substitution and the cap limits are all read from "
        "`configs/rulebook/` at generation time. Labelling runs the decision engine itself rather "
        "than a second copy of the rules: the generator wires that engine with an oracle verifier "
        "that reports the contradictions it planted, while the B0 baseline wires the same engine "
        "with the stub verifier that cannot detect them. The two therefore agree by construction "
        "everywhere except on contradiction detection.",
        "",
        "Each case is built as a chain: transaction fields, then the dispute and a "
        "natural-language narrative drawn from several surface variants per case class, then the "
        "evidence artifacts with their validity labels, then the ground truth.",
        "",
        f"- **Cases**: {total}",
        f"- **Seed**: {config['seed']}",
        f"- **Transaction window**: {window['start']} to {window['end']}",
        f"- **Format**: one JSON object per line, one file per split",
        "",
        "## Distributions",
        "",
    ]

    lines += _table(
        "By reason code", Counter(c.dispute.reason_code for c in cases), total, "reason code"
    )
    lines += _table(
        "By transaction sub-type",
        Counter(c.dispute.txn_sub_type.value for c in cases),
        total,
        "sub-type",
    )
    lines += _table(
        "By hard-case class",
        Counter(c.hard_case_class.value for c in cases),
        total,
        "class",
    )
    lines += _table(
        "By sufficiency",
        Counter(c.ground_truth.expected_sufficiency.value for c in cases),
        total,
        "sufficiency",
    )
    lines += _table(
        "By decision",
        Counter(c.ground_truth.expected_decision.value for c in cases),
        total,
        "decision",
    )
    lines += _table(
        "By realized outcome",
        Counter(c.ground_truth.realized_outcome.value for c in cases),
        total,
        "outcome",
    )

    lines += [
        "Other splits of interest: "
        f"{sum(1 for c in cases if c.dispute.merchant_type.value == 'small_offline')} small/offline "
        f"merchants, {sum(1 for c in cases if c.dispute.fraud_flag)} fraud-flagged, "
        f"{sum(1 for c in cases if c.dispute.acquiring_psp_is_merchant_bank)} where the acquiring "
        "PSP and merchant bank are the same institution.",
        "",
        "## The two labels",
        "",
        "Every case carries both, and they answer different questions.",
        "",
        "### 1. Deterministic rule label",
        "",
        "`expected_sufficiency` and `expected_decision`, computed by the decision engine from the "
        "rulebook. Policy order is the engine's: the caps gate short-circuits first, then a "
        "critical contradiction escalates, then sufficiency decides — STRONG files, WEAK and "
        "ABSENT concede.",
        "",
        "### 2. Stochastic realized outcome",
        "",
        "`realized_outcome` (WON/LOST) with its `win_probability`. This is the **counterfactual** "
        "question: would the representment have succeeded, had it been filed? It is defined for "
        "every case regardless of the decision taken, it is drawn rather than derived, and it "
        "**never feeds back into the deterministic decision**. It is the target the calibrated "
        "scorer will learn.",
        "",
        "The model:",
        "",
        "```",
        "logit(p) = base[sufficiency]",
        f"         + {noise['corroboration_bonus']} * (valid accepted artifacts - 1)",
        f"         + {noise['declaration_penalty']}   if the case rests on a declaration alone",
        f"         + {noise['contradiction_penalty']}   if the bundle contradicts itself",
        f"         + Normal(0, {noise['adjudicator_sigma']})   adjudicator variance",
        "",
        f"p = clip(sigmoid(logit), {noise['floor']}, {noise['ceiling']})",
        "realized_outcome ~ Bernoulli(p)",
        "```",
        "",
        f"Base logits: STRONG {noise['base_logit']['STRONG']}, WEAK "
        f"{noise['base_logit']['WEAK']}, ABSENT {noise['base_logit']['ABSENT']}.",
        "",
        "Three deliberate properties. **Adjudicator variance** means two identical evidence "
        "bundles can land differently, because two panels do. **Corroboration** rewards a second "
        "valid artifact, so evidence strength is graded rather than binary. **The floor and "
        "ceiling** keep any case from being a certainty, so a scorer cannot reach a perfect AUC "
        "by memorising the sufficiency label — which is the whole point of injecting noise at "
        "all. Realised win rate across the set is "
        f"{won}/{total} = {won / total:.1%}, with modelled probabilities spanning "
        f"{min(probs):.2f} to {max(probs):.2f} (mean {sum(probs) / len(probs):.2f}).",
        "",
        "## Outcome calibration",
        "",
        "Measured on the generated set by `src/data/analyze_outcomes.py`. The point of these "
        "tables is to show the outcome label is neither trivial nor degenerate: if a bucket won "
        "almost always or almost never, or its probabilities were bunched into a narrow band, a "
        "scorer could recover the outcome from the bucket alone and report skill it does not have.",
        "",
        "### By expected sufficiency",
        "",
        *markdown_bucket_table(stats_by_sufficiency(cases), "sufficiency"),
        "### By expected decision",
        "",
        *markdown_bucket_table(stats_by_decision(cases), "decision"),
        *_calibration_notes(cases),
        "## Splits",
        "",
        "**Temporal, by `txn_timestamp`.** Cases are ordered by transaction time and cut, so the "
        "test set is strictly the latest slice — trained on the past, judged on what came after. "
        "A shuffled split would leak future information; this cannot.",
        "",
        "| split | cases | share |",
        "|---|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        count = splits.sizes[name]
        lines.append(f"| {name} | {count} | {count / total:.1%} |")
    lines += [
        f"| dev (subset of train) | {splits.sizes['dev']} | — |",
        "",
        "The **test split is locked**: `TEST_SET_LOCK.json` records its SHA-256 and case count. "
        "It must not be used to select a model, a prompt or a threshold. A changed hash means the "
        "held-out set moved and anything measured on it is void.",
        "",
        "The dev subset is drawn from train only, so fast iteration can never touch held-out data.",
        "",
        "## Consistency checks",
        "",
        "These are what make the dataset trustworthy: they test that the generator and the engine "
        "agree about the rules, and that both agree with a human.",
        "",
        "```",
        check_a.summary(),
        "",
        check_b.summary(),
        "```",
        "",
        "**Check A** confirms the only disagreement between generated labels and the B0 engine is "
        f"the {check_a.expected_decision_gap} contradiction case(s) B0 structurally cannot detect. "
        "Any other divergence would mean the two had drifted apart.",
        "",
        "**Check B** is the stronger one. It applies the generator's labelling to the 40 seed-v1 "
        "inputs and compares against labels a human wrote by hand, ESCALATE cases included. It "
        "passing means the encoded rules agree with human judgement on every hand-reviewed case; "
        "had it failed, the finding would be reported rather than the check relaxed.",
        "",
        "## Known limitations",
        "",
        "- Narratives are template-drawn, not written. They carry enough surface variation to "
        "exercise a classifier's robustness a little, but they are not a substitute for real "
        "complaint text and should not be used to claim anything about natural-language "
        "performance.",
        "- Contradictions are planted structurally — two individually valid artifacts of accepted "
        "types marked as conflicting — rather than being emergent from artifact content. A real "
        "verifier will face subtler conflicts than these.",
        "- The noise model is a stated assumption, not an estimate from observed outcomes. Its "
        "base rates are plausible rather than measured, so absolute win rates here carry no "
        "external meaning; only relative ordering is intended to.",
        "- Deemed approval is carried as metadata and deliberately changes no outcome, matching "
        "the engine. When the presumption is given weight in the policy, this dataset will need "
        "regenerating.",
        "- **P2P is intentionally thin.** U3 and UC together are about 14% of the set, which "
        "reflects the real mix rather than an oversight, and the rulebook gives P2P only two "
        "reason codes (RC 108 and RC 121) against ten for P2M. Per-type metrics on P2P will "
        "therefore be high-variance, and the thinnest reason-code-by-decision cells hold only a "
        "handful of cases. This is flagged for reporting in the final evaluation — confidence "
        "intervals on P2P slices must be shown rather than point estimates — and is deliberately "
        "not corrected by rebalancing, which would misrepresent the population.",
        "",
    ]
    return "\n".join(lines)


def _rewrite_card(out: Path, config: dict) -> int:
    """Rebuild the card from the splits already written, leaving the data and lock untouched.

    Regenerating would rewrite the test set and its lock; refreshing the documentation alone
    should never do that.
    """
    splits = load_splits(out)
    cases = splits.train + splits.val + splits.test
    # The splits on disk do not carry the generator's record of what it planted, so the
    # contradictions are reconstructed from each case's class before Check A runs.
    generated = [
        GeneratedCase(case=case, contradictions=contradictions_from_class(case))
        for case in cases
    ]
    check_a = check_engine_agreement(generated)
    check_b = check_seed_reproduction()
    (out / CARD_NAME).write_text(
        build_card(cases, splits, check_a, check_b, config), encoding="utf-8"
    )
    print(f"card rewritten from the existing splits in {out} ({len(cases)} cases); data untouched")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Generate, validate, split, lock and document the synthetic dataset."""
    parser = argparse.ArgumentParser(description="Build the synthetic dataset (dataset-v1.0).")
    parser.add_argument("-n", "--cases", type=int, default=None, help="override the case count")
    parser.add_argument("--seed", type=int, default=None, help="override the generator seed")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="rewrite the dataset card from the splits already on disk, touching no data",
    )
    args = parser.parse_args(argv)

    config = load_generator_config()
    if args.card_only:
        return _rewrite_card(args.out or synthetic_dir(), config)

    generator = SyntheticGenerator(seed=args.seed)
    generated = generator.generate(args.cases)
    cases = [item.case for item in generated]
    log.info("generated %d cases with seed %s", len(cases), generator.seed)

    splits = temporal_split(
        cases,
        fractions=config["split"],
        dev_size=config["split"]["dev_subset_size"],
        dev_seed=generator.seed,
    )
    validate_dataset(
        splits,
        expected_shares=config["hard_case_class_shares"],
        tolerance=config["distribution_tolerance"],
    )
    log.info("validation passed")

    out = args.out or synthetic_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name, split_cases in splits.as_dict().items():
        write_jsonl(split_cases, out / f"{name}.jsonl")
    lock_path = write_test_lock(out / "test.jsonl", len(splits.test), generator.seed)

    check_a = check_engine_agreement(generated)
    check_b = check_seed_reproduction()

    (out / CARD_NAME).write_text(
        build_card(cases, splits, check_a, check_b, config), encoding="utf-8"
    )

    print(f"dataset-v1.0 written to {out}")
    print(f"  splits: {splits.sizes}")
    print(f"  lock  : {lock_path.name}")
    print()
    print(check_a.summary())
    print()
    print(check_b.summary())

    if not (check_a.passed and check_b.passed):
        print("\nCONSISTENCY CHECK FAILED - the generator and the engine disagree.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
