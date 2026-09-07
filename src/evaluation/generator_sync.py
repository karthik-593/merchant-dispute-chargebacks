"""The two checks that prove the generator and the engine agree on the rules.

Check A runs the B0 engine over generated cases. Sufficiency must match everywhere, and the
decision must match everywhere except the contradiction cases — B0's stub verifier cannot compare
artifacts, so it grades those STRONG and files them where the label says escalate. Any other
disagreement means the two have drifted.

Check B is the harder one. It applies the generator's labelling function to the seed-v1 inputs
and compares against labels a human wrote by hand. A mismatch there means the encoded rules
disagree with human judgement, which is a finding to report rather than a test to loosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.generator import GeneratedCase, label
from src.data.schemas import Decision, HardCaseClass, SeedCase
from src.data.seed_loader import load_seed_cases
from src.decision.engine import DecisionEngine
from src.decision.verifier import Contradiction


@dataclass(frozen=True)
class Mismatch:
    """One case where two labellings disagree."""

    case_id: str
    hard_case_class: str
    field_name: str
    produced: str
    expected: str
    detail: str = ""


@dataclass
class CheckResult:
    """Outcome of one consistency check."""

    name: str
    total: int
    sufficiency_matches: int
    decision_matches: int
    expected_decision_gap: int = 0
    unexpected: list[Mismatch] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when nothing disagreed beyond the documented, expected gap."""
        return not self.unexpected and self.sufficiency_matches == self.total

    def summary(self) -> str:
        """One-paragraph rendering for the report and the dataset card."""
        lines = [
            f"{self.name}",
            f"  cases checked        : {self.total}",
            f"  sufficiency agreement: {self.sufficiency_matches}/{self.total}",
            f"  decision agreement   : {self.decision_matches}/{self.total}",
        ]
        if self.expected_decision_gap:
            lines.append(
                f"  expected gap         : {self.expected_decision_gap} contradiction case(s) "
                f"B0 cannot detect"
            )
        if self.unexpected:
            lines.append(f"  UNEXPECTED mismatches: {len(self.unexpected)}")
            for m in self.unexpected[:10]:
                lines.append(
                    f"    {m.case_id} [{m.hard_case_class}] {m.field_name}: "
                    f"{m.produced} != {m.expected} {m.detail}".rstrip()
                )
        else:
            lines.append("  UNEXPECTED mismatches: none")
        lines.append(f"  result               : {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def check_engine_agreement(generated: list[GeneratedCase]) -> CheckResult:
    """Check A: run B0 over generated cases and compare with the generated labels."""
    engine = DecisionEngine()
    suff_hits = 0
    dec_hits = 0
    expected_gap = 0
    unexpected: list[Mismatch] = []

    for item in generated:
        case = item.case
        truth = case.ground_truth
        outcome = engine.decide(case.dispute, case.evidence)

        if outcome.sufficiency.level is truth.expected_sufficiency:
            suff_hits += 1
        else:
            unexpected.append(
                Mismatch(
                    case_id=case.case_id,
                    hard_case_class=case.hard_case_class.value,
                    field_name="sufficiency",
                    produced=outcome.sufficiency.level.value,
                    expected=truth.expected_sufficiency.value,
                )
            )

        if outcome.decision is truth.expected_decision:
            dec_hits += 1
            continue

        # The one disagreement the design predicts: B0 cannot see a contradiction, so a case
        # labelled ESCALATE comes back FILE.
        predicted_gap = (
            bool(item.contradictions)
            and truth.expected_decision is Decision.ESCALATE
            and outcome.decision is Decision.FILE
        )
        if predicted_gap:
            expected_gap += 1
        else:
            unexpected.append(
                Mismatch(
                    case_id=case.case_id,
                    hard_case_class=case.hard_case_class.value,
                    field_name="decision",
                    produced=outcome.decision.value,
                    expected=truth.expected_decision.value,
                )
            )

    return CheckResult(
        name="Check A - B0 engine vs generated labels",
        total=len(generated),
        sufficiency_matches=suff_hits,
        decision_matches=dec_hits,
        expected_decision_gap=expected_gap,
        unexpected=unexpected,
    )


def _seed_contradictions(case: SeedCase) -> list[Contradiction]:
    """Contradictions implied by a seed case's hand-assigned class.

    Reads `hard_case_class`, which the engine may never do. That is legitimate here: this is a
    check harness comparing two labellings, not a decision path.
    """
    if case.hard_case_class is not HardCaseClass.CONTRADICTORY or len(case.evidence) < 2:
        return []
    return [
        Contradiction(
            evidence_ids=[a.evidence_id for a in case.evidence],
            detail="hand-labelled contradictory bundle",
            critical=True,
        )
    ]


def check_seed_reproduction(cases: list[SeedCase] | None = None) -> CheckResult:
    """Check B: does the generator's labelling reproduce the hand-written seed-v1 labels?"""
    seed_cases = cases if cases is not None else load_seed_cases()
    suff_hits = 0
    dec_hits = 0
    unexpected: list[Mismatch] = []

    for case in seed_cases:
        sufficiency, decision, _, _ = label(case.dispute, case.evidence, _seed_contradictions(case))
        truth = case.ground_truth

        if sufficiency is truth.expected_sufficiency:
            suff_hits += 1
        else:
            unexpected.append(
                Mismatch(
                    case_id=case.case_id,
                    hard_case_class=case.hard_case_class.value,
                    field_name="sufficiency",
                    produced=sufficiency.value,
                    expected=truth.expected_sufficiency.value,
                    detail="generator logic disagrees with the human label",
                )
            )

        if decision is truth.expected_decision:
            dec_hits += 1
        else:
            unexpected.append(
                Mismatch(
                    case_id=case.case_id,
                    hard_case_class=case.hard_case_class.value,
                    field_name="decision",
                    produced=decision.value,
                    expected=truth.expected_decision.value,
                    detail="generator logic disagrees with the human label",
                )
            )

    return CheckResult(
        name="Check B - generator labelling vs hand-written seed-v1 labels",
        total=len(seed_cases),
        sufficiency_matches=suff_hits,
        decision_matches=dec_hits,
        unexpected=unexpected,
    )
