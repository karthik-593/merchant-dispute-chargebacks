"""Command-line runner for a single case, for eyeballing the thread on one dispute."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.seed_loader import load_seed_case, load_seed_cases, seed_files
from src.decision.engine import DecisionEngine, DecisionOutcome


def find_case_path(case_id: str) -> Path:
    """Locate a seed file by case id or by filename stem."""
    for path in seed_files():
        if path.stem == case_id or path.stem.startswith(f"{case_id}_"):
            return path
    raise SystemExit(f"no seed case matching {case_id!r}; try --list")


def render(outcome: DecisionOutcome, verbose: bool) -> str:
    """Render one decision as text, or as the full audit record when asked."""
    if verbose:
        return outcome.audit.model_dump_json(indent=2)
    lines = [
        f"decision    : {outcome.decision.value}",
        f"sufficiency : {outcome.sufficiency.level.value}",
        f"reason      : {outcome.reason}",
        f"rulebook    : {outcome.sufficiency.rulebook_entry}  [{outcome.sufficiency.source}]",
        "rules fired :",
    ]
    lines += [
        f"  - {rule.rule_id} -> {rule.outcome}  [{rule.source}]" for rule in outcome.rules_fired
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Decide one case and print the outcome."""
    parser = argparse.ArgumentParser(description="Run the B0 thread over a single dispute.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case", help="seed case id, e.g. seed_001")
    source.add_argument("--file", help="path to a case file outside the seed set")
    parser.add_argument("--list", action="store_true", help="list available seed case ids")
    parser.add_argument("--json", action="store_true", help="print the full audit record as JSON")
    args = parser.parse_args(argv)

    if args.list:
        for case in load_seed_cases():
            print(f"{case.case_id}  {case.dispute.reason_code:8} {case.dispute.txn_sub_type.value}")
        return 0

    if not (args.case or args.file):
        parser.error("give --case or --file (or --list)")

    path = Path(args.file) if args.file else find_case_path(args.case)
    case = load_seed_case(path)
    outcome = DecisionEngine().decide(case.dispute, case.evidence)
    print(f"case        : {case.case_id}  ({path.name})")
    print(render(outcome, verbose=args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
