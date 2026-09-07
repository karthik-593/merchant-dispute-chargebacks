"""Runs the B0 thread over the seed-v1 fixtures and scores it against the hand-set ground truth.

The engine sees only a dispute record and its artifacts. Ground truth and `hard_case_class` are
read here, after the fact, purely to score and to label the mismatch report — never passed in.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

from src.config import load_config
from src.data.schemas import Decision, SeedCase, Sufficiency
from src.data.seed_loader import load_seed_cases
from src.decision.engine import ENGINE_VERSION, DecisionEngine
from src.logging_setup import get_logger

log = get_logger(__name__)

RUN_NAME = "B0-seed"
DATASET_VERSION = "seed-v1"


@dataclass(frozen=True)
class CaseScore:
    """One case's predicted and expected labels."""

    case_id: str
    hard_case_class: str
    predicted_sufficiency: Sufficiency
    expected_sufficiency: Sufficiency
    predicted_decision: Decision
    expected_decision: Decision
    reason: str

    @property
    def sufficiency_match(self) -> bool:
        """Whether the predicted sufficiency equals the hand-set label."""
        return self.predicted_sufficiency is self.expected_sufficiency

    @property
    def decision_match(self) -> bool:
        """Whether the predicted decision equals the hand-set label."""
        return self.predicted_decision is self.expected_decision

    @property
    def matched(self) -> bool:
        """Whether both labels match."""
        return self.sufficiency_match and self.decision_match


def score_case(case: SeedCase, engine: DecisionEngine) -> CaseScore:
    """Run one case through the thread and compare it with the hand-set ground truth."""
    outcome = engine.decide(case.dispute, case.evidence)
    return CaseScore(
        case_id=case.case_id,
        hard_case_class=case.hard_case_class.value,
        predicted_sufficiency=outcome.sufficiency.level,
        expected_sufficiency=case.ground_truth.expected_sufficiency,
        predicted_decision=outcome.decision,
        expected_decision=case.ground_truth.expected_decision,
        reason=outcome.reason,
    )


def run(cases: list[SeedCase] | None = None) -> list[CaseScore]:
    """Score every seed case."""
    engine = DecisionEngine()
    return [score_case(case, engine) for case in (cases or load_seed_cases())]


def git_commit() -> str:
    """Current commit, so a run can be traced back to the code that produced it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def format_report(scores: list[CaseScore]) -> str:
    """Render the per-case table, the summary and the mismatch list."""
    total = len(scores)
    suff_hits = sum(s.sufficiency_match for s in scores)
    dec_hits = sum(s.decision_match for s in scores)

    lines = [
        f"B0 vertical slice over {DATASET_VERSION} ({total} cases, engine {ENGINE_VERSION})",
        "",
        f"{'case':10} {'class':21} {'suff pred':10} {'suff exp':10} {'dec pred':9} "
        f"{'dec exp':9} match",
        "-" * 88,
    ]
    for s in scores:
        mark = "ok" if s.matched else "MISS"
        lines.append(
            f"{s.case_id:10} {s.hard_case_class:21} {s.predicted_sufficiency.value:10} "
            f"{s.expected_sufficiency.value:10} {s.predicted_decision.value:9} "
            f"{s.expected_decision.value:9} {mark}"
        )

    lines += [
        "-" * 88,
        "",
        "SUMMARY",
        f"  sufficiency accuracy : {suff_hits}/{total} = {suff_hits / total:.3f}",
        f"  decision accuracy    : {dec_hits}/{total} = {dec_hits / total:.3f}",
        "",
    ]

    mismatches = [s for s in scores if not s.matched]
    if not mismatches:
        lines.append("MISMATCHES: none")
    else:
        lines.append(f"MISMATCHES ({len(mismatches)})")
        for s in mismatches:
            wrong = []
            if not s.sufficiency_match:
                wrong.append(
                    f"sufficiency {s.predicted_sufficiency.value} != {s.expected_sufficiency.value}"
                )
            if not s.decision_match:
                wrong.append(
                    f"decision {s.predicted_decision.value} != {s.expected_decision.value}"
                )
            lines.append(f"  {s.case_id} [{s.hard_case_class}] {'; '.join(wrong)}")
            lines.append(f"      engine said: {s.reason}")
    return "\n".join(lines)


def log_to_mlflow(scores: list[CaseScore]) -> str | None:
    """Log the two accuracy numbers to MLflow. Returns the run id, or None if logging failed."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping experiment logging")
        return None

    config = load_config()["mlflow"]
    total = len(scores)
    try:
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(config["experiment_name"])
        with mlflow.start_run(run_name=RUN_NAME) as run:
            mlflow.log_params(
                {
                    "baseline": ENGINE_VERSION,
                    "dataset_version": DATASET_VERSION,
                    "case_count": total,
                    "classifier": "PassthroughClassifier",
                    "verifier": "StubVerifier",
                    "git_commit": git_commit(),
                }
            )
            mlflow.log_metrics(
                {
                    "sufficiency_accuracy": sum(s.sufficiency_match for s in scores) / total,
                    "decision_accuracy": sum(s.decision_match for s in scores) / total,
                }
            )
            return run.info.run_id
    except Exception as error:  # noqa: BLE001 - logging must never fail the evaluation
        log.warning("MLflow logging failed: %s", error)
        return None


def main() -> None:
    """Entry point: run the thread, print the report, log the metrics."""
    parser = argparse.ArgumentParser(description="Run the B0 thread over the seed-v1 fixtures.")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="skip MLflow logging (report only)"
    )
    args = parser.parse_args()

    scores = run()
    print(format_report(scores))

    if not args.no_mlflow:
        run_id = log_to_mlflow(scores)
        print(f"\nMLflow run '{RUN_NAME}': {run_id or 'not logged'}")


if __name__ == "__main__":
    main()
