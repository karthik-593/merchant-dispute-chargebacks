"""Read the frozen M7 configuration, and log it to MLflow as one run.

The freeze lives in `configs/versions.yaml` as data. This module reads it, and records it once in
MLflow beside the arms that earned it, so a later reader can get from a frozen component to the
experiment that selected it without reading a changelog.

Nothing here decides anything. Every `eval_*` harness in this package prints "nothing is frozen
here" and means it; this is the one place a frozen value is even read.
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from src.config import load_config, project_path
from src.logging_setup import get_logger

log = get_logger(__name__)

EXPERIMENT = "M7-retrieval"
ARM = "freeze"
VERSIONS_FILE = "versions.yaml"
M8_EVAL_FILE = "eval/m8_query_rewrite_eval.yaml"

FROZEN_KEYS = (
    "CHUNKER_VERSION",
    "RETRIEVER_VERSION",
    "EMBEDDING_VERSION",
    "RERANKER_VERSION",
)

# The lever comparison the freeze rests on. Kept beside the freeze rather than only in prose, so
# the justification is queryable: every alternative, what it cost, and what it recovered.
SELECTION_TABLE = (
    ("encoder", "6 models / 2 sizes / 3 recipes", "0 hard rows; ensemble union = best single"),
    ("depth", "K_retrieve 25 -> 100", "1 hard row (h02)"),
    ("reranker", "cross-encoder over row-grain chunks", "negative: 1 win, 4-9 regressions"),
    ("reranker", "cross-encoder over +/-2-row parents", "negative: h02 and h32 both worse"),
    ("expansion", "parent = +/-2 rows", "0 hard rows recovered"),
    ("expansion", "parent = full table", "0 rows; +0.12 all-50 is the B3 ruler artefact"),
)


def load_versions() -> dict[str, Any]:
    """Read the frozen component versions."""
    with (project_path("configs") / VERSIONS_FILE).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_m8_eval_set() -> dict[str, Any]:
    """Read the M8 query-rewrite evaluation set."""
    with (project_path("data") / M8_EVAL_FILE).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def frozen_components(versions: dict[str, Any] | None = None) -> dict[str, str | None]:
    """The four frozen values, by name."""
    data = versions or load_versions()
    return {key: data[key]["value"] for key in FROZEN_KEYS}


def format_report(versions: dict[str, Any], eval_set: dict[str, Any]) -> str:
    """The freeze block, the selection table, and the scoped negatives."""
    meta = versions["meta"]
    lines = [
        f"M7 FREEZE - {meta['milestone']}, {meta['frozen_on']}",
        f"corpus {meta['corpus_version']} | query set {meta['query_set_version']}",
        f"metric: {meta['scoring_metric']}",
        "",
        "=== FROZEN ===",
        "",
    ]
    for key in FROZEN_KEYS:
        entry = versions[key]
        value = entry["value"] if entry["value"] is not None else "NONE (rejected)"
        lines.append(f"  {key:20} {value}")
        lines.append(f"  {'':20} selected by: {entry['selected_by']}")
    lines += ["", "", "=== SELECTION: every lever tested and priced ===", ""]
    lines.append(f"  {'lever':12} {'configuration':38} outcome")
    lines.append("  " + "-" * 96)
    for lever, configuration, outcome in SELECTION_TABLE:
        lines.append(f"  {lever:12} {configuration:38} {outcome}")

    lines += ["", "", "=== SCOPED NEGATIVES (none of these is a universal) ===", ""]
    for name, entry in versions["negative_results"].items():
        lines += [
            f"  {name}: {entry['verdict']}",
            f"      scope   : {' '.join(entry['scope'].split())}",
            f"      revisit : {entry['revisit_when']}",
            "",
        ]

    residual = versions["residual_finding"]
    lines += [
        "=== RESIDUAL FINDING ===",
        "",
        f"  {residual['headline']}",
        f"  rows    : {', '.join(residual['rows'])}",
        f"  levers  : {' '.join(residual['levers_exhausted'].split())}",
        f"  belongs : {residual['belongs_to']}",
        f"  filed   : {residual['eval_set']} "
        f"({eval_set['meta']['case_count']} cases + "
        f"{eval_set['meta']['borderline_count']} borderline)",
    ]
    return "\n".join(lines)


def log_to_mlflow(versions: dict[str, Any], eval_set: dict[str, Any]) -> str | None:
    """Record the frozen configuration as one run. Failure to log never fails anything."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping")
        return None
    config = load_config()["mlflow"]
    meta = versions["meta"]
    try:
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="M7 freeze") as run:
            params = {
                "arm": ARM,
                "corpus_version": meta["corpus_version"],
                "query_set": meta["query_set_version"],
                "scoring_metric": meta["scoring_metric"],
            }
            for key in FROZEN_KEYS:
                params[key] = str(versions[key]["value"])
                params[f"{key}__selected_by"] = versions[key]["selected_by"]
            for name, entry in versions["negative_results"].items():
                params[f"rejected__{name}"] = entry["verdict"]
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "gap_rows": float(len(versions["residual_finding"]["rows"])),
                    "m8_eval_cases": float(eval_set["meta"]["case_count"]),
                }
            )
            table = "\n".join(f"{lever} | {cfg} | {out}" for lever, cfg, out in SELECTION_TABLE)
            mlflow.log_text(table, "selection_table.txt")
            mlflow.log_text(format_report(versions, eval_set), "freeze_report.txt")
            return run.info.run_id
    except Exception as error:  # noqa: BLE001 - logging must not fail the freeze
        log.warning("MLflow logging failed: %s", error)
        return None


def main(argv: list[str] | None = None) -> int:
    """Print the freeze block and record it."""
    parser = argparse.ArgumentParser(description="M7 freeze record.")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    args = parser.parse_args(argv)

    versions = load_versions()
    eval_set = load_m8_eval_set()
    print(format_report(versions, eval_set))
    if not args.no_mlflow:
        run_id = log_to_mlflow(versions, eval_set)
        print(f"\nMLflow {EXPERIMENT}/{ARM}: {run_id or 'not logged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
