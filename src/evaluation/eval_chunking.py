"""M6: compare chunking strategies over the circulars corpus.

The retriever is held constant at BM25 on purpose. Choosing a retriever is M7's experiment, and
varying both at once would leave neither result attributable — a chunker that wins here has won
under one fixed, unglamorous ranking function, which is exactly the comparison M6 is for.

The metrics live in `retrieval_metrics`, shared with M7 so the two experiments are read on one
scale. Recall@K asks whether the right document was retrieved; rule-hit-rate@K asks whether a
retrieved chunk actually carried the answer. The second is the one that discriminates.

Nothing here picks a winner. It reports numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.config import load_config
from src.evaluation.retrieval_metrics import (
    K_VALUES,
    TOP_K,
    ScoreCard,
    length_split,
    load_queries,
    normalise,
    score_hits,
)
from src.logging_setup import get_logger
from src.retrieval.chunkers import Chunk, Chunker, all_chunkers, chunk_corpus
from src.retrieval.corpus_ingest import DocumentRecord, read_corpus
from src.retrieval.retrievers import BM25Retriever, tokenize

log = get_logger(__name__)

EXPERIMENT = "M6-chunking"

__all__ = [
    "ChunkerResult",
    "evaluate_chunker",
    "length_split",
    "load_queries",
    "main",
    "normalise",
    "run",
    "tokenize",
]


@dataclass
class ChunkerResult(ScoreCard):
    """Everything measured for one chunking strategy."""

    name: str = ""
    n_chunks: int = 0
    mean_tokens: float = 0.0
    mean_chars: float = 0.0
    max_tokens: int = 0


def evaluate_chunker(
    chunker: Chunker, records: list[DocumentRecord], queries: dict[str, Any]
) -> ChunkerResult:
    """Chunk the corpus, index it with BM25, and score every query."""
    chunks: list[Chunk] = chunk_corpus(records, chunker)
    retriever = BM25Retriever(chunks)

    result = ChunkerResult(
        name=chunker.name,
        n_chunks=len(chunks),
        mean_tokens=mean(c.n_tokens for c in chunks),
        mean_chars=mean(c.n_chars for c in chunks),
        max_tokens=max(c.n_tokens for c in chunks),
    )
    for query in queries["queries"]:
        result.outcomes.append(score_hits(retriever.search(query["question"], TOP_K), query))
    return result


def format_report(
    results: list[ChunkerResult],
    threshold: int,
    groups: dict[str, set[str]],
    queries: dict[str, Any],
) -> str:
    """Render the comparison table, the length split and the per-chunker misses."""
    version = queries["meta"]["version"]
    n = len(queries["queries"])
    lines = [
        f"M6 chunking comparison - {n} queries ({version}), BM25 held constant, top-{TOP_K}",
        "",
        f"{'chunker':17} {'chunks':>7} {'mean tok':>9} {'max tok':>8} "
        f"{'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} "
        f"{'rule@1':>7} {'rule@3':>7} {'rule@5':>7} {'hit tok':>8}",
        "-" * 113,
    ]
    for r in results:
        lines.append(
            f"{r.name:17} {r.n_chunks:>7} {r.mean_tokens:>9.1f} {r.max_tokens:>8} "
            f"{r.recall_at(1):>6.3f} {r.recall_at(3):>6.3f} {r.recall_at(5):>6.3f} "
            f"{r.mrr():>6.3f} "
            f"{r.rule_hit_at(1):>7.3f} {r.rule_hit_at(3):>7.3f} {r.rule_hit_at(5):>7.3f} "
            f"{r.mean_hit_tokens():>8.0f}"
        )

    lines += [
        "",
        f"Rule-hit-rate by target document length (corpus median = {threshold:,} chars)",
        "  short = every target at or below the median; long = every target above it; mixed = both",
        "",
        f"  {'chunker':17} "
        + " ".join(f"{name} (n={len(ids)})".rjust(14) for name, ids in groups.items()),
        "  " + "-" * 66,
    ]
    for r in results:
        cells = []
        for ids in groups.values():
            subset = [o for o in r.outcomes if o.query_id in ids]
            value = mean(float(o.rule_hit_at(TOP_K)) for o in subset) if subset else float("nan")
            cells.append(f"{value:>14.3f}")
        lines.append(f"  {r.name:17} " + " ".join(cells))

    lines += ["", "Queries where no top-5 chunk carried the answer", "-" * 60]
    for r in results:
        misses = r.misses()
        if not misses:
            lines.append(f"  {r.name:17} none")
            continue
        detail = ", ".join(f"{o.query_id}[{o.topic}]" for o in misses)
        lines.append(f"  {r.name:17} {len(misses):>2}: {detail}")

    lines += ["", "Topic breakdown of rule hits at top-5", "-" * 60]
    topics = sorted({q["topic"] for q in queries["queries"]})
    lines.append(f"  {'chunker':17} " + " ".join(t[:11].rjust(12) for t in topics))
    for r in results:
        cells = []
        for topic in topics:
            subset = [o for o in r.outcomes if o.topic == topic]
            value = mean(float(o.rule_hit_at(TOP_K)) for o in subset)
            cells.append(f"{value:>12.2f}")
        lines.append(f"  {r.name:17} " + " ".join(cells))
    return "\n".join(lines)


def log_to_mlflow(result: ChunkerResult, queries: dict[str, Any]) -> str | None:
    """Log one chunker's run. Failure to log never fails the experiment."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping")
        return None
    config = load_config()["mlflow"]
    try:
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=result.name) as run:
            mlflow.log_params(
                {
                    "chunker": result.name,
                    "query_set": queries["meta"]["version"],
                    "query_count": len(queries["queries"]),
                    "retriever": "bm25_okapi",
                    "top_k": TOP_K,
                }
            )
            metrics = {
                "n_chunks": result.n_chunks,
                "mean_tokens": result.mean_tokens,
                "mean_chars": result.mean_chars,
                "max_tokens": result.max_tokens,
                "mrr": result.mrr(),
                "mean_hit_tokens": result.mean_hit_tokens(),
            }
            for k in K_VALUES:
                metrics[f"recall_at_{k}"] = result.recall_at(k)
                metrics[f"rule_hit_rate_at_{k}"] = result.rule_hit_at(k)
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception as error:  # noqa: BLE001 - logging must not fail the experiment
        log.warning("MLflow logging failed: %s", error)
        return None


def run(records: list[DocumentRecord] | None = None) -> list[ChunkerResult]:
    """Evaluate every chunker."""
    docs = records if records is not None else read_corpus()
    queries = load_queries()
    return [evaluate_chunker(c, docs, queries) for c in all_chunkers()]


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and print the report."""
    parser = argparse.ArgumentParser(description="Compare chunking strategies (M6).")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    args = parser.parse_args(argv)

    docs = read_corpus()
    queries = load_queries()
    results = [evaluate_chunker(c, docs, queries) for c in all_chunkers()]
    threshold, groups = length_split(docs, queries)

    print(format_report(results, threshold, groups, queries))

    if not args.no_mlflow:
        print()
        for result in results:
            run_id = log_to_mlflow(result, queries)
            print(f"MLflow {EXPERIMENT}/{result.name}: {run_id or 'not logged'}")
    print("\nNo winner is frozen here. CHUNKER_VERSION is set by review, not by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
