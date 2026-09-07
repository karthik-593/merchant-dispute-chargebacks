"""M6: compare chunking strategies over the circulars corpus.

The retriever is held constant at BM25 on purpose. Choosing a retriever is M7's experiment, and
varying both at once would leave neither result attributable — a chunker that wins here has won
under one fixed, unglamorous ranking function, which is exactly the comparison M6 is for.

Two metrics matter, and they are not the same:

- **Recall@K** asks whether a chunk from the right *document* reached the top K. It is the usual
  measure and it is generous: on a 35-document corpus, retrieving the right document is not hard.
- **Rule-hit-rate** asks whether one of those chunks actually *carried the answer*, judged by the
  anchors frozen into the query set. This is the measure that discriminates. A chunker can score
  a perfect Recall@1 while every chunk it returns is a fragment that answers nothing.

Nothing here picks a winner. It reports numbers.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from rank_bm25 import BM25Okapi

from src.config import load_config
from src.logging_setup import get_logger
from src.retrieval.chunkers import Chunk, Chunker, all_chunkers, chunk_corpus
from src.retrieval.corpus_ingest import DocumentRecord, corpus_dir, read_corpus

log = get_logger(__name__)

EXPERIMENT = "M6-chunking"
QUERY_FILE = "retrieval_queries.yaml"
K_VALUES = (1, 3, 5)
TOP_K = max(K_VALUES)
_WORD = re.compile(r"[a-z0-9]+")


def load_queries(directory: Path | None = None) -> dict[str, Any]:
    """Read the frozen retrieval query set."""
    path = (directory or corpus_dir()) / QUERY_FILE
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25, split on underscores as well as whitespace.

    The corpus writes evidence types as `invoice_with_proof_of_delivery` and reason codes as
    `RC_1064`, while a question says "RC 1064" and "invoice with proof of delivery". Indexing the
    identifier whole would mean the two never meet, and the resulting scores would say something
    about tokenisation rather than about chunking.
    """
    return _WORD.findall(text.lower())


def normalise(text: str) -> str:
    """Collapse whitespace and lowercase, for anchor matching."""
    return " ".join(text.split()).lower()


@dataclass
class QueryOutcome:
    """How one query fared under one chunker."""

    query_id: str
    topic: str
    doc_hit_rank: int | None = None
    rule_hit_rank: int | None = None
    rule_hit_tokens: int | None = None

    def doc_hit_at(self, k: int) -> bool:
        """Whether a chunk from a target document reached the top k."""
        return self.doc_hit_rank is not None and self.doc_hit_rank <= k

    def rule_hit_at(self, k: int) -> bool:
        """Whether a chunk carrying the answer reached the top k."""
        return self.rule_hit_rank is not None and self.rule_hit_rank <= k


@dataclass
class ChunkerResult:
    """Everything measured for one chunking strategy."""

    name: str
    n_chunks: int
    mean_tokens: float
    mean_chars: float
    max_tokens: int
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        """Share of queries whose target document reached the top k."""
        return mean(float(o.doc_hit_at(k)) for o in self.outcomes)

    def rule_hit_at(self, k: int) -> float:
        """Share of queries where a chunk in the top k actually carried the answer."""
        return mean(float(o.rule_hit_at(k)) for o in self.outcomes)

    def mrr(self) -> float:
        """Mean reciprocal rank of the first document-level hit."""
        return mean(1.0 / o.doc_hit_rank if o.doc_hit_rank else 0.0 for o in self.outcomes)

    def mean_hit_tokens(self) -> float:
        """Average size of the chunk that carried the answer.

        Rule-hit-rate on its own rewards large chunks, because a bigger chunk contains more and so
        is likelier to hold every anchor. This is the price paid for those hits: the context a
        downstream model must read, and pay for, to get the answer.
        """
        sizes = [o.rule_hit_tokens for o in self.outcomes if o.rule_hit_tokens]
        return mean(sizes) if sizes else float("nan")

    def misses(self) -> list[QueryOutcome]:
        """Queries where no chunk in the top K carried the answer."""
        return [o for o in self.outcomes if not o.rule_hit_at(TOP_K)]


def evaluate_chunker(
    chunker: Chunker, records: list[DocumentRecord], queries: dict[str, Any]
) -> ChunkerResult:
    """Chunk the corpus, index it with BM25, and score every query."""
    chunks: list[Chunk] = chunk_corpus(records, chunker)
    index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
    normalised = [normalise(chunk.text) for chunk in chunks]

    result = ChunkerResult(
        name=chunker.name,
        n_chunks=len(chunks),
        mean_tokens=mean(c.n_tokens for c in chunks),
        mean_chars=mean(c.n_chars for c in chunks),
        max_tokens=max(c.n_tokens for c in chunks),
    )

    for query in queries["queries"]:
        scores = index.get_scores(tokenize(query["question"]))
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        targets = set(query["target_doc_ids"])
        anchors = [normalise(a) for a in query["answer_anchors"]]

        outcome = QueryOutcome(query_id=query["id"], topic=query["topic"])
        for rank, position in enumerate(ranked, start=1):
            chunk = chunks[position]
            if chunk.doc_id not in targets:
                continue
            if outcome.doc_hit_rank is None:
                outcome.doc_hit_rank = rank
            if outcome.rule_hit_rank is None and all(
                anchor in normalised[position] for anchor in anchors
            ):
                outcome.rule_hit_rank = rank
                outcome.rule_hit_tokens = chunk.n_tokens
        result.outcomes.append(outcome)
    return result


def length_split(
    records: list[DocumentRecord], queries: dict[str, Any]
) -> tuple[int, dict[str, set[str]]]:
    """Group queries by whether their target documents are short or long.

    The threshold is the corpus median, so the split describes this corpus rather than importing
    an assumption about document length from somewhere else.
    """
    sizes = sorted(record.char_count for record in records)
    threshold = sizes[len(sizes) // 2]
    by_id = {record.doc_id: record.char_count for record in records}

    groups: dict[str, set[str]] = {"short": set(), "long": set(), "mixed": set()}
    for query in queries["queries"]:
        lengths = [by_id[d] for d in query["target_doc_ids"] if d in by_id]
        if all(length <= threshold for length in lengths):
            groups["short"].add(query["id"])
        elif all(length > threshold for length in lengths):
            groups["long"].add(query["id"])
        else:
            groups["mixed"].add(query["id"])
    return threshold, groups


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
