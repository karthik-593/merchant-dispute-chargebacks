"""M7: compare retrievers and embeddings over the chunked circulars corpus.

M6 varied the chunker with BM25 fixed. M7 varies the retriever and the embedding, over the
chunkers M6 left standing, and scores on the same frozen 26-query set with the same metrics — so
an M7 cell can be read directly against its M6 row.

The grid is **pruned**, not full:

- `fixed_size` is eliminated. It scored rule@1 = 0.654 in M6, the worst of the four, and won on
  no other axis: its chunks are smaller than `sentence`'s but it hit less often, and it missed
  q15 outright. Carrying it forward would have spent a third of the runs re-measuring a strategy
  already beaten on every column.
- `whole_document` stays as the baseline. It is the retrieval result to beat, and — because these
  encoders see 512 tokens and its chunks average 1,091 — it is also the cell where dense
  retrieval should be expected to fail. That failure is a result, not an accident, and it is
  reported as `truncated_share`.
- No reranker in this pass. A cross-encoder over the top-K is the obvious next lever, but it
  reranks whatever the first stage produced, so it belongs after the first stage is chosen.

**Nothing here freezes anything.** The query set is 26 items, so a hairline lead in Recall@5 is
one query, and one query is noise. The report ends by naming the cluster of cells within a small
margin of the best and ranking that cluster on the things 26 queries cannot make noisy — tokens
per hit, latency, index size, and whether a known failure was actually fixed.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from src.config import load_config
from src.evaluation.retrieval_metrics import (
    K_VALUES,
    TOP_K,
    QueryOutcome,
    ScoreCard,
    load_queries,
    score_hits,
)
from src.logging_setup import get_logger
from src.retrieval.chunkers import (
    Chunk,
    Chunker,
    SentenceChunker,
    StructureAwareChunker,
    WholeDocumentChunker,
    chunk_corpus,
)
from src.retrieval.corpus_ingest import DocumentRecord, read_corpus
from src.retrieval.retrievers import (
    DEFAULT_FUSION_ALPHA,
    EMBEDDINGS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    load_embedding_model,
    resolve_device,
)

log = get_logger(__name__)

EXPERIMENT = "M7-retrieval"

# Eliminated in the pruning, kept here so the reason travels with the code rather than only with
# the results document.
ELIMINATED = {
    "fixed_size": (
        "M6 rule@1 = 0.654, worst of the four, and no redeeming axis: smaller chunks than "
        "sentence but fewer hits, and it missed q15 outright."
    )
}

GRID_CHUNKERS = ("sentence", "structure_aware", "whole_document")
GRID_EMBEDDINGS = ("bge-small-en-v1.5", "e5-small-v2", "all-MiniLM-L6-v2")

# The two queries M6 could not answer with any chunker worth keeping. Whether a retriever fixes
# them is a per-query question, not an average, so they are reported cell by cell.
WATCHED_QUERIES = ("q13", "q15")

# Latency is measured per query after a warm-up pass, because the first search of a run pays for
# lazily-initialised CUDA kernels and a cold tokenizer — costs a served system pays once at
# start-up, not once per dispute.
LATENCY_REPEATS = 3

# How close to the best a cell has to be to count as part of the top cluster. One query out of 26
# is 0.038, so anything inside a single query of the leader is not distinguishable on this set.
CLUSTER_MARGIN = 1.0 / 26

# Every cell retrieves this deep, but scores only the top-K. The extra results are not scored and
# cannot move a metric; they exist so a miss can say *how far* it missed by. "The answer is at
# rank 7" and "the answer is at rank 81" are the same MISS in the table and entirely different
# diagnoses, and the difference is exactly what decides whether a reranker would be worth adding.
DIAGNOSTIC_K = 50


@dataclass
class GridCell(ScoreCard):
    """One (chunker, retriever, embedding, alpha) configuration and everything measured for it."""

    chunker: str = ""
    retriever: str = ""
    embedding: str = "-"
    alpha: float | None = None
    n_chunks: int = 0
    mean_tokens: float = 0.0
    max_tokens: int = 0
    truncated_share: float = 0.0
    seen_share: float = 1.0
    index_build_seconds: float = 0.0
    index_bytes: int = 0
    query_latency_ms: float = 0.0
    device: str = "cpu"

    @property
    def label(self) -> str:
        """Short name for the cell, as it appears in the tables and in MLflow."""
        parts = [self.chunker, self.retriever]
        if self.embedding != "-":
            parts.append(self.embedding)
        if self.alpha is not None:
            parts.append(f"a{self.alpha:g}")
        return " / ".join(parts)

    def watched(self, query_id: str) -> QueryOutcome | None:
        """Outcome for one of the watched failure queries."""
        return self.outcome(query_id)


@dataclass
class Grid:
    """Every cell of one pass over the grid."""

    cells: list[GridCell] = field(default_factory=list)
    queries: dict[str, Any] = field(default_factory=dict)

    def best_rule_hit(self, k: int = TOP_K) -> float:
        """The best rule-hit-rate@k anywhere in the grid."""
        return max(cell.rule_hit_at(k) for cell in self.cells)

    def cluster(self, k: int = TOP_K, margin: float = CLUSTER_MARGIN) -> list[GridCell]:
        """Cells within `margin` of the best rule-hit-rate@k.

        Ranked on tokens per hit, then latency, then index size — tiebreakers that do not depend
        on 26 hand-labelled queries, and so do not move when one of them is relabelled.
        """
        threshold = self.best_rule_hit(k) - margin - 1e-9
        inside = [cell for cell in self.cells if cell.rule_hit_at(k) >= threshold]
        return sorted(
            inside,
            key=lambda c: (c.mean_hit_tokens(), c.query_latency_ms, c.index_bytes),
        )


def grid_chunkers() -> list[Chunker]:
    """The three chunkers that survived the pruning, in reporting order."""
    return [SentenceChunker(), StructureAwareChunker(), WholeDocumentChunker()]


def _index_bytes(obj: Any) -> int:
    """Serialised size of an index, as a comparable proxy for what it costs to hold or ship.

    Pickling is not how any of these would be persisted in production, but it measures the same
    thing for a BM25 posting structure and a float32 embedding matrix, which a dimension-times-
    rows formula could only do for one of them.
    """
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as error:  # noqa: BLE001 - a size estimate must never fail a run
        log.warning("could not size index: %s", error)
        return 0


def _measure_latency(retriever: Any, questions: list[str]) -> float:
    """Mean per-query search latency in milliseconds, after one warm-up pass."""
    for question in questions[:1]:
        retriever.search(question, TOP_K)
    started = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        for question in questions:
            retriever.search(question, TOP_K)
    elapsed = time.perf_counter() - started
    return 1000.0 * elapsed / (LATENCY_REPEATS * len(questions))


def evaluate_cell(
    retriever: Any,
    chunks: list[Chunk],
    queries: dict[str, Any],
    *,
    chunker: str,
    embedding: str = "-",
    alpha: float | None = None,
    build_seconds: float = 0.0,
    index_bytes: int = 0,
    truncated_share: float = 0.0,
    seen_share: float = 1.0,
    device: str = "cpu",
) -> GridCell:
    """Score one configuration over every query."""
    cell = GridCell(
        chunker=chunker,
        retriever=retriever.name,
        embedding=embedding,
        alpha=alpha,
        n_chunks=len(chunks),
        mean_tokens=mean(c.n_tokens for c in chunks),
        max_tokens=max(c.n_tokens for c in chunks),
        truncated_share=truncated_share,
        seen_share=seen_share,
        index_build_seconds=build_seconds,
        index_bytes=index_bytes,
        device=device,
    )
    for query in queries["queries"]:
        deep = retriever.search(query["question"], DIAGNOSTIC_K)
        outcome = score_hits([hit for hit in deep if hit.rank <= TOP_K], query)
        outcome.deep_rule_hit_rank = score_hits(deep, query).rule_hit_rank
        cell.outcomes.append(outcome)
    cell.query_latency_ms = _measure_latency(
        retriever, [q["question"] for q in queries["queries"]]
    )
    return cell


def run_grid(
    records: list[DocumentRecord] | None = None,
    queries: dict[str, Any] | None = None,
    *,
    embeddings: tuple[str, ...] = GRID_EMBEDDINGS,
    alpha: float = DEFAULT_FUSION_ALPHA,
    device: str | None = None,
) -> Grid:
    """Run the pruned grid: chunkers x {BM25, dense x embeddings, hybrid x embeddings}."""
    docs = records if records is not None else read_corpus()
    query_set = queries if queries is not None else load_queries()
    resolved = resolve_device(device)
    grid = Grid(queries=query_set)

    # Load and warm every encoder before the clock starts. `DenseRetriever` loads its model in the
    # constructor, so without this the first cell to use a model pays for the download check, the
    # weights, and the CUDA warm-up — which is how a 115-chunk index came out fifteen times slower
    # to build than a 306-chunk one. Those are start-up costs, paid once per process, not per index.
    for embedding in embeddings:
        load_embedding_model(embedding, device=resolved)

    for chunker in grid_chunkers():
        chunks = chunk_corpus(docs, chunker)

        started = time.perf_counter()
        lexical = BM25Retriever(chunks)
        lexical_seconds = time.perf_counter() - started
        lexical_bytes = _index_bytes(lexical.index)

        log.info("scoring %s / bm25 (%d chunks)", chunker.name, len(chunks))
        grid.cells.append(
            evaluate_cell(
                lexical,
                chunks,
                query_set,
                chunker=chunker.name,
                build_seconds=lexical_seconds,
                index_bytes=lexical_bytes,
                device="cpu",
            )
        )

        for embedding in embeddings:
            started = time.perf_counter()
            dense = DenseRetriever(chunks, embedding, device=resolved)
            dense_seconds = time.perf_counter() - started
            dense_bytes = _index_bytes(dense.matrix)
            truncated = dense.truncated_share()
            seen = dense.seen_share()

            log.info("scoring %s / dense / %s", chunker.name, embedding)
            grid.cells.append(
                evaluate_cell(
                    dense,
                    chunks,
                    query_set,
                    chunker=chunker.name,
                    embedding=embedding,
                    build_seconds=dense_seconds,
                    index_bytes=dense_bytes,
                    truncated_share=truncated,
                    seen_share=seen,
                    device=resolved,
                )
            )

            log.info("scoring %s / hybrid / %s (alpha=%.2f)", chunker.name, embedding, alpha)
            grid.cells.append(
                evaluate_cell(
                    HybridRetriever(lexical, dense, alpha=alpha),
                    chunks,
                    query_set,
                    chunker=chunker.name,
                    embedding=embedding,
                    alpha=alpha,
                    build_seconds=lexical_seconds + dense_seconds,
                    index_bytes=lexical_bytes + dense_bytes,
                    truncated_share=truncated,
                    seen_share=seen,
                    device=resolved,
                )
            )
    return grid


def run_alpha_sweep(
    records: list[DocumentRecord] | None = None,
    queries: dict[str, Any] | None = None,
    *,
    embedding: str,
    alphas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    device: str | None = None,
) -> Grid:
    """Sweep the fusion weight, for one embedding, across every chunker.

    A hybrid cell that beats BM25 at exactly alpha=0.5 and nowhere else has not shown that fusion
    helps; it has shown that one weight on one 26-query set happened to land well. The sweep is
    cheap once the two indexes exist, and it says which of those two it is.
    """
    docs = records if records is not None else read_corpus()
    query_set = queries if queries is not None else load_queries()
    resolved = resolve_device(device)
    load_embedding_model(embedding, device=resolved)
    grid = Grid(queries=query_set)

    for chunker in grid_chunkers():
        chunks = chunk_corpus(docs, chunker)
        lexical = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedding, device=resolved)
        for alpha in alphas:
            grid.cells.append(
                evaluate_cell(
                    HybridRetriever(lexical, dense, alpha=alpha),
                    chunks,
                    query_set,
                    chunker=chunker.name,
                    embedding=embedding,
                    alpha=alpha,
                    device=resolved,
                )
            )
    return grid


# --- reporting ---------------------------------------------------------------------------------


def _grid_table(cells: list[GridCell]) -> list[str]:
    """The full grid, one row per cell."""
    lines = [
        f"{'chunker':16} {'retriever':10} {'embedding':17} {'a':>4} {'chunks':>6} "
        f"{'trunc':>6} {'seen':>6} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} "
        f"{'rule@1':>7} {'rule@3':>7} {'rule@5':>7} {'hit tok':>8} "
        f"{'build s':>8} {'idx MB':>7} {'ms/q':>7}",
        "-" * 159,
    ]
    for c in cells:
        alpha = f"{c.alpha:.1f}" if c.alpha is not None else "-"
        lines.append(
            f"{c.chunker:16} {c.retriever:10} {c.embedding:17} {alpha:>4} {c.n_chunks:>6} "
            f"{c.truncated_share:>6.2f} {c.seen_share:>6.2f} "
            f"{c.recall_at(1):>6.3f} {c.recall_at(3):>6.3f} {c.recall_at(5):>6.3f} "
            f"{c.mrr():>6.3f} "
            f"{c.rule_hit_at(1):>7.3f} {c.rule_hit_at(3):>7.3f} {c.rule_hit_at(5):>7.3f} "
            f"{c.mean_hit_tokens():>8.0f} "
            f"{c.index_build_seconds:>8.2f} {c.index_bytes / 1e6:>7.2f} {c.query_latency_ms:>7.2f}"
        )
    return lines


def _watched_table(grid: Grid) -> list[str]:
    """Per-query detail for the two queries M6 failed."""
    by_id = {q["id"]: q for q in grid.queries["queries"]}
    lines = []
    for query_id in WATCHED_QUERIES:
        query = by_id.get(query_id)
        if query is None:
            continue
        question = " ".join(query["question"].split())
        lines += [
            "",
            f"{query_id} [{query['topic']}] {question}",
            f"    anchors: {query['answer_anchors']}  target: {query['target_section']}",
            "",
            f"    {'chunker':16} {'retriever':10} {'embedding':17} {'a':>4} "
            f"{'doc rank':>9} {'rule rank':>10} {'true rank':>10} {'hit tok':>8}",
            "    " + "-" * 89,
        ]
        for cell in grid.cells:
            outcome = cell.watched(query_id)
            if outcome is None:
                continue
            alpha = f"{cell.alpha:.1f}" if cell.alpha is not None else "-"
            doc = str(outcome.doc_hit_rank) if outcome.doc_hit_rank else "miss"
            rule = str(outcome.rule_hit_rank) if outcome.rule_hit_rank else "MISS"
            tokens = str(outcome.rule_hit_tokens) if outcome.rule_hit_tokens else "-"
            deep = (
                str(outcome.deep_rule_hit_rank)
                if outcome.deep_rule_hit_rank
                else f">{DIAGNOSTIC_K}"
            )
            lines.append(
                f"    {cell.chunker:16} {cell.retriever:10} {cell.embedding:17} {alpha:>4} "
                f"{doc:>9} {rule:>10} {deep:>10} {tokens:>8}"
            )
        fixed = sum(1 for c in grid.cells if (o := c.watched(query_id)) and o.rule_hit_at(TOP_K))
        near = sum(
            1
            for c in grid.cells
            if (o := c.watched(query_id))
            and not o.rule_hit_at(TOP_K)
            and o.deep_rule_hit_rank
            and o.deep_rule_hit_rank <= 10
        )
        lines.append(
            f"    fixed at top-{TOP_K} in {fixed}/{len(grid.cells)} cells; "
            f"a further {near} have the answer inside the top 10 but below the cut"
        )
    return lines


def _cluster_table(grid: Grid) -> list[str]:
    """The top cluster, ranked on tiebreakers that do not depend on 26 queries."""
    best = grid.best_rule_hit()
    cluster = grid.cluster()
    lines = [
        f"Top cluster: rule@5 within {CLUSTER_MARGIN:.3f} (one query) of the best, {best:.3f}",
        "Ranked on tokens per hit, then latency, then index size - not on rule@5 itself, because",
        "on 26 queries the differences inside this cluster are one query wide.",
        "",
        f"  {'#':>2} {'cell':56} {'rule@5':>7} {'hit tok':>8} {'ms/q':>7} {'idx MB':>7} "
        f"{'q13':>5} {'q15':>5}",
        "  " + "-" * 102,
    ]
    for rank, cell in enumerate(cluster, start=1):
        marks = []
        for query_id in WATCHED_QUERIES:
            outcome = cell.watched(query_id)
            marks.append("yes" if outcome and outcome.rule_hit_at(TOP_K) else "no")
        lines.append(
            f"  {rank:>2} {cell.label:56} {cell.rule_hit_at(5):>7.3f} "
            f"{cell.mean_hit_tokens():>8.0f} {cell.query_latency_ms:>7.2f} "
            f"{cell.index_bytes / 1e6:>7.2f} {marks[0]:>5} {marks[1]:>5}"
        )
    return lines


def format_report(grid: Grid) -> str:
    """Render the full grid, the watched queries, the misses, and the top cluster."""
    meta = grid.queries["meta"]
    n = len(grid.queries["queries"])
    lines = [
        f"M7 retrieval x embedding grid - {n} queries ({meta['version']}), top-{TOP_K}",
        f"{len(grid.cells)} cells. Eliminated before running: "
        + "; ".join(f"{name} ({why})" for name, why in ELIMINATED.items()),
        "",
    ]
    lines += _grid_table(grid.cells)

    lines += ["", "", "The two queries M6 could not answer", "=" * 60]
    lines += _watched_table(grid)

    lines += ["", "", "Queries where no top-5 chunk carried the answer", "=" * 60]
    for cell in grid.cells:
        misses = cell.misses()
        detail = ", ".join(f"{o.query_id}[{o.topic}]" for o in misses) if misses else "none"
        lines.append(f"  {cell.label:56} {len(misses):>2}: {detail}")

    lines += ["", "", "=" * 60]
    lines += _cluster_table(grid)
    lines += [
        "",
        "Nothing is frozen here. RETRIEVER_VERSION, EMBEDDING_VERSION and CHUNKER_VERSION are",
        "set together, by review, not by this script.",
    ]
    return "\n".join(lines)


def format_alpha_report(grid: Grid) -> str:
    """Render the fusion-weight sweep."""
    lines = [
        f"Fusion-weight sweep - {grid.cells[0].embedding}. alpha 0 = pure BM25, 1 = pure dense.",
        "Does fusion win across a band of weights, or only at one lucky value?",
        "",
        f"  {'chunker':16} {'alpha':>6} {'R@5':>7} {'MRR':>7} {'rule@1':>7} {'rule@3':>7} "
        f"{'rule@5':>7} {'hit tok':>8}",
        "  " + "-" * 71,
    ]
    previous: str | None = None
    for cell in grid.cells:
        if previous is not None and cell.chunker != previous:
            lines.append("")
        previous = cell.chunker
        lines.append(
            f"  {cell.chunker:16} {cell.alpha:>6.1f} {cell.recall_at(5):>7.3f} "
            f"{cell.mrr():>7.3f} {cell.rule_hit_at(1):>7.3f} {cell.rule_hit_at(3):>7.3f} "
            f"{cell.rule_hit_at(5):>7.3f} {cell.mean_hit_tokens():>8.0f}"
        )
    return "\n".join(lines)


def log_to_mlflow(cell: GridCell, queries: dict[str, Any], group: str = "main") -> str | None:
    """Log one cell. Failure to log never fails the experiment."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping")
        return None
    config = load_config()["mlflow"]
    try:
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=cell.label) as run:
            mlflow.log_params(
                {
                    "chunker": cell.chunker,
                    "retriever": cell.retriever,
                    "embedding": cell.embedding,
                    "fusion_alpha": "-" if cell.alpha is None else f"{cell.alpha:g}",
                    "grid": group,
                    "query_set": queries["meta"]["version"],
                    "query_count": len(queries["queries"]),
                    "top_k": TOP_K,
                    "device": cell.device,
                }
            )
            metrics = {
                "n_chunks": cell.n_chunks,
                "mean_tokens": cell.mean_tokens,
                "max_tokens": cell.max_tokens,
                "truncated_share": cell.truncated_share,
                "seen_share": cell.seen_share,
                "mrr": cell.mrr(),
                "mean_hit_tokens": cell.mean_hit_tokens(),
                "index_build_seconds": cell.index_build_seconds,
                "index_bytes": cell.index_bytes,
                "query_latency_ms": cell.query_latency_ms,
            }
            for k in K_VALUES:
                metrics[f"recall_at_{k}"] = cell.recall_at(k)
                metrics[f"rule_hit_rate_at_{k}"] = cell.rule_hit_at(k)
            for query_id in WATCHED_QUERIES:
                outcome = cell.watched(query_id)
                metrics[f"{query_id}_rule_hit_at_{TOP_K}"] = float(
                    bool(outcome and outcome.rule_hit_at(TOP_K))
                )
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception as error:  # noqa: BLE001 - logging must not fail the experiment
        log.warning("MLflow logging failed: %s", error)
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the pruned grid and print the report."""
    parser = argparse.ArgumentParser(description="Compare retrievers and embeddings (M7).")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=list(GRID_EMBEDDINGS),
        choices=sorted(EMBEDDINGS),
        help="embedding models for the dense and hybrid cells",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_FUSION_ALPHA,
        help="fusion weight on the dense side for the hybrid cells",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument(
        "--alpha-sweep",
        metavar="EMBEDDING",
        choices=sorted(EMBEDDINGS),
        help="after the grid, sweep the fusion weight across every chunker for this embedding",
    )
    args = parser.parse_args(argv)

    docs = read_corpus()
    queries = load_queries()
    grid = run_grid(
        docs,
        queries,
        embeddings=tuple(args.embeddings),
        alpha=args.alpha,
        device=args.device,
    )
    print(format_report(grid))

    sweep: Grid | None = None
    if args.alpha_sweep:
        sweep = run_alpha_sweep(docs, queries, embedding=args.alpha_sweep, device=args.device)
        print("\n\n" + format_alpha_report(sweep))

    if not args.no_mlflow:
        print()
        for cell in grid.cells:
            run_id = log_to_mlflow(cell, queries, group="main")
            print(f"MLflow {EXPERIMENT}/{cell.label}: {run_id or 'not logged'}")
        for cell in sweep.cells if sweep else []:
            run_id = log_to_mlflow(cell, queries, group="alpha_sweep")
            print(f"MLflow {EXPERIMENT}/{cell.label}: {run_id or 'not logged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
