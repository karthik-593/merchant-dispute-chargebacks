"""M7 closing arm: does a cross-encoder reranker earn its place, and over which first stage?

**Pre-registered question, fixed before the run.** `structure_aware / dense / bge` retrieves the
right document often (R@5 0.780) but leaves the answer-bearing chunk at ranks 7-15 when it misses
the top 5. Prediction: a cross-encoder pulls those into the top 1 and reaches `whole_document`
level rule@1 *at 80 tokens per hit instead of 753*. If it holds, that cell is the freeze candidate
and every downstream LLM call costs a ninth of the context. If it fails - the cross-encoder cannot
rank terse curated rows either - then small structural chunks have a ranking ceiling here, and
that is the result. Nothing in this module is allowed to force the first outcome.

**`whole_document` is not a peer here.** With 36 chunks, asking for 25 candidates hands the
reranker 69% of the corpus: the first stage has selected almost nothing and what is being measured
is a cross-encoder sorting two-thirds of everything. It is run as an explicitly labelled ceiling
row, never ranked beside the real feeders, and it cannot be the winner. The `K% of corpus` column
exists so this can never again be missed by eye.

**Primary metrics are rule@1 and rule@5, not Recall@K.** The 26 carried-over queries have a known
target-listing defect that inflates Recall@K; rule-hit-rate reads the chunk text and is unaffected.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from src.config import load_config
from src.evaluation.retrieval_metrics import (
    DISCRIMINATOR,
    REGRESSION_GUARD,
    ScoreCard,
    load_queries,
    score_hits,
)
from src.logging_setup import get_logger
from src.retrieval.chunkers import (
    SentenceChunker,
    StructureAwareChunker,
    WholeDocumentChunker,
    chunk_corpus,
)
from src.retrieval.corpus_ingest import DocumentRecord, read_corpus
from src.retrieval.rerankers import (
    CROSS_ENCODERS,
    K_CONTEXT,
    K_RETRIEVE,
    CrossEncoderReranker,
    RerankedRetriever,
    load_cross_encoder,
)
from src.retrieval.retrievers import DenseRetriever, load_embedding_model

log = get_logger(__name__)

EXPERIMENT = "M7-retrieval"
ARM = "reranker"
FEEDER_EMBEDDING = "bge-small-en-v1.5"

# The two real feeders, and the ceiling row. Kept in separate lists so no loop can accidentally
# rank them together.
REAL_FEEDERS = ("structure_aware", "sentence")
CEILING_FEEDER = "whole_document"

# How deep to look when asking "could the reranker ever have found this?". Beyond K_RETRIEVE the
# answer is simply not in the candidate set and no reranker can rescue it.
PROBE_K = 60

CHUNKERS = {
    "sentence": SentenceChunker,
    "structure_aware": StructureAwareChunker,
    "whole_document": WholeDocumentChunker,
}

# Queries where the prediction lives or dies: the reconciliation-only reject rows and the RGNB
# curated rows are exactly the terse near-duplicate passages structure_aware/dense leaves at 7-15.
FOCUS = ("h32", "h34", "h35", "h36") + tuple(f"h{i:02d}" for i in (1, 2, 3, 4, 5, 9, 10))


@dataclass
class Movement:
    """Where one query's answer sat before reranking, and where it ended up."""

    query_id: str
    topic: str
    query_class: str
    role: str
    before_rank: int | None
    after_rank: int | None
    beyond_candidates: bool

    @property
    def is_win(self) -> bool:
        """Outside the context window before, inside it after."""
        return (self.before_rank is None or self.before_rank > K_CONTEXT) and (
            self.after_rank is not None and self.after_rank <= K_CONTEXT
        )

    @property
    def is_regression(self) -> bool:
        """Inside the context window before, pushed out of it by the reranker."""
        return (self.before_rank is not None and self.before_rank <= K_CONTEXT) and (
            self.after_rank is None or self.after_rank > K_CONTEXT
        )


@dataclass
class RerankCell:
    """One (feeder, cross-encoder) pair, with both stages kept separate."""

    feeder: str
    cross_encoder: str
    n_chunks: int = 0
    mean_chunk_tokens: float = 0.0
    coverage_share: float = 0.0
    is_ceiling: bool = False
    rerank_latency_ms: float = 0.0
    pre: ScoreCard = field(default_factory=ScoreCard)
    post: ScoreCard = field(default_factory=ScoreCard)
    movements: list[Movement] = field(default_factory=list)

    @property
    def label(self) -> str:
        """How the cell is named in tables and in MLflow."""
        return f"{self.feeder} / dense / {FEEDER_EMBEDDING} + {self.cross_encoder}"

    @property
    def context_tokens(self) -> float:
        """What K_CONTEXT chunks would actually cost a downstream model to read."""
        return K_CONTEXT * self.mean_chunk_tokens

    def scored_movements(self, role: str = DISCRIMINATOR) -> list[Movement]:
        """Movements for one role. Guards stay out of every count that gets reported as a total."""
        return [m for m in self.movements if m.role == role]

    def wins(self, role: str = DISCRIMINATOR) -> list[Movement]:
        """Queries the reranker pulled into the context window."""
        return [m for m in self.scored_movements(role) if m.is_win]

    def regressions(self, role: str = DISCRIMINATOR) -> list[Movement]:
        """Queries the reranker pushed out of it. Never averaged away."""
        return [m for m in self.scored_movements(role) if m.is_regression]

    def unreachable(self, role: str = DISCRIMINATOR) -> list[Movement]:
        """Answers the feeder never handed over, so no reranker could have found them."""
        return [m for m in self.scored_movements(role) if m.beyond_candidates]


def build_cell(
    records: list[DocumentRecord],
    queries: dict[str, Any],
    *,
    feeder_name: str,
    cross_encoder: str,
    k_retrieve: int = K_RETRIEVE,
    device: str | None = None,
) -> RerankCell:
    """Score one feeder x cross-encoder pair, before and after."""
    chunks = chunk_corpus(records, CHUNKERS[feeder_name]())
    dense = DenseRetriever(chunks, FEEDER_EMBEDDING, device=device)
    reranker = CrossEncoderReranker(cross_encoder, device=device)
    pipeline = RerankedRetriever(dense, reranker, k_retrieve)

    cell = RerankCell(
        feeder=feeder_name,
        cross_encoder=cross_encoder,
        n_chunks=len(chunks),
        mean_chunk_tokens=mean(c.n_tokens for c in chunks),
        coverage_share=pipeline.coverage_share,
        is_ceiling=feeder_name == CEILING_FEEDER,
    )

    elapsed = 0.0
    for query in queries["queries"]:
        probe = dense.search(query["question"], PROBE_K)
        before_deep = score_hits(probe, query).rule_hit_rank

        candidates = pipeline.candidates(query["question"])
        started = time.perf_counter()
        reranked = pipeline.reranker.rerank(query["question"], candidates)
        elapsed += time.perf_counter() - started

        cell.pre.outcomes.append(score_hits(candidates[:K_CONTEXT], query))
        cell.post.outcomes.append(score_hits(reranked[:K_CONTEXT], query))

        before_rank = score_hits(candidates, query).rule_hit_rank
        after_rank = score_hits(reranked, query).rule_hit_rank
        cell.movements.append(
            Movement(
                query_id=query["id"],
                topic=query["topic"],
                query_class=query.get("class", "unclassified"),
                role=query.get("role", DISCRIMINATOR),
                before_rank=before_rank,
                after_rank=after_rank,
                beyond_candidates=before_rank is None
                and (before_deep is None or before_deep > k_retrieve),
            )
        )
    cell.rerank_latency_ms = 1000.0 * elapsed / len(queries["queries"])
    return cell


def run_arm(
    records: list[DocumentRecord] | None = None,
    queries: dict[str, Any] | None = None,
    *,
    k_retrieve: int = K_RETRIEVE,
    device: str | None = None,
) -> list[RerankCell]:
    """Both real feeders and the ceiling row, against both cross-encoders."""
    docs = records if records is not None else read_corpus()
    query_set = queries if queries is not None else load_queries()

    # Warm everything before any clock starts.
    load_embedding_model(FEEDER_EMBEDDING, device=device)
    for name in CROSS_ENCODERS:
        load_cross_encoder(name, device=device)

    cells = []
    for feeder in (*REAL_FEEDERS, CEILING_FEEDER):
        for cross_encoder in CROSS_ENCODERS:
            log.info("scoring %s + %s", feeder, cross_encoder)
            cells.append(
                build_cell(
                    docs,
                    query_set,
                    feeder_name=feeder,
                    cross_encoder=cross_encoder,
                    k_retrieve=k_retrieve,
                    device=device,
                )
            )
    return cells


# --- reporting -----------------------------------------------------------------------------------


def _cell_table(cells: list[RerankCell], ceiling: bool) -> list[str]:
    rows = [c for c in cells if c.is_ceiling == ceiling]
    lines = [
        f"{'feeder + cross-encoder':52} {'K% corp':>8} {'pre r@1':>8} {'post r@1':>9} "
        f"{'d':>6} {'pre r@5':>8} {'post r@5':>9} {'d':>6} {'ctx tok':>8} {'ms/q':>7}",
        "-" * 128,
    ]
    for c in rows:
        pre1, post1 = c.pre.rule_hit_at(1), c.post.rule_hit_at(1)
        pre5, post5 = c.pre.rule_hit_at(5), c.post.rule_hit_at(5)
        lines.append(
            f"{c.label:52} {c.coverage_share * 100:>7.0f}% {pre1:>8.3f} {post1:>9.3f} "
            f"{post1 - pre1:>+6.3f} {pre5:>8.3f} {post5:>9.3f} {post5 - pre5:>+6.3f} "
            f"{c.context_tokens:>8.0f} {c.rerank_latency_ms:>7.1f}"
        )
    return lines


def format_report(cells: list[RerankCell], queries: dict[str, Any]) -> str:
    """Everything the arm was run to answer."""
    meta = queries["meta"]
    n = len(cells[0].pre.scored()) if cells else 0
    lines = [
        f"M7 reranker arm - {meta['version']}, discriminators only (n={n})",
        f"K_retrieve={K_RETRIEVE}  ->  cross-encoder  ->  K_context={K_CONTEXT}",
        "Primary metrics are rule@1 and rule@5. Recall@K is NOT quoted: the carried-over 26 have",
        "a known target-listing defect that inflates it, deferred to v1.3.",
        "",
        "=== REAL FEEDERS (K_retrieve is a genuine slice of the corpus) ===",
    ]
    lines += _cell_table(cells, ceiling=False)
    lines += [
        "",
        "=== CEILING ROW - NOT A RETRIEVAL RESULT ===",
        "whole_document has 36 chunks, so K_retrieve=25 hands the reranker ~69% of the corpus.",
        "The first stage selects almost nothing; this measures a cross-encoder sorting two thirds",
        "of everything. Contrast only. It does not generalise and cannot be the winner.",
        "",
    ]
    lines += _cell_table(cells, ceiling=True)

    lines += ["", "", "=== SAFETY: answers the feeder never handed over ===",
              f"Beyond rank {K_RETRIEVE} pre-rerank, so no reranker could reach them.", ""]
    for c in cells:
        missing = c.unreachable()
        detail = ", ".join(m.query_id for m in missing) if missing else "none"
        lines.append(f"  {c.label:52} {len(missing):>2}: {detail}")

    lines += ["", "", "=== PER-QUERY MOVEMENT ===", ""]
    for c in cells:
        wins, regressions = c.wins(), c.regressions()
        tag = "  [CEILING]" if c.is_ceiling else ""
        lines.append(f"{c.label}{tag}")
        lines.append(f"  WINS ({len(wins)}): pulled into top-{K_CONTEXT}")
        for m in wins:
            before = m.before_rank if m.before_rank else f">{K_RETRIEVE}"
            lines.append(f"      {m.query_id:5} {m.query_class:15} {before} -> {m.after_rank}")
        lines.append(f"  REGRESSIONS ({len(regressions)}): pushed out of top-{K_CONTEXT}")
        for m in regressions:
            after = m.after_rank if m.after_rank else f">{K_RETRIEVE}"
            lines.append(f"      {m.query_id:5} {m.query_class:15} {m.before_rank} -> {after}")
        focus = [m for m in c.scored_movements() if m.query_id in FOCUS]
        inside = sum(1 for m in focus if m.after_rank and m.after_rank <= K_CONTEXT)
        was = sum(1 for m in focus if m.before_rank and m.before_rank <= K_CONTEXT)
        lines.append(f"  FOCUS (RGNB + reconciliation-only, n={len(focus)}): {was} -> {inside}")
        lines.append("")

    lines += ["", "=== GUARDS (reported apart, never in a headline) ==="]
    for c in cells:
        marks = " ".join(
            f"{o.query_id}={'PASS' if ok else 'FAIL'}" for o, ok in c.post.guards()
        )
        lines.append(f"  {c.label:52} {marks}")
    return "\n".join(lines)


def log_to_mlflow(cell: RerankCell, queries: dict[str, Any]) -> str | None:
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
                    "arm": ARM,
                    "feeder_chunker": cell.feeder,
                    "feeder_retriever": "dense",
                    "embedding": FEEDER_EMBEDDING,
                    "cross_encoder": cell.cross_encoder,
                    "k_retrieve": K_RETRIEVE,
                    "k_context": K_CONTEXT,
                    "query_set": queries["meta"]["version"],
                    "role_scored": DISCRIMINATOR,
                    "is_ceiling_row": cell.is_ceiling,
                }
            )
            metrics = {
                "n_chunks": cell.n_chunks,
                "coverage_share": cell.coverage_share,
                "mean_chunk_tokens": cell.mean_chunk_tokens,
                "context_tokens": cell.context_tokens,
                "rerank_latency_ms": cell.rerank_latency_ms,
                "wins": len(cell.wins()),
                "regressions": len(cell.regressions()),
                "unreachable": len(cell.unreachable()),
            }
            for k in (1, 5):
                metrics[f"pre_rule_hit_rate_at_{k}"] = cell.pre.rule_hit_at(k)
                metrics[f"post_rule_hit_rate_at_{k}"] = cell.post.rule_hit_at(k)
                delta = cell.post.rule_hit_at(k) - cell.pre.rule_hit_at(k)
                metrics[f"delta_rule_hit_rate_at_{k}"] = delta
            for outcome, holds in cell.post.guards():
                metrics[f"guard_{outcome.query_id}_holds"] = float(holds)
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception as error:  # noqa: BLE001 - logging must not fail the experiment
        log.warning("MLflow logging failed: %s", error)
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the reranker arm and print the report."""
    parser = argparse.ArgumentParser(description="Cross-encoder reranker arm (M7).")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    parser.add_argument("--k-retrieve", type=int, default=K_RETRIEVE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    docs = read_corpus()
    queries = load_queries()
    cells = run_arm(docs, queries, k_retrieve=args.k_retrieve, device=args.device)
    print(format_report(cells, queries))

    if not args.no_mlflow:
        print()
        for cell in cells:
            run_id = log_to_mlflow(cell, queries)
            print(f"MLflow {EXPERIMENT}/{cell.label}: {run_id or 'not logged'}")
    print(f"\nNothing is frozen here. Guard role: {REGRESSION_GUARD}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
