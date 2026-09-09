"""M7 Stage 3: why does the first stage bury the verified hard rows - depth, embedding, or chunk?

Run on `structure_aware` ONLY. `sentence` and `whole_document` are unmeasurable on these queries
because of the B3 ruler defect: the whole 9-row RGNB table is a single chunk for both, so they
score a hit by retrieving the table without ever isolating the row. A diagnosis run there would
be measuring the ruler.

Three causes, three different fixes, and they must not be confused:

- **DEPTH** - the answer exists and ranks, just below the cut. Raising `K_retrieve` and reranking
  reaches it. Cheap.
- **EMBEDDING** - one encoder ranks the row well where another buries it. Swapping the embedding
  fixes it, and it reopens a choice earlier arms made on other evidence.
- **CHUNK** - all three encoders bury it. The row is too terse to match a paraphrased question at
  all, and no amount of depth or reranking invents signal that is not there. Parent-expansion is
  the candidate fix; this module identifies which rows need it and does NOT implement it.

Primary metric is rule@5, not Recall@K: the carried-over 26 have a known target-listing defect
that inflates Recall@K, deferred to v1.3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from src.config import load_config
from src.evaluation.retrieval_metrics import (
    TOP_K,
    load_queries,
    normalise,
    score_hits,
)
from src.logging_setup import get_logger
from src.retrieval.chunkers import StructureAwareChunker, chunk_corpus
from src.retrieval.corpus_ingest import DocumentRecord, read_corpus
from src.retrieval.retrievers import DenseRetriever, load_embedding_model

log = get_logger(__name__)

EXPERIMENT = "M7-retrieval"
ARM = "first-stage-diagnosis"
FEEDER = "structure_aware"

# The queries whose ground truth survived the source-verification pass: the RGNB rows and the four
# reject rows that exist only because of the OC 208A reconciliation.
VERIFIED_HARD_SET = (
    "h01", "h02", "h03", "h04", "h05", "h09", "h10",  # RGNB, OC 184B
    "h32", "h34", "h35", "h36",  # reject taxonomy, OC 208A
)

EMBEDDINGS = ("bge-small-en-v1.5", "e5-small-v2", "all-MiniLM-L6-v2")
DEPTHS = (25, 50, 100)
PROBE_K = 200  # deep enough that "not found" means not found, not "not looked for"

# A row counts as rescued by a different encoder when one ranks it at least this many times better
# than another. Three is deliberately blunt: on eleven queries a subtler rule would be fitting
# noise, and the per-row table is printed in full so the threshold can be second-guessed.
EMBEDDING_SPREAD = 3


@dataclass
class RowResult:
    """Where one query's answer lands under each embedding."""

    query_id: str
    topic: str
    anchors: list[str]
    ranks: dict[str, int | None] = field(default_factory=dict)

    def best(self) -> int | None:
        """Best rank across the embeddings, or None if all three lose it."""
        found = [r for r in self.ranks.values() if r is not None]
        return min(found) if found else None

    def worst(self) -> int | None:
        """Worst rank, treating a miss as worse than any rank."""
        if any(r is None for r in self.ranks.values()):
            return None
        return max(self.ranks.values())

    def unreachable_at(self, depth: int) -> dict[str, bool]:
        """Per embedding, whether the answer sits beyond this retrieval depth."""
        return {e: (r is None or r > depth) for e, r in self.ranks.items()}

    def verdict(self) -> str:
        """DEPTH, EMBEDDING, CHUNK - or FOUND when the first stage already has it."""
        best = self.best()
        if best is not None and best <= TOP_K:
            return "FOUND"
        if best is None:
            return "CHUNK"
        worst = self.worst()
        # One encoder doing markedly better than another means the signal exists and the wrong
        # encoder is being asked. That is a swap, not a depth or a chunk problem.
        if worst is None or worst >= best * EMBEDDING_SPREAD:
            return "EMBEDDING"
        if best <= max(DEPTHS):
            return "DEPTH"
        return "CHUNK"


@dataclass
class Diagnosis:
    """Everything the arm measured."""

    rows: list[RowResult] = field(default_factory=list)
    rule_at_5: dict[str, float] = field(default_factory=dict)
    rule_at_5_all: dict[str, float] = field(default_factory=dict)
    chunk_text: dict[str, tuple[str, int]] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)

    def by_verdict(self, verdict: str) -> list[RowResult]:
        """Rows sharing one diagnosis."""
        return [r for r in self.rows if r.verdict() == verdict]


def _answer_rank(retriever: DenseRetriever, query: dict[str, Any], k: int) -> int | None:
    """Rank of the first chunk that is in a target document AND carries every anchor."""
    return score_hits(retriever.search(query["question"], k), query).rule_hit_rank


def _answering_chunk(chunks, query) -> tuple[str, int] | None:
    """The chunk that carries the answer, whatever its rank. Part 3's evidence."""
    anchors = [normalise(a) for a in query["answer_anchors"]]
    targets = set(query["target_doc_ids"])
    for chunk in chunks:
        if chunk.doc_id in targets and all(a in normalise(chunk.text) for a in anchors):
            return chunk.text, chunk.n_tokens
    return None


def run(
    records: list[DocumentRecord] | None = None,
    queries: dict[str, Any] | None = None,
    *,
    embeddings: tuple[str, ...] = EMBEDDINGS,
    device: str | None = None,
) -> Diagnosis:
    """Score the verified hard set under every embedding, on structure_aware."""
    docs = records if records is not None else read_corpus()
    query_set = queries if queries is not None else load_queries()
    by_id = {q["id"]: q for q in query_set["queries"]}
    wanted = [by_id[q] for q in VERIFIED_HARD_SET if q in by_id]

    for embedding in embeddings:  # warm everything before any measurement
        load_embedding_model(embedding, device=device)

    chunks = chunk_corpus(docs, StructureAwareChunker())
    diagnosis = Diagnosis()
    diagnosis.questions = {q["id"]: " ".join(q["question"].split()) for q in wanted}
    for query in wanted:
        found = _answering_chunk(chunks, query)
        if found:
            diagnosis.chunk_text[query["id"]] = found

    rows = {q["id"]: RowResult(q["id"], q["topic"], list(q["answer_anchors"])) for q in wanted}
    discriminators = [
        q for q in query_set["queries"] if q.get("role", "discriminator") == "discriminator"
    ]

    for embedding in embeddings:
        retriever = DenseRetriever(chunks, embedding, device=device)
        log.info("diagnosing %s / dense / %s", FEEDER, embedding)
        for query in wanted:
            rows[query["id"]].ranks[embedding] = _answer_rank(retriever, query, PROBE_K)
        diagnosis.rule_at_5[embedding] = mean(
            float(score_hits(retriever.search(q["question"], TOP_K), q).rule_hit_at(TOP_K))
            for q in wanted
        )
        diagnosis.rule_at_5_all[embedding] = mean(
            float(score_hits(retriever.search(q["question"], TOP_K), q).rule_hit_at(TOP_K))
            for q in discriminators
        )

    diagnosis.rows = [rows[q] for q in VERIFIED_HARD_SET if q in rows]
    return diagnosis


# --- reporting -----------------------------------------------------------------------------------


def format_report(diagnosis: Diagnosis) -> str:
    """Parts 1-3 and the per-row verdict."""
    n = len(diagnosis.rows)
    lines = [
        f"M7 Stage 3 - first-stage diagnosis, {FEEDER} only, verified hard set (n={n})",
        "sentence and whole_document are excluded: the whole RGNB table is one chunk for both, so",
        "they score a hit without isolating the row (B3). A diagnosis there measures the ruler.",
        "",
        "=== PART 1: depth sweep ===",
        "",
        "rule@5 is a property of the top 5 and does NOT vary with K_retrieve for a first stage",
        "with no reranker - the same ranking is cut at the same place. It is reported once per",
        "embedding. What varies with depth is how many answers are out of reach entirely.",
        "",
        f"  {'embedding':22} {'rule@5 (hard set)':>18} {'rule@5 (all 50 disc)':>21}",
        "  " + "-" * 63,
    ]
    for embedding, value in diagnosis.rule_at_5.items():
        lines.append(
            f"  {embedding:22} {value:>18.3f} {diagnosis.rule_at_5_all[embedding]:>21.3f}"
        )

    lines += ["", f"  unreachable (answer beyond K_retrieve), out of {n}:", ""]
    lines.append(f"  {'embedding':22} " + " ".join(f"K={d:<7}" for d in DEPTHS))
    lines.append("  " + "-" * 55)
    for embedding in diagnosis.rule_at_5:
        cells = []
        for depth in DEPTHS:
            count = sum(1 for r in diagnosis.rows if r.unreachable_at(depth)[embedding])
            cells.append(f"{count:<9}")
        lines.append(f"  {embedding:22} " + " ".join(cells))

    lines += ["", "", "=== PART 2: per-row rank forensics (probed to 200) ===", ""]
    lines.append(
        f"  {'query':6} {'anchor(s)':52} {'bge':>6} {'e5':>6} {'MiniLM':>7}  verdict"
    )
    lines.append("  " + "-" * 96)
    for row in diagnosis.rows:
        cells = [
            f"{(row.ranks.get(e) if row.ranks.get(e) is not None else '>200')!s:>6}"
            for e in EMBEDDINGS
        ]
        anchors = ", ".join(row.anchors)
        lines.append(
            f"  {row.query_id:6} {anchors[:52]:52} {cells[0]} {cells[1]} {cells[2]:>7}  "
            f"{row.verdict()}"
        )

    buried = diagnosis.by_verdict("CHUNK")
    lines += [
        "",
        "",
        "=== PART 3: the rows every embedding buries ===",
        "",
        "Does the chunk carry enough text to plausibly match a paraphrased question, or is it a",
        "bare identifier and a few words? Evidence for the parent-expansion question, which is",
        "NOT implemented here.",
    ]
    if not buried:
        lines.append("\n  none - no row is buried by all three embeddings.")
    for row in buried:
        text, tokens = diagnosis.chunk_text.get(row.query_id, ("(no answering chunk found)", 0))
        lines += [
            "",
            f"  {row.query_id} [{row.topic}]  ranks: "
            + ", ".join(f"{e}={row.ranks.get(e) or '>200'}" for e in EMBEDDINGS),
            f"    question: {diagnosis.questions.get(row.query_id, '')}",
            f"    anchors : {row.anchors}",
            f"    chunk   : {tokens} tokens",
            f"      {text}",
        ]

    lines += ["", "", "=== VERDICTS ===", ""]
    for verdict in ("FOUND", "DEPTH", "EMBEDDING", "CHUNK"):
        ids = [r.query_id for r in diagnosis.by_verdict(verdict)]
        lines.append(f"  {verdict:10} {len(ids):>2}: {', '.join(ids) if ids else '-'}")
    return "\n".join(lines)


def log_to_mlflow(diagnosis: Diagnosis, queries: dict[str, Any]) -> str | None:
    """Log the arm. Failure to log never fails the experiment."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping")
        return None
    config = load_config()["mlflow"]
    try:
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"{ARM} / {FEEDER}") as run:
            mlflow.log_params(
                {
                    "arm": ARM,
                    "feeder_chunker": FEEDER,
                    "feeder_retriever": "dense",
                    "query_set": queries["meta"]["version"],
                    "hard_set_size": len(diagnosis.rows),
                    "probe_k": PROBE_K,
                    "embeddings": ",".join(EMBEDDINGS),
                }
            )
            metrics: dict[str, float] = {}
            for embedding, value in diagnosis.rule_at_5.items():
                key = embedding.replace("-", "_").replace(".", "_")
                metrics[f"rule_hit_at_5_hard__{key}"] = value
                metrics[f"rule_hit_at_5_all__{key}"] = diagnosis.rule_at_5_all[embedding]
                for depth in DEPTHS:
                    unreachable = sum(
                        1 for r in diagnosis.rows if r.unreachable_at(depth)[embedding]
                    )
                    metrics[f"unreachable_k{depth}__{key}"] = float(unreachable)
            for verdict in ("FOUND", "DEPTH", "EMBEDDING", "CHUNK"):
                metrics[f"verdict_{verdict.lower()}"] = float(len(diagnosis.by_verdict(verdict)))
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception as error:  # noqa: BLE001 - logging must not fail the experiment
        log.warning("MLflow logging failed: %s", error)
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the diagnosis and print the report."""
    parser = argparse.ArgumentParser(description="First-stage diagnosis (M7 Stage 3).")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    docs = read_corpus()
    queries = load_queries()
    diagnosis = run(docs, queries, device=args.device)
    print(format_report(diagnosis))

    if not args.no_mlflow:
        run_id = log_to_mlflow(diagnosis, queries)
        print(f"\nMLflow {EXPERIMENT}/{ARM}: {run_id or 'not logged'}")
    print("\nNothing is frozen here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
