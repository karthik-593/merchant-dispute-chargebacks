"""M7 Stage B: re-open the embedding choice on per-row hard-set evidence.

Stage 3 found that no incumbent covers the hard rows: MiniLM wins h03 and h04, e5 wins h10, bge
wins h32 and h35, and every one of them buries several. bge leads on the aggregate (0.640 over 50
discriminators) while losing individual rows badly — **the aggregate is what hid the failures**,
and it is what earlier arms selected on.

So selection here is on **per-row hard-set coverage**, not on the mean. The aggregate is still
reported, because a candidate that wins the hard rows and collapses everywhere else is not a
candidate — but it does not decide.

The question this arm exists to answer, in order:

1. Does ONE larger single encoder cover the hard rows? If so it beats an ensemble on simplicity
   and cost, and `bge-small` was an aggregate artefact.
2. If not, and different rows still need different encoders, a per-row ensemble is warranted — and
   its gain has to be quantified against tripling encode cost and index size.
3. Rows that STILL bury under every candidate are not an embedding problem at all. They are the
   parent-expansion residue, and they are reported separately so the two fixes do not get
   confused.

`structure_aware` only, for the same reason as Stage 3: the other chunkers cannot measure these
queries without the B3 ruler defect answering for them.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from src.config import load_config
from src.evaluation.eval_first_stage import FEEDER, VERIFIED_HARD_SET
from src.evaluation.retrieval_metrics import TOP_K, load_queries, score_hits
from src.logging_setup import get_logger
from src.retrieval.chunkers import StructureAwareChunker, chunk_corpus
from src.retrieval.corpus_ingest import DocumentRecord, read_corpus
from src.retrieval.retrievers import DenseRetriever, load_embedding_model

log = get_logger(__name__)

EXPERIMENT = "M7-retrieval"
ARM = "embedding-reselection"

INCUMBENTS = ("bge-small-en-v1.5", "e5-small-v2", "all-MiniLM-L6-v2")
CANDIDATES = ("bge-large-en-v1.5", "e5-large-v2", "gte-large")
ALL_ENCODERS = INCUMBENTS + CANDIDATES

PROBE_K = 200
K_RETRIEVE = 100
BURIED_PAST = 50  # a row past this is badly buried, not merely below the cut


@dataclass
class EncoderResult:
    """One encoder, judged on the hard rows and reported on the aggregate."""

    name: str
    incumbent: bool
    ranks: dict[str, int | None] = field(default_factory=dict)
    rule_at_5_hard: float = 0.0
    rule_at_5_all: float = 0.0
    query_latency_ms: float = 0.0
    index_build_seconds: float = 0.0
    vram_gb: float = 0.0
    dim: int = 0

    def in_top_k(self, k: int = TOP_K) -> list[str]:
        """Hard rows this encoder lands inside the context window."""
        return [q for q, r in self.ranks.items() if r is not None and r <= k]

    def past(self, depth: int = BURIED_PAST) -> list[str]:
        """Hard rows this encoder buries beyond `depth` - the ones no reranker rescues."""
        return [q for q, r in self.ranks.items() if r is None or r > depth]

    def reachable_at(self, depth: int = K_RETRIEVE) -> list[str]:
        """Hard rows a reranker at this retrieval depth could still reach."""
        return [q for q, r in self.ranks.items() if r is not None and r <= depth]


@dataclass
class Reselection:
    """Every encoder measured, plus the rows nothing reaches."""

    results: list[EncoderResult] = field(default_factory=list)

    def rows(self) -> list[str]:
        """The queries actually measured, in hard-set order where they overlap.

        Derived from the results rather than read off the module constant, so the reasoning below
        describes what was run instead of what was expected to run.
        """
        measured = {q for r in self.results for q in r.ranks}
        ordered = [q for q in VERIFIED_HARD_SET if q in measured]
        return ordered + sorted(measured - set(ordered))

    def best_per_row(self) -> dict[str, tuple[str, int | None]]:
        """For each hard row, the encoder that ranks it best."""
        out: dict[str, tuple[str, int | None]] = {}
        for query in self.rows():
            ranked = [
                (r.name, r.ranks.get(query))
                for r in self.results
                if r.ranks.get(query) is not None
            ]
            out[query] = min(ranked, key=lambda pair: pair[1]) if ranked else ("-", None)
        return out

    def universally_buried(self) -> list[str]:
        """Rows every encoder buries past BURIED_PAST. Not an embedding problem."""
        return [
            query
            for query, (_, rank) in self.best_per_row().items()
            if rank is None or rank > BURIED_PAST
        ]

    def single_best(self) -> EncoderResult:
        """The encoder covering most hard rows in the top 5, ties broken on fewest buried."""
        return max(self.results, key=lambda r: (len(r.in_top_k()), -len(r.past())))


def run(
    records: list[DocumentRecord] | None = None,
    queries: dict[str, Any] | None = None,
    *,
    encoders: tuple[str, ...] = ALL_ENCODERS,
    device: str | None = None,
) -> Reselection:
    """Score every encoder on the hard rows and on all discriminators."""
    import torch

    docs = records if records is not None else read_corpus()
    query_set = queries if queries is not None else load_queries()
    by_id = {q["id"]: q for q in query_set["queries"]}
    hard = [by_id[q] for q in VERIFIED_HARD_SET if q in by_id]
    discriminators = [
        q for q in query_set["queries"] if q.get("role", "discriminator") == "discriminator"
    ]
    chunks = chunk_corpus(docs, StructureAwareChunker())

    selection = Reselection()
    for name in encoders:
        load_embedding_model(name, device=device)  # warm before any clock starts
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        retriever = DenseRetriever(chunks, name, device=device)
        build = time.perf_counter() - started
        log.info("re-selection: %s", name)

        result = EncoderResult(
            name=name,
            incumbent=name in INCUMBENTS,
            index_build_seconds=build,
            dim=int(retriever.matrix.shape[1]),
            vram_gb=(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0,
        )
        for query in hard:
            result.ranks[query["id"]] = score_hits(
                retriever.search(query["question"], PROBE_K), query
            ).rule_hit_rank
        result.rule_at_5_hard = mean(
            float(score_hits(retriever.search(q["question"], TOP_K), q).rule_hit_at(TOP_K))
            for q in hard
        )
        result.rule_at_5_all = mean(
            float(score_hits(retriever.search(q["question"], TOP_K), q).rule_hit_at(TOP_K))
            for q in discriminators
        )
        questions = [q["question"] for q in discriminators]
        retriever.search(questions[0], TOP_K)
        clock = time.perf_counter()
        for question in questions:
            retriever.search(question, TOP_K)
        result.query_latency_ms = 1000.0 * (time.perf_counter() - clock) / len(questions)
        selection.results.append(result)
    return selection


# --- reporting -----------------------------------------------------------------------------------


def format_report(selection: Reselection) -> str:
    """Per-row ranks, coverage counts, and the decision logic made explicit."""
    n = len(VERIFIED_HARD_SET)
    lines = [
        f"M7 Stage B - embedding re-selection, {FEEDER}/dense, hard set n={n}",
        "Selection is on PER-ROW hard-set coverage. The aggregate is reported but does not decide:",
        "it is what hid these failures from the earlier arms.",
        "",
        "=== PER-ROW TRUE RANK (probed to 200) ===",
        "",
    ]
    header = f"  {'query':6} " + " ".join(f"{e[:17]:>17}" for e in ALL_ENCODERS) + "   best"
    lines += [header, "  " + "-" * (len(header) + 4)]
    best = selection.best_per_row()
    for query in VERIFIED_HARD_SET:
        cells = []
        for encoder in ALL_ENCODERS:
            result = next(r for r in selection.results if r.name == encoder)
            rank = result.ranks.get(query)
            cells.append(f"{('>200' if rank is None else rank)!s:>17}")
        who, rank = best[query]
        lines.append(f"  {query:6} " + " ".join(cells) + f"   {who[:16]}@{rank or '>200'}")

    lines += ["", "", "=== COVERAGE AND COST ===", ""]
    lines.append(
        f"  {'encoder':20} {'kind':10} {'dim':>5} {'top5':>5} {'past50':>7} "
        f"{'reach@100':>10} {'r@5 hard':>9} {'r@5 all50':>10} {'ms/q':>7} {'VRAM GB':>8}"
    )
    lines.append("  " + "-" * 106)
    for result in selection.results:
        lines.append(
            f"  {result.name:20} {('incumbent' if result.incumbent else 'candidate'):10} "
            f"{result.dim:>5} {len(result.in_top_k()):>5} {len(result.past()):>7} "
            f"{len(result.reachable_at()):>10} {result.rule_at_5_hard:>9.3f} "
            f"{result.rule_at_5_all:>10.3f} {result.query_latency_ms:>7.1f} {result.vram_gb:>8.2f}"
        )

    buried = selection.universally_buried()
    winner = selection.single_best()
    incumbent_best = max(
        (r for r in selection.results if r.incumbent),
        key=lambda r: (len(r.in_top_k()), -len(r.past())),
    )
    lines += [
        "",
        "",
        "=== DECISION LOGIC ===",
        "",
        f"  best single encoder on hard-row coverage : {winner.name} "
        f"({len(winner.in_top_k())}/{n} in top-5, {len(winner.past())} past {BURIED_PAST})",
        f"  best incumbent                           : {incumbent_best.name} "
        f"({len(incumbent_best.in_top_k())}/{n} in top-5, {len(incumbent_best.past())} past "
        f"{BURIED_PAST})",
        "",
        f"  rows every encoder buries past {BURIED_PAST} : "
        f"{len(buried)}: {', '.join(buried) if buried else 'none'}",
        "    -> NOT an embedding problem. Parent-expansion residue; a different encoder cannot",
        "       invent signal a 26-token row does not carry.",
        "",
    ]

    ensemble = {q for q in selection.rows() if any(
        (r.ranks.get(q) or 10**6) <= TOP_K for r in selection.results
    )}
    lines += [
        f"  union over ALL encoders (the per-row ensemble ceiling): {len(ensemble)}/{n} in top-5",
        f"    ensemble gain over the best single encoder: "
        f"{len(ensemble) - len(winner.in_top_k())} row(s)",
        "    cost: one index and one encode pass per encoder, growing with the corpus.",
    ]
    return "\n".join(lines)


def log_to_mlflow(selection: Reselection, queries: dict[str, Any]) -> list[str]:
    """Log one run per encoder. Failure to log never fails the experiment."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow is not installed; skipping")
        return []
    config = load_config()["mlflow"]
    ids = []
    try:
        mlflow.set_tracking_uri(config["tracking_uri"])
        mlflow.set_experiment(EXPERIMENT)
        for result in selection.results:
            with mlflow.start_run(run_name=f"{ARM} / {result.name}") as run:
                mlflow.log_params(
                    {
                        "arm": ARM,
                        "embedding": result.name,
                        "is_incumbent": result.incumbent,
                        "feeder_chunker": FEEDER,
                        "feeder_retriever": "dense",
                        "query_set": queries["meta"]["version"],
                        "probe_k": PROBE_K,
                        "k_retrieve": K_RETRIEVE,
                    }
                )
                mlflow.log_metrics(
                    {
                        "rule_hit_at_5_hard": result.rule_at_5_hard,
                        "rule_hit_at_5_all": result.rule_at_5_all,
                        "hard_rows_in_top5": float(len(result.in_top_k())),
                        "hard_rows_past_50": float(len(result.past())),
                        "hard_rows_reachable_at_100": float(len(result.reachable_at())),
                        "query_latency_ms": result.query_latency_ms,
                        "index_build_seconds": result.index_build_seconds,
                        "vram_gb": result.vram_gb,
                        "embedding_dim": float(result.dim),
                    }
                )
                ids.append(run.info.run_id)
    except Exception as error:  # noqa: BLE001 - logging must not fail the experiment
        log.warning("MLflow logging failed: %s", error)
    return ids


def main(argv: list[str] | None = None) -> int:
    """Run the re-selection and print the report."""
    parser = argparse.ArgumentParser(description="Embedding re-selection (M7 Stage B).")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    docs = read_corpus()
    queries = load_queries()
    selection = run(docs, queries, device=args.device)
    print(format_report(selection))

    if not args.no_mlflow:
        ids = log_to_mlflow(selection, queries)
        print(f"\nMLflow {EXPERIMENT}/{ARM}: {len(ids)} run(s) logged")
    print("\nNothing is frozen here. EMBEDDING_VERSION is set by review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
