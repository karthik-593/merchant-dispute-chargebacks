"""M7 final arm: does parent-expansion fix the buried rows, and does it unblock the reranker?

What is already settled and is not re-tested here: the encoder is not the lever (six encoders
across two size classes and three recipes fail the same eight rows; the ensemble union equals the
best single at 3/11), depth buys one row, and a reranker over bare rows had nothing to read.

So the chunk is the last lever. This arm separates two failure modes that need opposite things:

- **TERSE** - the row carries too little text to match a paraphrased question, and nothing from
  its table ranks well either. Expansion should help: it adds matchable context.
- **SIBLING-CONFUSION** - the table IS found, but a near-identical sibling ranks above the target.
  Expansion trivially satisfies the anchor test, because the parent contains the target row. That
  is not the same as solving it: a full-table parent is the B3 ruler defect arriving by another
  road - a hit that means "the table was found", not "the row was found". Reported with the
  context cost attached so the trade is visible.

Step 4 tests the one configuration the reranker arm could not: a cross-encoder over EXPANDED
candidates. That arm failed on 120-token rows because there was nothing to read. Expanded parents
are the richer text it needed, and whether it can now tell two sibling rows apart is the open
question.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from src.config import load_config
from src.evaluation.eval_first_stage import FEEDER, VERIFIED_HARD_SET
from src.evaluation.retrieval_metrics import TOP_K, load_queries, normalise, score_hits
from src.logging_setup import get_logger
from src.retrieval.chunkers import StructureAwareChunker, chunk_corpus
from src.retrieval.corpus_ingest import DocumentRecord, read_corpus
from src.retrieval.parent_expansion import (
    DOCUMENT,
    NONE,
    WINDOW,
    ExpandingRetriever,
    ParentExpander,
)
from src.retrieval.rerankers import CrossEncoderReranker, load_cross_encoder
from src.retrieval.retrievers import DenseRetriever, load_embedding_model

log = get_logger(__name__)

EXPERIMENT = "M7-retrieval"
ARM = "parent-expansion"
EMBEDDING = "bge-small-en-v1.5"
RERANKER = "ms-marco-MiniLM-L-6-v2"

PROBE_K = 200
K_RETRIEVE_RERANK = 25
WINDOW_SIZE = 2

# Identifiers by which a table row announces itself, so siblings can be found and counted.
ROW_ID = re.compile(r"(?:Reason code|Code)\s+((?:RC_)?[A-Z0-9_]+)")

# Classification thresholds, stated rather than tuned: a row is terse below this many tokens, and
# a sibling counts as "ranking well" inside this depth. Both are round numbers chosen before the
# measurement, and the evidence columns are printed so they can be second-guessed.
TERSE_TOKENS = 40
SIBLING_WELL_RANKED = 25


@dataclass
class RowDiagnosis:
    """Step 1: what kind of failure this row is, with the evidence for saying so."""

    query_id: str
    chunk_tokens: int
    chunk_text: str
    baseline_rank: int | None
    best_sibling: tuple[str, int] | None
    sibling_count: int

    @property
    def mode(self) -> str:
        """TERSE, SIBLING-CONFUSION, BOTH, or FOUND."""
        if self.baseline_rank is not None and self.baseline_rank <= TOP_K:
            return "FOUND"
        terse = self.chunk_tokens < TERSE_TOKENS
        confused = (
            self.best_sibling is not None
            and self.best_sibling[1] <= SIBLING_WELL_RANKED
            and (self.baseline_rank is None or self.best_sibling[1] < self.baseline_rank)
        )
        if terse and confused:
            return "BOTH"
        if confused:
            return "SIBLING-CONFUSION"
        return "TERSE"


@dataclass
class ExpansionResult:
    """One expansion setting, measured across the hard set and the full discriminator set."""

    mode: str
    ranks: dict[str, int | None] = field(default_factory=dict)
    rule_at_5_hard: float = 0.0
    rule_at_5_all: float = 0.0
    mean_returned_tokens: float = 0.0

    @property
    def label(self) -> str:
        """How the setting is named in tables and MLflow."""
        return {
            NONE: "baseline (row grain)",
            WINDOW: f"parent = +/-{WINDOW_SIZE} rows",
            DOCUMENT: "parent = full table",
        }[self.mode]

    @property
    def context_tokens(self) -> float:
        """What K_context returned units cost a downstream model to read."""
        return TOP_K * self.mean_returned_tokens

    def in_top_k(self) -> list[str]:
        """Hard rows inside the context window."""
        return [q for q, r in self.ranks.items() if r is not None and r <= TOP_K]


def _answering_chunk(chunks, query):
    anchors = [normalise(a) for a in query["answer_anchors"]]
    targets = set(query["target_doc_ids"])
    for chunk in chunks:
        if chunk.doc_id in targets and all(a in normalise(chunk.text) for a in anchors):
            return chunk
    return None


def _sibling_ranks(retriever, chunks, query, target_chunk) -> tuple[tuple[str, int] | None, int]:
    """Best-ranked sibling row of the target's table, and how many siblings there are."""
    if target_chunk is None:
        return None, 0
    siblings = [
        c
        for c in chunks
        if c.doc_id == target_chunk.doc_id
        and c.chunk_id != target_chunk.chunk_id
        and ROW_ID.search(c.text)
    ]
    if not siblings:
        return None, 0
    ids = {c.chunk_id for c in siblings}
    for hit in retriever.search(query["question"], PROBE_K):
        if hit.chunk.chunk_id in ids:
            label = ROW_ID.search(hit.chunk.text)
            return (label.group(1) if label else hit.chunk.chunk_id, hit.rank), len(siblings)
    return None, len(siblings)


@dataclass
class Arm:
    """Everything the arm measured."""

    diagnoses: list[RowDiagnosis] = field(default_factory=list)
    settings: list[ExpansionResult] = field(default_factory=list)
    rerank_before: dict[str, int | None] = field(default_factory=dict)
    rerank_after: dict[str, int | None] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)

    def mode_of(self, query_id: str) -> str:
        """The Step-1 classification for one row."""
        found = next((d for d in self.diagnoses if d.query_id == query_id), None)
        return found.mode if found else "?"


def run(
    records: list[DocumentRecord] | None = None,
    queries: dict[str, Any] | None = None,
    *,
    device: str | None = None,
) -> Arm:
    """Steps 1-4."""
    docs = records if records is not None else read_corpus()
    query_set = queries if queries is not None else load_queries()
    by_id = {q["id"]: q for q in query_set["queries"]}
    hard = [by_id[q] for q in VERIFIED_HARD_SET if q in by_id]
    discriminators = [
        q for q in query_set["queries"] if q.get("role", "discriminator") == "discriminator"
    ]

    load_embedding_model(EMBEDDING, device=device)
    load_cross_encoder(RERANKER, device=device)
    chunks = chunk_corpus(docs, StructureAwareChunker())
    base = DenseRetriever(chunks, EMBEDDING, device=device)
    arm = Arm(questions={q["id"]: " ".join(q["question"].split()) for q in hard})

    # --- Step 1 -------------------------------------------------------------------------------
    for query in hard:
        target = _answering_chunk(chunks, query)
        best_sibling, count = _sibling_ranks(base, chunks, query, target)
        arm.diagnoses.append(
            RowDiagnosis(
                query_id=query["id"],
                chunk_tokens=target.n_tokens if target else 0,
                chunk_text=target.text if target else "(no answering chunk)",
                baseline_rank=score_hits(base.search(query["question"], PROBE_K), query
                                         ).rule_hit_rank,
                best_sibling=best_sibling,
                sibling_count=count,
            )
        )

    # --- Steps 2 and 3 ------------------------------------------------------------------------
    for mode in (NONE, WINDOW, DOCUMENT):
        expander = ParentExpander(chunks, mode=mode, window=WINDOW_SIZE)
        retriever = ExpandingRetriever(base, expander, probe=PROBE_K)
        result = ExpansionResult(mode=mode)
        log.info("expansion setting: %s", result.label)
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
        returned = [
            hit.chunk.n_tokens
            for q in discriminators
            for hit in retriever.search(q["question"], TOP_K)
        ]
        result.mean_returned_tokens = mean(returned) if returned else 0.0
        arm.settings.append(result)

    # --- Step 4: reranker over EXPANDED candidates --------------------------------------------
    window = ExpandingRetriever(
        base, ParentExpander(chunks, mode=WINDOW, window=WINDOW_SIZE), probe=PROBE_K
    )
    reranker = CrossEncoderReranker(RERANKER, device=device)
    windowed = next(s for s in arm.settings if s.mode == WINDOW)
    unfixed = [
        q
        for q in hard
        if arm.mode_of(q["id"]) in {"SIBLING-CONFUSION", "BOTH"}
        and (windowed.ranks[q["id"]] is None or windowed.ranks[q["id"]] > TOP_K)
    ]
    for query in unfixed:
        candidates = window.search(query["question"], K_RETRIEVE_RERANK)
        arm.rerank_before[query["id"]] = score_hits(candidates, query).rule_hit_rank
        arm.rerank_after[query["id"]] = score_hits(
            reranker.rerank(query["question"], candidates), query
        ).rule_hit_rank
    return arm


# --- reporting -----------------------------------------------------------------------------------


def format_report(arm: Arm) -> str:
    """Steps 1-4, per row and per mode. No aggregate-only verdicts."""
    lines = [
        f"M7 parent-expansion arm - {FEEDER}/dense/{EMBEDDING}, hard set n={len(arm.diagnoses)}",
        "",
        "=== STEP 1: failure mode BEFORE expanding ===",
        "",
        f"  {'query':6} {'tok':>4} {'rank':>6} {'best sibling':>16} {'sibs':>5}  mode",
        "  " + "-" * 62,
    ]
    for d in arm.diagnoses:
        sibling = f"{d.best_sibling[0]}@{d.best_sibling[1]}" if d.best_sibling else "-"
        rank = d.baseline_rank if d.baseline_rank else ">200"
        lines.append(
            f"  {d.query_id:6} {d.chunk_tokens:>4} {rank!s:>6} {sibling:>16} "
            f"{d.sibling_count:>5}  {d.mode}"
        )

    lines += ["", "", "=== STEPS 2-3: rank under each expansion setting ===", ""]
    header = f"  {'query':6} {'mode':18} " + " ".join(
        f"{s.label[:22]:>22}" for s in arm.settings
    )
    lines += [header, "  " + "-" * len(header)]
    for d in arm.diagnoses:
        cells = []
        for setting in arm.settings:
            rank = setting.ranks.get(d.query_id)
            cells.append(f"{('>200' if rank is None else rank)!s:>22}")
        lines.append(f"  {d.query_id:6} {d.mode:18} " + " ".join(cells))

    lines += ["", f"  {'setting':24} {'top5/11':>8} {'r@5 hard':>9} {'r@5 all50':>10} "
              f"{'ret tok':>8} {'K_ctx tok':>10}", "  " + "-" * 74]
    for s in arm.settings:
        lines.append(
            f"  {s.label:24} {len(s.in_top_k()):>8} {s.rule_at_5_hard:>9.3f} "
            f"{s.rule_at_5_all:>10.3f} {s.mean_returned_tokens:>8.0f} {s.context_tokens:>10.0f}"
        )

    lines += ["", "", "=== PER-MODE VERDICT ===", ""]
    baseline = next(s for s in arm.settings if s.mode == NONE)
    for mode in ("TERSE", "SIBLING-CONFUSION", "BOTH", "FOUND"):
        ids = [d.query_id for d in arm.diagnoses if d.mode == mode]
        if not ids:
            continue
        lines.append(f"  {mode} ({len(ids)}): {', '.join(ids)}")
        for setting in arm.settings[1:]:
            recovered = [
                q
                for q in ids
                if (baseline.ranks[q] is None or baseline.ranks[q] > TOP_K)
                and setting.ranks[q] is not None
                and setting.ranks[q] <= TOP_K
            ]
            hurt = [
                q
                for q in ids
                if baseline.ranks[q] is not None
                and baseline.ranks[q] <= TOP_K
                and (setting.ranks[q] is None or setting.ranks[q] > TOP_K)
            ]
            lines.append(
                f"      {setting.label:24} recovered {len(recovered)}/{len(ids)}"
                f"{' ' + str(recovered) if recovered else ''}"
                f"{'   HURT ' + str(hurt) if hurt else ''}"
            )
        lines.append("")

    lines += ["=== STEP 4: cross-encoder over EXPANDED candidates ===", ""]
    if not arm.rerank_before:
        lines.append("  no sibling-confusion row was left unfixed by expansion; nothing to test.")
    else:
        lines.append(
            f"  {'query':6} {'before (expanded)':>18} {'after rerank':>14}   outcome"
        )
        lines.append("  " + "-" * 58)
        for query_id, before in arm.rerank_before.items():
            after = arm.rerank_after[query_id]
            gained = (before is None or before > TOP_K) and (after is not None and after <= TOP_K)
            lost = (before is not None and before <= TOP_K) and (after is None or after > TOP_K)
            outcome = "RESCUED" if gained else ("REGRESSED" if lost else "no change")
            lines.append(
                f"  {query_id:6} {(before or '>K')!s:>18} {(after or '>K')!s:>14}   {outcome}"
            )
    return "\n".join(lines)


def log_to_mlflow(arm: Arm, queries: dict[str, Any]) -> list[str]:
    """Log one run per expansion setting."""
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
        for setting in arm.settings:
            with mlflow.start_run(run_name=f"{ARM} / {setting.label}") as run:
                mlflow.log_params(
                    {
                        "arm": ARM,
                        "expansion_mode": setting.mode,
                        "window_size": WINDOW_SIZE,
                        "embedding": EMBEDDING,
                        "feeder_chunker": FEEDER,
                        "query_set": queries["meta"]["version"],
                        "k_context": TOP_K,
                    }
                )
                mlflow.log_metrics(
                    {
                        "rule_hit_at_5_hard": setting.rule_at_5_hard,
                        "rule_hit_at_5_all": setting.rule_at_5_all,
                        "hard_rows_in_top5": float(len(setting.in_top_k())),
                        "mean_returned_tokens": setting.mean_returned_tokens,
                        "context_tokens": setting.context_tokens,
                    }
                )
                ids.append(run.info.run_id)
    except Exception as error:  # noqa: BLE001 - logging must not fail the experiment
        log.warning("MLflow logging failed: %s", error)
    return ids


def main(argv: list[str] | None = None) -> int:
    """Run the arm and print the report."""
    parser = argparse.ArgumentParser(description="Parent-expansion arm (M7).")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment logging")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    docs = read_corpus()
    queries = load_queries()
    arm = run(docs, queries, device=args.device)
    print(format_report(arm))

    if not args.no_mlflow:
        ids = log_to_mlflow(arm, queries)
        print(f"\nMLflow {EXPERIMENT}/{ARM}: {len(ids)} run(s) logged")
    print("\nNothing is frozen here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
