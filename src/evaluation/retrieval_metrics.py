"""Scoring shared by the chunking (M6) and retrieval (M7) experiments.

Two metrics, and they are not the same:

- **Recall@K** asks whether a chunk from the right *document* reached the top K. It is the usual
  measure and it is generous: on a 35-document corpus, retrieving the right document is not hard.
- **Rule-hit-rate@K** asks whether one of those chunks actually *carried the answer*, judged by
  the anchors frozen into the query set. This is the measure that discriminates. A run can score
  a perfect Recall@1 while every chunk it returns is a fragment that answers nothing.

Both experiments score the same way against the same frozen query set, so an M7 number can be
read next to an M6 number without an asterisk. Keeping one definition of the metric is the reason
this module exists at all; two copies would drift, and the first sign of the drift would be a
comparison that quietly stopped meaning anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from src.retrieval.corpus_ingest import DocumentRecord, corpus_dir
from src.retrieval.retrievers import RetrievalHit

QUERY_FILE = "retrieval_queries.yaml"
K_VALUES = (1, 3, 5)
TOP_K = max(K_VALUES)

# Two roles, kept apart everywhere a headline is derived.
#
# A DISCRIMINATOR is there to separate configurations: it is hard, and how often it is answered is
# the experiment's signal. A REGRESSION_GUARD is a tripwire - the system already answers it, and
# the only interesting outcome is it breaking. Averaging the two would let a guard that everything
# passes inflate every cell by the same amount and quietly shrink the margins the experiment
# exists to measure, so aggregates default to discriminators only.
DISCRIMINATOR = "discriminator"
REGRESSION_GUARD = "regression_guard"

# The fields that existed before roles and classes were added. Hashing only these gives a
# continuity check across a schema change: adding descriptive metadata to a query does not alter
# what it scores, and the hash should say so.
SCORING_FIELDS = (
    "id",
    "question",
    "topic",
    "target_doc_ids",
    "target_section",
    "answer_anchors",
)


def load_queries(directory: Path | None = None) -> dict[str, Any]:
    """Read the frozen retrieval query set."""
    path = (directory or corpus_dir()) / QUERY_FILE
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalise(text: str) -> str:
    """Collapse whitespace and lowercase, for anchor matching."""
    return " ".join(text.split()).lower()


def query_fingerprint(query: dict[str, Any]) -> str:
    """Content hash of one query: its meaning, not its formatting.

    Hashes the loaded structure rather than the YAML bytes, so re-wrapping a folded question or
    moving a comment does not read as an amendment, while any change to an id, question, target,
    anchor or topic does. That is the property the freeze actually needs: the set is frozen as a
    scoring target, not as a file layout.
    """
    canonical = json.dumps(query, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def query_set_fingerprint(queries: list[dict[str, Any]]) -> str:
    """Content hash of a whole query set, in file order."""
    canonical = json.dumps(queries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scoring_fingerprint(queries: list[dict[str, Any]]) -> str:
    """Hash over only the fields that decide scoring.

    Used to prove that carrying queries across a schema change left their meaning alone: v1.2 adds
    `class` and `role` to every query, which moves the full fingerprint of all of them while
    changing what none of them scores. The narrow hash is the honest continuity claim.
    """
    reduced = [{k: q[k] for k in SCORING_FIELDS if k in q} for q in queries]
    return query_set_fingerprint(reduced)


@dataclass
class QueryOutcome:
    """How one query fared under one configuration."""

    query_id: str
    topic: str
    role: str = DISCRIMINATOR
    query_class: str = "unclassified"
    doc_hit_rank: int | None = None
    rule_hit_rank: int | None = None
    rule_hit_tokens: int | None = None
    deep_rule_hit_rank: int | None = None
    """Where the answering chunk really sits, looked up past the top-K cut.

    Diagnostic only - no metric reads it. A query that scores MISS at 5 because its answer is at
    rank 7 and one that misses because its answer is at rank 81 are the same number in the table
    and completely different problems: the first is a cutoff away from working, the second means
    the ranking never found it at all.
    """

    def doc_hit_at(self, k: int) -> bool:
        """Whether a chunk from a target document reached the top k."""
        return self.doc_hit_rank is not None and self.doc_hit_rank <= k

    def rule_hit_at(self, k: int) -> bool:
        """Whether a chunk carrying the answer reached the top k."""
        return self.rule_hit_rank is not None and self.rule_hit_rank <= k


def score_hits(hits: list[RetrievalHit], query: dict[str, Any]) -> QueryOutcome:
    """Judge one ranked result list against one query's targets and anchors.

    A hit counts at the document level when it comes from a target document, and at the rule level
    only when its text contains *every* anchor. The anchors are literal strings, so "the answer is
    in there somewhere" is not enough: the chunk has to hold the whole answer at once, which is
    what a downstream model would need in order to cite it.
    """
    targets = set(query["target_doc_ids"])
    anchors = [normalise(anchor) for anchor in query["answer_anchors"]]
    outcome = QueryOutcome(
        query_id=query["id"],
        topic=query["topic"],
        role=query.get("role", DISCRIMINATOR),
        query_class=query.get("class", "unclassified"),
    )

    for hit in hits:
        if hit.doc_id not in targets:
            continue
        if outcome.doc_hit_rank is None:
            outcome.doc_hit_rank = hit.rank
        if outcome.rule_hit_rank is None:
            text = normalise(hit.chunk.text)
            if all(anchor in text for anchor in anchors):
                outcome.rule_hit_rank = hit.rank
                outcome.rule_hit_tokens = hit.chunk.n_tokens
    return outcome


@dataclass
class ScoreCard:
    """Aggregate metrics over every query for one configuration."""

    outcomes: list[QueryOutcome] = field(default_factory=list)

    def scored(self, role: str | None = DISCRIMINATOR) -> list[QueryOutcome]:
        """Outcomes an aggregate should average over.

        Defaults to discriminators. Pass `role=None` for every query, or REGRESSION_GUARD to look
        at the tripwires - which is a separate report, never a headline.
        """
        if role is None:
            return list(self.outcomes)
        return [o for o in self.outcomes if o.role == role]

    def recall_at(self, k: int, role: str | None = DISCRIMINATOR) -> float:
        """Share of queries whose target document reached the top k."""
        subset = self.scored(role)
        return mean(float(o.doc_hit_at(k)) for o in subset) if subset else float("nan")

    def rule_hit_at(self, k: int, role: str | None = DISCRIMINATOR) -> float:
        """Share of queries where a chunk in the top k actually carried the answer."""
        subset = self.scored(role)
        return mean(float(o.rule_hit_at(k)) for o in subset) if subset else float("nan")

    def mrr(self, role: str | None = DISCRIMINATOR) -> float:
        """Mean reciprocal rank of the first document-level hit."""
        subset = self.scored(role)
        if not subset:
            return float("nan")
        return mean(1.0 / o.doc_hit_rank if o.doc_hit_rank else 0.0 for o in subset)

    def mean_hit_tokens(self, role: str | None = DISCRIMINATOR) -> float:
        """Average size of the chunk that carried the answer.

        Rule-hit-rate on its own rewards large chunks, because a bigger chunk contains more and so
        is likelier to hold every anchor. This is the price paid for those hits: the context a
        downstream model must read, and pay for, to get the answer.
        """
        sizes = [o.rule_hit_tokens for o in self.scored(role) if o.rule_hit_tokens]
        return mean(sizes) if sizes else float("nan")

    def guards(self, k: int = TOP_K) -> list[tuple[QueryOutcome, bool]]:
        """Every tripwire and whether it still holds. Reported apart from the metrics."""
        return [(o, o.rule_hit_at(k)) for o in self.scored(REGRESSION_GUARD)]

    def misses(self, k: int = TOP_K, role: str | None = DISCRIMINATOR) -> list[QueryOutcome]:
        """Queries where no chunk in the top k carried the answer."""
        return [o for o in self.scored(role) if not o.rule_hit_at(k)]

    def outcome(self, query_id: str) -> QueryOutcome | None:
        """The outcome for one query, by id."""
        return next((o for o in self.outcomes if o.query_id == query_id), None)


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
