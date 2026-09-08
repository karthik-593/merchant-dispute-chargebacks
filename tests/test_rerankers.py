"""Tests for the cross-encoder reranking layer.

The reranker's contract is narrow and easy to get subtly wrong: it must reorder a shortlist
without inventing, dropping or duplicating candidates, it must renumber ranks from 1, and it must
never be able to reach past what the feeder handed it. Those are asserted against a stub whose
scores are dictated, so they hold without a network.

The three depths - K_retrieve, rerank, K_context - are the thing most likely to be conflated by a
later edit, so they are pinned separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.eval_reranker import CEILING_FEEDER, REAL_FEEDERS, Movement
from src.evaluation.retrieval_metrics import DISCRIMINATOR
from src.retrieval.chunkers import SentenceChunker, WholeDocumentChunker, chunk_corpus
from src.retrieval.corpus_ingest import CORPUS_FILE, corpus_dir, read_corpus
from src.retrieval.rerankers import (
    CROSS_ENCODERS,
    K_CONTEXT,
    K_RETRIEVE,
    RerankedRetriever,
)
from src.retrieval.retrievers import BM25Retriever, RetrievalHit

CORPUS_PATH = corpus_dir() / CORPUS_FILE
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="ingested corpus not on disk (dvc pull)"
)


class StubReranker:
    """Reverses whatever it is given, so reordering is observable without a model."""

    name = "stub"

    def rerank(self, question: str, hits):  # noqa: ARG002 - fixed by construction
        """Return the candidates in reverse order, renumbered from 1."""
        return [
            RetrievalHit(rank=rank, score=float(-rank), chunk=hit.chunk)
            for rank, hit in enumerate(reversed(hits), start=1)
        ]


@pytest.fixture(scope="module")
def corpus():
    return read_corpus()


@pytest.fixture(scope="module")
def chunks(corpus):
    return chunk_corpus(corpus, SentenceChunker())


# --- the three depths stay separate ---------------------------------------------------------


def test_the_three_knobs_are_distinct_and_ordered():
    """K_retrieve must exceed K_context, or there is nothing for the reranker to reorder."""
    assert K_RETRIEVE > K_CONTEXT
    assert K_RETRIEVE == 25
    assert K_CONTEXT == 5


@needs_corpus
def test_the_feeder_hands_over_k_retrieve_and_only_k_context_survives(chunks):
    pipeline = RerankedRetriever(BM25Retriever(chunks), StubReranker())
    question = "What evidence does RC 1064 accept?"
    assert len(pipeline.candidates(question)) == K_RETRIEVE
    assert len(pipeline.search(question, K_CONTEXT)) == K_CONTEXT


@needs_corpus
def test_reranking_reorders_without_inventing_or_losing_candidates(chunks):
    """The reranker may only permute. Anything else is a bug that would silently change recall."""
    pipeline = RerankedRetriever(BM25Retriever(chunks), StubReranker())
    outcome = pipeline.rerank_outcome("chargeback cap per customer")
    before = [h.chunk.chunk_id for h in outcome.before]
    after = [h.chunk.chunk_id for h in outcome.after]
    assert sorted(before) == sorted(after)
    assert len(after) == len(set(after))
    assert after == list(reversed(before)), "the stub reverses; the pipeline must respect that"
    assert [h.rank for h in outcome.after] == list(range(1, len(after) + 1))


@needs_corpus
def test_a_reranker_cannot_reach_past_what_the_feeder_gave_it(chunks):
    """The property that makes K_retrieve a ceiling, and the safety check meaningful."""
    pipeline = RerankedRetriever(BM25Retriever(chunks), StubReranker(), k_retrieve=10)
    outcome = pipeline.rerank_outcome("deemed acceptance no response")
    assert outcome.candidates == 10
    reachable = {h.chunk.chunk_id for h in outcome.before}
    assert {h.chunk.chunk_id for h in outcome.after} <= reachable


@needs_corpus
def test_provenance_survives_reranking(chunks):
    pipeline = RerankedRetriever(BM25Retriever(chunks), StubReranker())
    for hit in pipeline.search("RGNB response time", K_CONTEXT):
        assert hit.doc_id
        assert hit.chunk.page_start >= 1
        assert hit.doc_id in hit.citation()


# --- coverage, the thing that decides whether a feeder is a peer -------------------------------


@needs_corpus
def test_coverage_share_exposes_a_feeder_that_is_not_really_retrieving(corpus):
    """whole_document at K_retrieve=25 hands over most of the corpus; that must be visible."""
    fine = RerankedRetriever(
        BM25Retriever(chunk_corpus(corpus, SentenceChunker())), StubReranker()
    )
    coarse = RerankedRetriever(
        BM25Retriever(chunk_corpus(corpus, WholeDocumentChunker())), StubReranker()
    )
    assert fine.coverage_share < 0.3, "a real selection"
    assert coarse.coverage_share > 0.6, "not a selection - most of the corpus"
    assert coarse.coverage_share <= 1.0


def test_the_ceiling_feeder_is_named_and_kept_out_of_the_peer_list():
    assert CEILING_FEEDER == "whole_document"
    assert CEILING_FEEDER not in REAL_FEEDERS
    assert set(REAL_FEEDERS) == {"structure_aware", "sentence"}


def test_both_cross_encoders_are_declared():
    assert set(CROSS_ENCODERS) == {"ms-marco-MiniLM-L-6-v2", "bge-reranker-base"}


# --- win / regression bookkeeping --------------------------------------------------------------


def _movement(before, after, role=DISCRIMINATOR):
    return Movement(
        query_id="x",
        topic="t",
        query_class="near-duplicate",
        role=role,
        before_rank=before,
        after_rank=after,
        beyond_candidates=before is None,
    )


def test_a_win_is_outside_the_context_window_before_and_inside_after():
    assert _movement(9, 2).is_win
    assert _movement(None, 3).is_win
    assert not _movement(2, 1).is_win, "already inside; an improvement is not a win"


def test_a_regression_is_inside_before_and_outside_after():
    """These must never be averaged away - a reranker that trades 4 for 1 is not an improvement."""
    assert _movement(2, 12).is_regression
    assert _movement(3, None).is_regression
    assert not _movement(9, 12).is_regression, "outside both; no ground was lost"


def test_a_movement_cannot_be_both():
    for before, after in ((9, 2), (2, 12), (1, 1), (30, 40)):
        move = _movement(before, after)
        assert not (move.is_win and move.is_regression)


def test_guard_movements_are_separable_from_discriminator_movements():
    """Guards must not land in a win or regression count that gets reported as a total."""
    guard = _movement(9, 2, role="regression_guard")
    assert guard.role != DISCRIMINATOR
    assert guard.is_win, "the property still computes; it is the aggregation that must exclude it"


@needs_corpus
def test_scores_are_finite_so_ranking_is_well_defined(chunks):
    pipeline = RerankedRetriever(BM25Retriever(chunks), StubReranker())
    scores = [h.score for h in pipeline.search("illegible evidence", K_CONTEXT)]
    assert all(np.isfinite(s) for s in scores)
