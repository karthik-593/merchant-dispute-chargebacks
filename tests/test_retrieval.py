"""Tests for the M7 retrieval layer: BM25, dense, fusion, and the grid harness.

The dense cells need a downloaded encoder, so anything touching a real model is skipped unless
that model is already in the local Hugging Face cache. Fusion, ranking, provenance and scoring
are tested against a stub whose scores are chosen by hand — those are the parts that have to be
*right*, as opposed to merely measured, and they should not depend on a network.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from src.evaluation.eval_retrieval import (
    ELIMINATED,
    GRID_CHUNKERS,
    WATCHED_QUERIES,
    Grid,
    GridCell,
    evaluate_cell,
    grid_chunkers,
    run_grid,
)
from src.evaluation.retrieval_metrics import (
    TOP_K,
    load_queries,
    normalise,
    score_hits,
)
from src.retrieval.chunkers import SentenceChunker, chunk_corpus
from src.retrieval.corpus_ingest import CORPUS_FILE, corpus_dir, read_corpus
from src.retrieval.retrievers import (
    EMBEDDINGS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalHit,
    _minmax,
    tokenize,
)

CORPUS_PATH = corpus_dir() / CORPUS_FILE
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="ingested corpus not on disk (dvc pull)"
)

TEST_EMBEDDING = "all-MiniLM-L6-v2"


def _model_is_cached(embedding: str) -> bool:
    """Whether the encoder is already downloaded, so a test can use it without a network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cached = try_to_load_from_cache(EMBEDDINGS[embedding]["model_id"], "config.json")
    return isinstance(cached, str)


needs_encoder = pytest.mark.skipif(
    not _model_is_cached(TEST_EMBEDDING), reason=f"{TEST_EMBEDDING} not in the local HF cache"
)


@pytest.fixture(scope="module")
def corpus():
    return read_corpus()


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def chunks(corpus):
    return chunk_corpus(corpus, SentenceChunker())


class StubDense:
    """A dense retriever whose scores are dictated, so fusion can be checked exactly."""

    name = "dense"

    def __init__(self, chunks, scores):
        """Hold the chunks and the scores this stub will always return."""
        self.chunks = chunks
        self._scores = np.asarray(scores, dtype=np.float64)

    def scores(self, question: str) -> np.ndarray:  # noqa: ARG002 - fixed by construction
        """Return the dictated scores, whatever the question."""
        return self._scores


# --- ranking primitives ------------------------------------------------------------------------


def test_minmax_maps_to_the_unit_interval():
    scaled = _minmax(np.array([2.0, 4.0, 6.0]))
    assert scaled.tolist() == [0.0, 0.5, 1.0]


def test_minmax_of_a_flat_vector_carries_no_signal():
    """An all-equal vector ranks nothing; it must not go all-ones and outvote the other side."""
    assert _minmax(np.array([3.0, 3.0, 3.0])).tolist() == [0.0, 0.0, 0.0]


def test_tokenizer_splits_identifiers_so_queries_can_match_them():
    assert tokenize("RC_1064") == ["rc", "1064"]
    assert "delivery" in tokenize("invoice_with_proof_of_delivery")


# --- BM25 --------------------------------------------------------------------------------------


@needs_corpus
def test_bm25_returns_k_hits_ranked_best_first(chunks):
    hits = BM25Retriever(chunks).search("What evidence does RC 1064 accept?", TOP_K)
    assert len(hits) == TOP_K
    assert [h.rank for h in hits] == list(range(1, TOP_K + 1))
    assert all(a.score >= b.score for a, b in pairwise(hits))


@needs_corpus
def test_every_hit_carries_its_provenance(chunks):
    """A retrieved passage that cannot say where it came from is not usable downstream."""
    for hit in BM25Retriever(chunks).search("chargeback cap per customer", TOP_K):
        assert hit.doc_id
        assert hit.chunk.page_start >= 1
        assert hit.chunk.page_end >= hit.chunk.page_start
        assert hit.chunk.source_path
        assert hit.doc_id in hit.citation()


@needs_corpus
def test_ranking_is_reproducible(chunks):
    retriever = BM25Retriever(chunks)
    first = retriever.search("RGNB response time for P2P", TOP_K)
    second = retriever.search("RGNB response time for P2P", TOP_K)
    assert [h.chunk.chunk_id for h in first] == [h.chunk.chunk_id for h in second]


@needs_corpus
def test_ties_break_on_index_order_not_at_random(chunks):
    """Equal scores must resolve the same way every run, or the grid is not reproducible."""
    scores = np.zeros(len(chunks))
    from src.retrieval.retrievers import _top_k

    hits = _top_k(chunks, scores, 5)
    assert [h.chunk.chunk_id for h in hits] == [c.chunk_id for c in chunks[:5]]


# --- fusion ------------------------------------------------------------------------------------


@needs_corpus
def test_fusion_at_alpha_zero_is_bm25(chunks):
    lexical = BM25Retriever(chunks)
    dense = StubDense(chunks, np.random.default_rng(0).normal(size=len(chunks)))
    question = "What is the chargeback limit for a payer-payee VPA pair?"
    fused = HybridRetriever(lexical, dense, alpha=0.0).search(question, TOP_K)
    assert [h.chunk.chunk_id for h in fused] == [
        h.chunk.chunk_id for h in lexical.search(question, TOP_K)
    ]


@needs_corpus
def test_fusion_at_alpha_one_is_dense(chunks):
    lexical = BM25Retriever(chunks)
    scores = np.zeros(len(chunks))
    scores[7] = 1.0
    fused = HybridRetriever(lexical, StubDense(chunks, scores), alpha=1.0).search("anything", 1)
    assert fused[0].chunk.chunk_id == chunks[7].chunk_id


@needs_corpus
def test_fusion_weight_is_bounded(chunks):
    lexical = BM25Retriever(chunks)
    dense = StubDense(chunks, np.zeros(len(chunks)))
    for alpha in (-0.1, 1.1):
        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(lexical, dense, alpha=alpha)


@needs_corpus
def test_fusion_rejects_mismatched_indexes(chunks):
    lexical = BM25Retriever(chunks)
    with pytest.raises(ValueError, match="same chunks"):
        HybridRetriever(lexical, StubDense(chunks[:5], np.zeros(5)), alpha=0.5)


# --- scoring -----------------------------------------------------------------------------------


@needs_corpus
def test_rule_hit_requires_every_anchor(chunks):
    """One anchor out of two is not the answer; the chunk has to hold the whole thing.

    The chunk here is from the right document and does contain the first anchor, so it scores a
    document hit. It must still not score a rule hit, or the metric would be rewarding a passage
    that carries half an answer - which downstream is a citation to a rule the chunk never states.
    """
    from src.retrieval.retrievers import RetrievalHit

    chunk = chunks[0]
    present = normalise(chunk.text).split()[0]
    query = {
        "id": "synthetic",
        "topic": "synthetic",
        "target_doc_ids": [chunk.doc_id],
        "answer_anchors": [present, "no chunk anywhere in this corpus contains this"],
    }
    outcome = score_hits([RetrievalHit(rank=1, score=1.0, chunk=chunk)], query)
    assert outcome.doc_hit_rank == 1
    assert outcome.rule_hit_rank is None

    query["answer_anchors"] = [present]
    assert score_hits([RetrievalHit(rank=1, score=1.0, chunk=chunk)], query).rule_hit_rank == 1


@needs_corpus
def test_scoring_matches_m6_for_the_same_configuration(corpus, queries):
    """M7's sentence/BM25 cell is M6's sentence row. If they disagree, one of them is wrong."""
    from src.evaluation.eval_chunking import evaluate_chunker

    m6 = evaluate_chunker(SentenceChunker(), corpus, queries)
    chunks = chunk_corpus(corpus, SentenceChunker())
    m7 = evaluate_cell(BM25Retriever(chunks), chunks, queries, chunker="sentence")
    for k in (1, 3, 5):
        assert m6.recall_at(k) == pytest.approx(m7.recall_at(k))
        assert m6.rule_hit_at(k) == pytest.approx(m7.rule_hit_at(k))
    assert m6.mrr() == pytest.approx(m7.mrr())


# --- the grid ----------------------------------------------------------------------------------


def test_the_pruned_grid_drops_fixed_size_and_says_why():
    names = [c.name for c in grid_chunkers()]
    assert set(names) == set(GRID_CHUNKERS)
    assert "fixed_size" not in names
    assert "fixed_size" in ELIMINATED
    assert "0.654" in ELIMINATED["fixed_size"]


def test_watched_queries_are_the_two_known_m6_failures(queries):
    ids = {q["id"] for q in queries["queries"]}
    assert set(WATCHED_QUERIES) <= ids


def test_cluster_is_within_one_query_of_the_best_and_ranked_on_tiebreakers():
    """Selection must not turn on a hairline, so the cluster is wide and ordered by cost."""

    def cell(rule5: float, tokens: int, latency: float) -> GridCell:
        made = GridCell(chunker="c", retriever="r")
        made.rule_hit_at = lambda k=5, value=rule5: value  # type: ignore[method-assign]
        made.mean_hit_tokens = lambda value=tokens: value  # type: ignore[method-assign]
        made.query_latency_ms = latency
        return made

    grid = Grid(cells=[cell(1.0, 900, 5.0), cell(0.98, 150, 9.0), cell(0.80, 100, 1.0)])
    cluster = grid.cluster()
    assert len(cluster) == 2, "0.80 is more than one query behind and is not in the cluster"
    assert cluster[0].mean_hit_tokens() == 150, "cheaper context ranks first inside the cluster"


@needs_corpus
@needs_encoder
def test_dense_reports_how_much_it_could_not_see(corpus):
    """whole_document chunks overrun a 512-token encoder; the run must say so, not hide it."""
    from src.retrieval.chunkers import WholeDocumentChunker

    chunks = chunk_corpus(corpus, WholeDocumentChunker())
    dense = DenseRetriever(chunks, TEST_EMBEDDING)
    assert dense.max_seq_length <= 512
    assert dense.truncated_share() > 0.0


@needs_corpus
@needs_encoder
def test_dense_embeddings_are_normalised(chunks):
    dense = DenseRetriever(chunks[:16], TEST_EMBEDDING)
    norms = np.linalg.norm(dense.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@needs_corpus
@needs_encoder
def test_the_grid_runs_end_to_end_with_provenance_intact(corpus, queries):
    grid = run_grid(corpus, queries, embeddings=(TEST_EMBEDDING,))
    assert len(grid.cells) == len(GRID_CHUNKERS) * 3
    for cell in grid.cells:
        assert len(cell.outcomes) == len(queries["queries"])
        assert cell.query_latency_ms > 0.0
        assert 0.0 <= cell.rule_hit_at(5) <= cell.recall_at(5) <= 1.0
    assert grid.cluster(), "there is always at least one cell within a margin of the best"


# --- first-stage diagnosis (M7 Stage 3) --------------------------------------------------------
#
# The verdict rule decides which fix gets built, so it is worth pinning: DEPTH buys a deeper
# K_retrieve, EMBEDDING buys a different encoder, CHUNK buys parent-expansion. Confusing them
# spends effort on the wrong layer.


def _row(**ranks):
    from src.evaluation.eval_first_stage import RowResult

    return RowResult(query_id="x", topic="t", anchors=["a"], ranks=ranks)


def test_a_row_the_first_stage_already_finds_is_not_a_problem():
    assert _row(bge=1, e5=9, mini=40).verdict() == "FOUND"
    assert _row(bge=5, e5=5, mini=5).verdict() == "FOUND"


def test_one_encoder_rescuing_a_row_is_an_embedding_problem():
    """Signal exists; the wrong encoder is being asked. That is a swap, not depth or chunking."""
    assert _row(bge=90, e5=200, mini=None).verdict() == "EMBEDDING"
    assert _row(bge=10, e5=60, mini=90).verdict() == "EMBEDDING"


def test_all_three_agreeing_but_below_the_cut_is_a_depth_problem():
    assert _row(bge=30, e5=40, mini=35).verdict() == "DEPTH"


def test_all_three_burying_a_row_is_a_chunk_problem():
    """Nothing to match. No depth and no reranker invents signal that is not there."""
    assert _row(bge=None, e5=None, mini=None).verdict() == "CHUNK"
    assert _row(bge=126, e5=157, mini=129).verdict() == "CHUNK"


def test_unreachable_counts_what_a_given_depth_cannot_reach():
    row = _row(bge=30, e5=None, mini=10)
    assert row.unreachable_at(25) == {"bge": True, "e5": True, "mini": False}
    assert row.unreachable_at(50) == {"bge": False, "e5": True, "mini": False}


def test_the_verified_hard_set_is_the_source_verified_eleven():
    from src.evaluation.eval_first_stage import VERIFIED_HARD_SET

    assert set(VERIFIED_HARD_SET) == {
        "h01", "h02", "h03", "h04", "h05", "h09", "h10", "h32", "h34", "h35", "h36"
    }


def test_the_diagnosis_runs_on_structure_aware_only():
    """The other two chunkers cannot measure these queries - the B3 ruler defect."""
    from src.evaluation.eval_first_stage import FEEDER

    assert FEEDER == "structure_aware"


# --- embedding re-selection (M7 Stage B) -------------------------------------------------------


def _enc(name, incumbent=False, **ranks):
    from src.evaluation.eval_embedding_reselection import EncoderResult

    return EncoderResult(name=name, incumbent=incumbent, ranks=ranks)


def test_top5_and_buried_counts_split_reachable_from_hopeless():
    """A row at 40 is a reranker away; a row past 50 is not. The counts must not blur them."""
    result = _enc("x", a=1, b=5, c=40, d=51, e=None)
    assert set(result.in_top_k()) == {"a", "b"}
    assert set(result.past()) == {"d", "e"}
    assert set(result.reachable_at(100)) == {"a", "b", "c", "d"}, "51 is within a depth-100 cut"
    assert set(result.reachable_at(50)) == {"a", "b", "c"}


def test_a_row_every_encoder_buries_is_not_an_embedding_problem():
    """The distinction the arm exists to draw: swapping encoders cannot invent absent signal."""
    from src.evaluation.eval_embedding_reselection import Reselection

    selection = Reselection(
        results=[_enc("one", a=1, b=180), _enc("two", a=60, b=None), _enc("three", a=90, b=195)]
    )
    assert selection.universally_buried() == ["b"]
    assert selection.best_per_row()["a"] == ("one", 1)


def test_the_ensemble_ceiling_is_the_union_not_the_sum():
    """If every encoder wins the same rows, an ensemble buys nothing - that must be visible."""
    from src.evaluation.eval_embedding_reselection import Reselection

    same = Reselection(results=[_enc("one", a=1, b=90), _enc("two", a=2, b=95)])
    covered = {q for q in ("a", "b") if any((r.ranks.get(q) or 10**6) <= 5 for r in same.results)}
    assert covered == {"a"}, "two encoders agreeing on one row is still one row"


def test_the_incumbents_and_candidates_are_kept_apart():
    from src.evaluation.eval_embedding_reselection import (
        ALL_ENCODERS,
        CANDIDATES,
        INCUMBENTS,
    )

    assert not set(INCUMBENTS) & set(CANDIDATES)
    assert set(ALL_ENCODERS) == set(INCUMBENTS) | set(CANDIDATES)
    assert len(ALL_ENCODERS) == 6


def test_every_encoder_under_test_is_registered_with_its_trained_prefixes():
    from src.evaluation.eval_embedding_reselection import ALL_ENCODERS
    from src.retrieval.retrievers import EMBEDDINGS

    for name in ALL_ENCODERS:
        assert name in EMBEDDINGS, name
        spec = EMBEDDINGS[name]
        assert "model_id" in spec and "query_prefix" in spec and "passage_prefix" in spec
    # e5 needs its prefixes or it degrades; gte and MiniLM were trained without any.
    assert EMBEDDINGS["e5-large-v2"]["query_prefix"] == "query: "
    assert EMBEDDINGS["gte-large"]["query_prefix"] == ""


# --- parent expansion (M7 final arm) -----------------------------------------------------------


@needs_corpus
def test_expansion_collapses_siblings_to_one_parent(chunks):
    """The only way expansion moves a rank: many rows of a table become one unit."""
    from src.retrieval.parent_expansion import DOCUMENT, ParentExpander

    expander = ParentExpander(chunks, mode=DOCUMENT)
    hits = BM25Retriever(chunks).search("reason code evidence", 40)
    parents = expander.expand(hits)
    assert len(parents) < len(hits), "collapse is the mechanism; without it nothing moves"
    assert len({p.chunk.doc_id for p in parents}) == len(parents), "one parent per document"
    assert [p.rank for p in parents] == list(range(1, len(parents) + 1))


@needs_corpus
def test_a_parent_inherits_the_best_rank_of_its_rows(chunks):
    """A buried row becomes reachable through a well-ranked sibling. That is the whole claim."""
    from src.retrieval.parent_expansion import DOCUMENT, ParentExpander

    hits = BM25Retriever(chunks).search("deemed acceptance", 60)
    parents = ParentExpander(chunks, mode=DOCUMENT).expand(hits)
    for parent in parents:
        members = [h.rank for h in hits if h.chunk.doc_id == parent.chunk.doc_id]
        assert members, parent.chunk.doc_id
    first_doc = hits[0].chunk.doc_id
    assert parents[0].chunk.doc_id == first_doc, "the best row's document must lead"


@needs_corpus
def test_a_window_parent_stays_bounded_while_a_document_parent_does_not(corpus):
    """The bounded window is what keeps a hit meaning something about locality.

    Built on structure_aware chunks deliberately: parent expansion is a row-grain idea, and the
    module's sentence-grain fixture puts the whole reject table in two chunks, where a window and
    a document are the same thing and the test would pass without testing anything.
    """
    from src.retrieval.chunkers import StructureAwareChunker
    from src.retrieval.parent_expansion import DOCUMENT, WINDOW, ParentExpander

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    target = next(
        c
        for c in chunks
        if c.doc_id == "oc_208a_annexure_a_reject_taxonomy_curated"
        and "Reason code 1140" in c.text
    )
    hit = RetrievalHit(rank=1, score=1.0, chunk=target)
    window = ParentExpander(chunks, mode=WINDOW, window=2).expand([hit])[0]
    document = ParentExpander(chunks, mode=DOCUMENT).expand([hit])[0]
    assert window.chunk.n_tokens < document.chunk.n_tokens
    assert target.text in window.chunk.text and target.text in document.chunk.text


@needs_corpus
def test_expansion_never_loses_the_row_it_expanded(chunks):
    """A parent that dropped its own row would silently break every anchor test."""
    from src.retrieval.parent_expansion import MODES, ParentExpander

    sample = chunks[:40]
    for mode in MODES:
        expander = ParentExpander(chunks, mode=mode, window=2)
        for chunk in sample:
            parent = expander.expand([RetrievalHit(rank=1, score=1.0, chunk=chunk)])[0]
            assert chunk.text in parent.chunk.text, f"{mode} lost {chunk.chunk_id}"


@needs_corpus
def test_expansion_preserves_provenance(chunks):
    from src.retrieval.parent_expansion import WINDOW, ParentExpander

    hit = RetrievalHit(rank=1, score=1.0, chunk=chunks[10])
    parent = ParentExpander(chunks, mode=WINDOW).expand([hit])[0]
    assert parent.doc_id == chunks[10].doc_id
    assert parent.chunk.page_start >= 1
    assert parent.doc_id in parent.citation()


def test_an_unknown_expansion_mode_is_refused():
    from src.retrieval.parent_expansion import ParentExpander

    with pytest.raises(ValueError, match="unknown expansion mode"):
        ParentExpander(chunks=[], mode="parent-ish")
