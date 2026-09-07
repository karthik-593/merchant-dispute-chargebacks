"""Tests for the frozen retrieval query set and the four chunking strategies."""

from __future__ import annotations

import pytest

from src.evaluation.eval_chunking import (
    evaluate_chunker,
    length_split,
    load_queries,
    normalise,
    tokenize,
)
from src.retrieval.chunkers import (
    FixedSizeChunker,
    StructureAwareChunker,
    WholeDocumentChunker,
    all_chunkers,
    chunk_corpus,
)
from src.retrieval.corpus_ingest import CORPUS_FILE, corpus_dir, read_corpus

CORPUS_PATH = corpus_dir() / CORPUS_FILE
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="ingested corpus not on disk (dvc pull)"
)


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def corpus():
    return read_corpus()


# --- the query set ---------------------------------------------------------------------------


def test_query_set_is_frozen_and_versioned(queries):
    assert queries["meta"]["version"] == "retrieval-v1"
    assert queries["meta"]["frozen"] is True
    assert queries["meta"]["query_count"] == len(queries["queries"])
    assert 20 <= len(queries["queries"]) <= 30


def test_every_query_has_a_question_and_at_least_one_target(queries):
    for query in queries["queries"]:
        assert query["question"].strip(), query["id"]
        assert len(query["question"].split()) >= 5, query["id"]
        assert query["target_doc_ids"], query["id"]
        assert query["target_section"].strip(), query["id"]
        assert query["answer_anchors"], query["id"]


def test_query_ids_are_unique(queries):
    ids = [q["id"] for q in queries["queries"]]
    assert len(ids) == len(set(ids))


@needs_corpus
def test_every_target_doc_id_exists_in_the_corpus(queries, corpus):
    known = {record.doc_id for record in corpus}
    for query in queries["queries"]:
        missing = [d for d in query["target_doc_ids"] if d not in known]
        assert not missing, f"{query['id']} targets unknown doc_ids: {missing}"


@needs_corpus
def test_every_query_is_answerable_from_at_least_one_target(queries, corpus):
    """A query whose answer is in none of its targets scores nothing and measures nothing."""
    text = {record.doc_id: normalise(record.text) for record in corpus}
    for query in queries["queries"]:
        answerable = any(
            all(normalise(a) in text[d] for a in query["answer_anchors"])
            for d in query["target_doc_ids"]
            if d in text
        )
        assert answerable, f"{query['id']}: no target document contains all its anchors"


@needs_corpus
def test_the_required_topics_are_covered(queries):
    topics = {q["topic"] for q in queries["queries"]}
    assert {
        "evidence_by_reason_code",
        "rgnb",
        "caps",
        "tat",
        "fees",
        "evidence_window",
        "small_offline",
        "deemed_approval",
        "reject_reasons",
        "compensation",
    } <= topics


# --- chunkers --------------------------------------------------------------------------------


@needs_corpus
def test_every_chunk_can_cite_its_source(corpus):
    """A chunk that cannot name its document and page is not usable as evidence."""
    for chunker in all_chunkers():
        for chunk in chunk_corpus(corpus, chunker):
            assert chunk.doc_id, chunk.chunk_id
            assert chunk.source_path, chunk.chunk_id
            assert chunk.page_start >= 1, chunk.chunk_id
            assert chunk.page_end >= chunk.page_start, chunk.chunk_id
            assert chunk.text.strip(), chunk.chunk_id
            assert chunk.doc_id in chunk.citation()


@needs_corpus
def test_chunk_ids_are_unique_within_a_strategy(corpus):
    for chunker in all_chunkers():
        ids = [c.chunk_id for c in chunk_corpus(corpus, chunker)]
        assert len(ids) == len(set(ids)), chunker.name


@needs_corpus
def test_chunk_pages_stay_inside_their_document(corpus):
    by_id = {record.doc_id: record for record in corpus}
    for chunker in all_chunkers():
        for chunk in chunk_corpus(corpus, chunker):
            assert chunk.page_end <= by_id[chunk.doc_id].page_count, chunk.chunk_id


@needs_corpus
def test_whole_document_produces_one_chunk_per_record(corpus):
    chunks = chunk_corpus(corpus, WholeDocumentChunker())
    assert len(chunks) == len(corpus)
    assert {c.doc_id for c in chunks} == {r.doc_id for r in corpus}


@needs_corpus
def test_fixed_size_respects_its_window_and_overlaps(corpus):
    chunker = FixedSizeChunker()
    chunks = chunk_corpus(corpus, chunker)
    assert all(c.n_tokens <= chunker.size for c in chunks)
    assert chunker.step < chunker.size, "25% overlap means the stride is shorter than the window"


@needs_corpus
def test_structure_aware_isolates_one_reason_code_per_chunk(corpus):
    """The curated evidence map should split where the table splits, not mid-entry."""
    record = next(r for r in corpus if r.doc_id == "oc_208_sec_c_evidence_map_curated")
    chunks = StructureAwareChunker().chunk(record)
    starts = [c for c in chunks if c.text.lower().startswith("reason code")]
    assert len(starts) >= 10, "expected roughly one chunk per reason-code entry"
    for chunk in starts:
        assert chunk.text.lower().count("reason code") == 1, chunk.chunk_id


@needs_corpus
def test_structure_aware_isolates_rgnb_rows(corpus):
    record = next(r for r in corpus if r.doc_id == "oc_184b_rgnb_response_table_curated")
    chunks = StructureAwareChunker().chunk(record)
    assert len(chunks) >= 5
    assert any("ND1" in c.text for c in chunks)
    assert any("NA2" in c.text for c in chunks)


@needs_corpus
def test_no_chunker_loses_a_document(corpus):
    """Every document must survive chunking, or the corpus silently shrinks."""
    for chunker in all_chunkers():
        covered = {c.doc_id for c in chunk_corpus(corpus, chunker)}
        assert covered == {r.doc_id for r in corpus}, chunker.name


# --- the harness -----------------------------------------------------------------------------


def test_tokenizer_splits_identifiers_so_queries_can_match_them():
    """`RC_1064` must meet a question that says "RC 1064"."""
    assert tokenize("RC_1064") == ["rc", "1064"]
    assert "delivery" in tokenize("invoice_with_proof_of_delivery")


@needs_corpus
def test_evaluation_scores_every_query(corpus, queries):
    result = evaluate_chunker(WholeDocumentChunker(), corpus, queries)
    assert len(result.outcomes) == len(queries["queries"])
    assert 0.0 <= result.recall_at(5) <= 1.0
    assert 0.0 <= result.rule_hit_at(5) <= result.recall_at(5)


@needs_corpus
def test_length_split_partitions_every_query(corpus, queries):
    threshold, groups = length_split(corpus, queries)
    assert threshold > 0
    assigned = set().union(*groups.values())
    assert assigned == {q["id"] for q in queries["queries"]}
    assert sum(len(ids) for ids in groups.values()) == len(queries["queries"])
