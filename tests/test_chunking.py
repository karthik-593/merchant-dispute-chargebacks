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
    """Shape only.

    The version, its hashes and the amendment trail live in test_query_freeze.py, so a re-freeze
    does not have to be edited into two places.
    """
    assert queries["meta"]["version"].startswith("retrieval-v")
    assert queries["meta"]["frozen"] is True
    assert queries["meta"]["query_count"] == len(queries["queries"])
    assert 20 <= len(queries["queries"]) <= 80


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


# --- the merge floor is for prose, not for table rows ------------------------------------------
#
# STRUCTURE_MIN_TOKENS exists to stop running prose fragmenting into slivers. Applied to a table
# it did the opposite of its job: reject rows fell under the floor once their uniform boilerplate
# was removed, so adjacent near-duplicate codes fused into one chunk. A chunk holding 1156 and
# 1157 is not a citation unit, and a query about either cannot be scored against it.

CURATED_TABLES = {
    "oc_208a_annexure_a_reject_taxonomy_curated": r"Reason code (\d{4})",
    "oc_184b_rgnb_response_table_curated": r"Code (N[BARD]\d)",
    "oc_208_sec_c_evidence_map_curated": r"Reason code (RC_\S+)",
}


@needs_corpus
def test_a_table_row_is_never_fused_with_its_neighbour(corpus):
    """One row, one chunk - at any length. This is what structure_aware claims to do."""
    import re

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    for doc_id, pattern in CURATED_TABLES.items():
        for chunk in (c for c in chunks if c.doc_id == doc_id):
            codes = re.findall(pattern, chunk.text)
            assert len(codes) <= 1, (
                f"{chunk.chunk_id} fuses {codes}; a chunk answering two rows cites neither"
            )


@needs_corpus
def test_every_table_row_gets_its_own_chunk(corpus):
    """Counted from the other side: no row may go missing into a neighbour's chunk."""
    import re

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    for doc_id, pattern in CURATED_TABLES.items():
        record = next(r for r in corpus if r.doc_id == doc_id)
        in_source = len(re.findall(pattern, record.text))
        carried = sum(1 for c in chunks if c.doc_id == doc_id and re.findall(pattern, c.text))
        assert carried == in_source, f"{doc_id}: {in_source} rows but {carried} row chunks"


@needs_corpus
def test_short_rows_survive_the_floor(corpus):
    """The rows this fix exists for are well under STRUCTURE_MIN_TOKENS and must stand alone."""
    from src.retrieval.chunkers import STRUCTURE_MIN_TOKENS

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    reject = [
        c for c in chunks if c.doc_id == "oc_208a_annexure_a_reject_taxonomy_curated"
    ]
    short = [c for c in reject if c.n_tokens < STRUCTURE_MIN_TOKENS]
    assert short, "the reject rows are short; if none are, the record changed shape"


@needs_corpus
def test_the_floor_still_applies_to_prose(corpus):
    """The other half: prose must not start fragmenting into slivers.

    Row boundaries occur only in the curated tables (plus one harmless table header in OC 206A),
    so prose chunking is structurally untouched by the row rule - this pins that.
    """
    from src.retrieval.chunkers import STRUCTURE_MIN_TOKENS

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    prose = [c for c in chunks if c.doc_id not in CURATED_TABLES]
    tiny = [c for c in prose if c.n_tokens < STRUCTURE_MIN_TOKENS]
    assert len(tiny) <= 2, (
        f"prose over-fragmented: {[(c.doc_id, c.n_tokens) for c in tiny]}"
    )


def test_row_and_prose_boundaries_are_kept_apart():
    from src.retrieval.chunkers import (
        _PROSE_BOUNDARIES,
        _ROW_BOUNDARIES,
        _STRUCTURE_BOUNDARIES,
    )

    assert len(_ROW_BOUNDARIES) == 2, "reason-code entries and RGNB rows"
    assert not set(_ROW_BOUNDARIES) & set(_PROSE_BOUNDARIES)
    assert set(_STRUCTURE_BOUNDARIES) == set(_ROW_BOUNDARIES) | set(_PROSE_BOUNDARIES)
    chunker = StructureAwareChunker()
    assert chunker._is_row_boundary("Reason code 1142 - Invalid evidence")
    assert chunker._is_row_boundary("Flag BGND / Code ND2 - P2M U2")
    assert not chunker._is_row_boundary("1. Total chargebacks per customer")
    assert chunker._is_boundary("1. Total chargebacks per customer")


# --- metadata must never be rendered into row text ---------------------------------------------
#
# The regression guard for one defect that has now appeared three times:
#
#   Stage 1   the NVB/NVR verdict band was rendered on every reject row - identical across 26
#             codes, so it diluted the only thing separating them.
#   Stage 1b  stripping that boilerplate dropped rows under the merge floor and fused them.
#   Stage 3   the ND2 transcription note, added while fixing the first two, merged into ND2's
#             chunk. h01's diagnosis was measuring an 87-token provenance note, not the row.
#
# Every time, the metadata was CORRECT. That is the point: correctness at one layer does not make
# text safe to retrieve at another. A row's retrievable text is the rule and nothing else;
# provenance rides on the chunk as doc_id, source_section, source_path and page.

METADATA_IN_ROW_TEXT = (
    "Transcription rule",
    "Transcription note",
    "Source:",
    "Verdict:",
    "reconciliation",
    "normalised",
    "OCR-corrected",
)


@needs_corpus
def test_no_row_chunk_carries_metadata_boilerplate(corpus):
    """A table row's text is the code and its rule. Anything else is a retrieval hazard."""
    from src.retrieval.chunkers import _ROW_BOUNDARIES

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    offenders = []
    for chunk in chunks:
        head = chunk.text.splitlines()[0]
        if not any(pattern.match(head) for pattern in _ROW_BOUNDARIES):
            continue
        for marker in METADATA_IN_ROW_TEXT:
            if marker.lower() in chunk.text.lower():
                offenders.append((chunk.doc_id, head[:44], marker, chunk.n_tokens))
    assert not offenders, (
        "metadata rendered into row text - the defect that voided h01's diagnosis:\n  "
        + "\n  ".join(f"{d}: {h!r} contains {m!r} ({n} tok)" for d, h, m, n in offenders)
    )


@needs_corpus
def test_trailing_record_text_does_not_land_in_the_last_row(corpus):
    """Text after the final row merges into it. All three curated records were losing this way."""
    import re

    chunks = chunk_corpus(corpus, StructureAwareChunker())
    for doc_id, pattern in CURATED_TABLES.items():
        rows = [c for c in chunks if c.doc_id == doc_id and re.findall(pattern, c.text)]
        assert rows, doc_id
        largest = max(rows, key=lambda c: c.n_tokens)
        median = sorted(c.n_tokens for c in rows)[len(rows) // 2]
        assert largest.n_tokens <= median * 4, (
            f"{doc_id}: one row chunk is {largest.n_tokens} tokens against a median of {median} - "
            f"trailing record text has almost certainly merged into it"
        )
