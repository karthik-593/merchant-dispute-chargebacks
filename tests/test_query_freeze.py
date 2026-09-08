"""Tests for the frozen retrieval query set and its amendment trail.

A freeze that nobody checks is a comment. These tests make the declared hashes load-bearing: the
set cannot be edited, even by accident, without one of them failing and forcing the editor to
record what changed and why.

The hashes cover the *loaded* queries rather than the file bytes, so reformatting the YAML or
adding a comment is free while any change of meaning is caught. That is the right granularity for
something frozen as a scoring target.
"""

from __future__ import annotations

import pytest

from src.evaluation.retrieval_metrics import (
    DISCRIMINATOR,
    REGRESSION_GUARD,
    load_queries,
    normalise,
    query_fingerprint,
    query_set_fingerprint,
    scoring_fingerprint,
)
from src.retrieval.corpus_ingest import CORPUS_FILE, corpus_dir, read_corpus

CORPUS_PATH = corpus_dir() / CORPUS_FILE
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="ingested corpus not on disk (dvc pull)"
)
CARRIED_OVER = tuple(f"q{i:02d}" for i in range(1, 27))


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def meta(queries):
    return queries["meta"]


def test_the_set_is_frozen_and_versioned(meta, queries):
    assert meta["frozen"] is True
    assert meta["version"] == "retrieval-v1.2"
    assert meta["query_count"] == len(queries["queries"]) == 52
    roles = [q["role"] for q in queries["queries"]]
    assert roles.count(DISCRIMINATOR) == meta["discriminator_count"] == 50
    assert roles.count(REGRESSION_GUARD) == meta["regression_guard_count"] == 2


def test_every_query_declares_a_class_and_a_role(queries):
    valid_classes = {"identifier", "paraphrase", "near-duplicate", "evidence-by-RC"}
    for query in queries["queries"]:
        assert query["class"] in valid_classes, query["id"]
        assert query["role"] in {DISCRIMINATOR, REGRESSION_GUARD}, query["id"]


def test_the_carried_over_queries_keep_their_scoring_content(meta, queries):
    """v1.2 adds `class` and `role` to every query, which changes no query's scoring content.

    Hashed over the scoring fields only, the 26 carried-over queries must be identical to v1.1.
    The full-record hash of those same queries necessarily moved - that is the schema change, and
    conflating the two would either hide a real edit or cry wolf over a metadata one.
    """
    carried = [q for q in queries["queries"] if q["id"] in CARRIED_OVER]
    assert len(carried) == 26
    assert scoring_fingerprint(carried) == meta["carried_over_scoring_sha256"]
    assert meta["carried_over_scoring_sha256"] == meta["changelog"][-1]["hashes"]["all_before"]


def test_the_discriminator_hash_covers_exactly_the_discriminators(meta, queries):
    disc = [q for q in queries["queries"] if q["role"] == DISCRIMINATOR]
    assert len(disc) == 50
    assert query_set_fingerprint(disc) == meta["discriminators_sha256"]


def _containment(queries, text):
    """Documents that contain every anchor, per query id."""
    out = {}
    for query in queries:
        anchors = [normalise(a) for a in query["answer_anchors"]]
        out[query["id"]] = {d for d, t in text.items() if all(a in t for a in anchors)}
    return out


@needs_corpus
def test_added_targets_are_derived_by_anchor_containment_not_hand_listed(queries):
    """Re-derive every v1.2 target and assert it matches what is written down.

    A document is a target only if it contains every anchor. Recomputing it here means a corpus
    change that moves an anchor fails this test instead of silently rotting a target into one that
    can no longer answer - the q19/q21/q22 failure, caught before it is written in.

    Scoped to the queries v1.2 added, because the rule is applied FORWARD: the carried-over 26
    were hand-listed before it existed and are pinned separately below rather than rewritten,
    which would change their scoring content.
    """
    text = {r.doc_id: normalise(r.text) for r in read_corpus()}
    added = [q for q in queries["queries"] if q["id"] not in CARRIED_OVER]
    derived = _containment(added, text)
    assert len(added) == 26
    for query in added:
        assert derived[query["id"]], f"{query['id']}: no document contains all its anchors"
        assert set(query["target_doc_ids"]) == derived[query["id"]], (
            f"{query['id']}: listed targets disagree with containment"
        )


@needs_corpus
def test_the_carried_over_targets_disagree_with_containment_in_exactly_the_known_way(queries):
    """The 26 pre-date the containment rule and do not satisfy it. Pinned, not fixed.

    Two opposite defects, both real and both inherited:

    OVER-listed - the query names a document that does NOT contain all its anchors, so a
    document-level hit can be scored on a document that cannot answer. Recall@K is inflated for
    these 13; rule-hit-rate is not, because it checks the chunk text.

    UNDER-listed - a document that DOES contain all the anchors is not named, so retrieving a
    legitimate source scores as a non-target. This is exactly the q25 problem that forced v1.1,
    still present in 7 more queries.

    Neither is repaired here: rewriting these targets would change the scoring content of the
    carried-over set, which v1.2 exists to leave alone. Pinning the lists means the defect is
    tracked and a future v1.3 can act on it deliberately, and that it cannot grow unnoticed.
    """
    text = {r.doc_id: normalise(r.text) for r in read_corpus()}
    carried = [q for q in queries["queries"] if q["id"] in CARRIED_OVER]
    derived = _containment(carried, text)
    over = sorted(q["id"] for q in carried if set(q["target_doc_ids"]) - derived[q["id"]])
    under = sorted(q["id"] for q in carried if derived[q["id"]] - set(q["target_doc_ids"]))
    assert over == [
        "q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10",
        "q12", "q13", "q14",
    ]
    assert under == ["q15", "q16", "q17", "q18", "q19", "q20", "q21"]


def test_the_guards_are_the_two_tripwires_and_say_what_they_guard(queries):
    guards = [q for q in queries["queries"] if q["role"] == REGRESSION_GUARD]
    assert {g["id"] for g in guards} == {"h21", "h33"}
    for guard in guards:
        assert guard["note"].strip(), guard["id"]
    h33 = next(g for g in guards if g["id"] == "h33")
    assert "Illegible evidence" in h33["answer_anchors"]
    assert "reconciliation" in h33["note"].lower()


def test_the_reconciliation_only_queries_are_present(queries):
    """h32/h34/h35/h36 anchor on codes the OCR scan did not carry at all."""
    by_id = {q["id"]: q for q in queries["queries"]}
    for qid in ("h32", "h34", "h35", "h36"):
        assert by_id[qid]["role"] == DISCRIMINATOR
        assert by_id[qid]["target_doc_ids"] == ["oc_208a_annexure_a_reject_taxonomy_curated"]


def test_the_declared_hash_matches_the_queries_it_describes(meta, queries):
    """If this fails, the set was edited without updating the freeze."""
    assert query_set_fingerprint(queries["queries"]) == meta["queries_sha256"]


def test_the_v11_amendment_record_is_still_intact(meta, queries):
    """The 25 queries v1.1 did not touch must hash exactly as they did under retrieval-v1.

    This is the claim that makes the amendment reviewable: one query changed, and the evidence
    that only one changed is a hash taken over the other 25 before the edit.
    """
    v11 = next(c for c in meta["changelog"] if c["version"] == "retrieval-v1.1")
    rest = [
        q for q in queries["queries"]
        if q["id"] in CARRIED_OVER and q["id"] not in set(v11["queries_amended"])
    ]
    assert len(rest) == 25
    assert scoring_fingerprint(rest) == meta["unamended_sha256"]


def test_the_changelog_accounts_for_the_amendment(meta, queries):
    entry = meta["changelog"][-1]
    assert entry["version"] == meta["version"] == "retrieval-v1.2"
    assert entry["supersedes"] == "retrieval-v1.1"
    assert entry["queries_added"] == 26
    assert entry["discriminators_added"] == 24
    assert entry["regression_guards_added"] == 2
    for field in ("change", "why", "weighting", "targets_computed", "reconciliation_only",
                  "regression_guards", "not_added", "metric_effect"):
        assert entry[field].strip(), field

    hashes = entry["hashes"]
    assert hashes["all_after"] == meta["queries_sha256"]
    assert hashes["all_before"] != hashes["all_after"]
    assert hashes["discriminators_after"] == meta["discriminators_sha256"]


def test_every_added_query_has_a_recorded_hash(meta, queries):
    recorded = meta["changelog"][-1]["added_query_hashes"]
    added = [q for q in queries["queries"] if q["id"] not in CARRIED_OVER]
    assert len(added) == len(recorded) == 26
    for query in added:
        assert query_fingerprint(query) == recorded[query["id"]], query["id"]


def test_the_amendment_only_added_a_target(queries):
    """Question, topic, section and anchors must be untouched; only targets grew."""
    q25 = next(q for q in queries["queries"] if q["id"] == "q25")
    assert q25["answer_anchors"] == ["generic statements"]
    assert q25["target_section"] == "OC 208A Annexure A, reason code 1142"
    assert q25["topic"] == "reject_reasons"
    assert "invalid generic" in " ".join(q25["question"].split())
    assert q25["target_doc_ids"] == [
        "oc_208a_annexure_a_reject_taxonomy_curated",
        "upi_oc_no_208_a_fy_2025_26_addendum_to_oc_208_implementation_of_nrp_prd_process_",
    ]


def test_query_ids_are_unique_and_fingerprints_distinguish_them(queries):
    ids = [q["id"] for q in queries["queries"]]
    assert len(ids) == len(set(ids))
    prints = {query_fingerprint(q) for q in queries["queries"]}
    assert len(prints) == len(ids), "two queries hash alike; the fingerprint is not discriminating"


def test_a_fingerprint_ignores_formatting_but_not_meaning():
    """The property the freeze depends on, asserted rather than assumed."""
    base = {"id": "x", "question": "a  b", "target_doc_ids": ["d"], "answer_anchors": ["k"]}
    reformatted = {"answer_anchors": ["k"], "target_doc_ids": ["d"], "question": "a  b", "id": "x"}
    changed = {**base, "answer_anchors": ["k2"]}
    assert query_fingerprint(base) == query_fingerprint(reformatted), "key order must not matter"
    assert query_fingerprint(base) != query_fingerprint(changed), "an anchor change must matter"


def test_the_queries_that_were_deliberately_not_amended_stayed_that_way(queries):
    """q19, q21 and q22 name a superseded circular but must not gain its curated record.

    Those records cover a different section and do not carry these queries' anchors, so listing
    them would let Recall@K credit a document that cannot answer the question.
    """
    by_id = {q["id"]: q for q in queries["queries"]}
    assert "oc_208a_annexure_a_reject_taxonomy_curated" not in by_id["q19"]["target_doc_ids"]
    for qid in ("q21", "q22"):
        assert "oc_208_sec_c_evidence_map_curated" not in by_id[qid]["target_doc_ids"]
