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
    load_queries,
    query_fingerprint,
    query_set_fingerprint,
)


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def meta(queries):
    return queries["meta"]


def test_the_set_is_frozen_and_versioned(meta, queries):
    assert meta["frozen"] is True
    assert meta["version"] == "retrieval-v1.1"
    assert meta["query_count"] == len(queries["queries"]) == 26


def test_the_declared_hash_matches_the_queries_it_describes(meta, queries):
    """If this fails, the set was edited without updating the freeze."""
    assert query_set_fingerprint(queries["queries"]) == meta["queries_sha256"]


def test_the_unamended_queries_reproduce_byte_for_byte(meta, queries):
    """The 25 queries v1.1 did not touch must hash exactly as they did under retrieval-v1.

    This is the claim that makes the amendment reviewable: one query changed, and the evidence
    that only one changed is a hash taken over the other 25 before the edit.
    """
    amended = set(meta["changelog"][-1]["queries_amended"])
    rest = [q for q in queries["queries"] if q["id"] not in amended]
    assert len(rest) == 25
    assert query_set_fingerprint(rest) == meta["unamended_sha256"]


def test_the_changelog_accounts_for_the_amendment(meta, queries):
    entry = meta["changelog"][-1]
    assert entry["version"] == meta["version"]
    assert entry["supersedes"] == "retrieval-v1"
    assert entry["queries_amended"] == ["q25"]
    assert entry["queries_unchanged"] == 25
    for field in ("change", "why", "convention", "not_amended", "metric_effect"):
        assert entry[field].strip(), field

    hashes = entry["hashes"]
    assert hashes["all_after"] == meta["queries_sha256"]
    assert hashes["all_before"] != hashes["all_after"]
    q25 = next(q for q in queries["queries"] if q["id"] == "q25")
    assert query_fingerprint(q25) == hashes["q25_after"]
    assert hashes["q25_before"] != hashes["q25_after"]


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
