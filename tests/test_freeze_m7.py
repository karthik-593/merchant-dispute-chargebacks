"""Tests for the M7 freeze and the M8 evaluation set it produced.

A freeze is a claim that an experiment settled a choice, so these tests check that the record can
still back that claim: every frozen component names the arm that selected it, every rejection
carries the scope it was rejected under, and the versions named are the ones actually on disk.

The scope requirement is the one worth stating. A negative result recorded without its conditions
hardens into a universal, and the next person inherits a closed door with no sign on it.
"""

from __future__ import annotations

import re

import pytest
import yaml

from src.config import project_path
from src.evaluation.freeze_m7 import (
    FROZEN_KEYS,
    SELECTION_TABLE,
    frozen_components,
    load_m8_eval_set,
    load_versions,
)


@pytest.fixture(scope="module")
def versions():
    return load_versions()


@pytest.fixture(scope="module")
def eval_set():
    return load_m8_eval_set()


# --- the frozen block ---------------------------------------------------------------------------


def test_the_four_components_are_frozen(versions):
    frozen = frozen_components(versions)
    assert frozen["CHUNKER_VERSION"] == "structure_aware/row-grain"
    assert frozen["RETRIEVER_VERSION"] == "dense"
    assert frozen["EMBEDDING_VERSION"] == "bge-small-en-v1.5"
    assert frozen["RERANKER_VERSION"] is None, "the reranker was rejected, not chosen"
    assert versions["RERANKER_VERSION"]["status"].startswith("REJECTED")


def test_every_frozen_component_names_the_arm_that_selected_it(versions):
    """A freeze without provenance is an assumption wearing a version number."""
    for key in FROZEN_KEYS:
        entry = versions[key]
        assert entry["selected_by"].strip(), key


def test_the_frozen_embedding_is_one_the_retriever_can_actually_load(versions):
    from src.retrieval.retrievers import EMBEDDINGS

    assert frozen_components(versions)["EMBEDDING_VERSION"] in EMBEDDINGS


def test_the_named_versions_match_what_is_on_disk(versions):
    """The freeze must name the corpus and query set that produced its evidence."""
    corpus = yaml.safe_load(
        (project_path("configs") / "corpus" / "corpus_version.yaml").read_text(encoding="utf-8")
    )
    queries = yaml.safe_load(
        (project_path("data") / "corpus" / "retrieval_queries.yaml").read_text(encoding="utf-8")
    )
    assert versions["meta"]["corpus_version"] == corpus["version"]
    assert versions["meta"]["query_set_version"] == queries["meta"]["version"]


def test_the_frozen_chunker_matches_the_corrected_floor(versions):
    """structure_aware/row-grain is only meaningful with the prose-only floor from Stage 1b."""
    from src.retrieval.chunkers import _PROSE_BOUNDARIES, _ROW_BOUNDARIES, STRUCTURE_MIN_TOKENS

    assert STRUCTURE_MIN_TOKENS == 24
    assert _ROW_BOUNDARIES and _PROSE_BOUNDARIES
    assert "row-grain" in versions["CHUNKER_VERSION"]["value"]
    assert "PROSE" in versions["CHUNKER_VERSION"]["detail"]


# --- scoped negatives ----------------------------------------------------------------------------


def test_every_rejection_carries_its_scope_and_a_revisit_condition(versions):
    """The point of this test: a negative without scope becomes a universal by accident."""
    rejections = versions["negative_results"]
    assert set(rejections) == {"reranker", "parent_expansion", "hybrid_retrieval"}
    for name, entry in rejections.items():
        assert entry["verdict"].strip(), name
        assert entry["evidence"].strip(), name
        assert entry["scope"].strip(), name
        assert entry["revisit_when"].strip(), name


def test_the_reranker_rejection_is_explicitly_not_universal(versions):
    entry = versions["negative_results"]["reranker"]
    scope = " ".join(entry["scope"].split())
    assert "NOT re-tested" in scope
    assert "scoped, not universal" in scope
    assert "M8" in entry["revisit_when"]


def test_the_expansion_rejection_is_scoped_to_first_stage_retrieval(versions):
    entry = versions["negative_results"]["parent_expansion"]
    scope = " ".join(entry["scope"].split())
    assert "first-stage retrieval" in scope
    assert "M13/M14" in scope or "M13/M14" in entry["revisit_when"]


# --- the selection record ------------------------------------------------------------------------


def test_the_selection_table_prices_every_lever():
    levers = {lever for lever, _, _ in SELECTION_TABLE}
    assert levers == {"encoder", "depth", "reranker", "expansion"}
    assert len(SELECTION_TABLE) == 6, "two reranker configurations and two parent definitions"


def test_the_residual_finding_names_the_gap_rows_and_where_they_go(versions):
    residual = versions["residual_finding"]
    assert len(residual["rows"]) == 8
    assert "h10" not in residual["rows"], "h10 is reachable by e5-small; it is not a gap row"
    assert "M8" in residual["belongs_to"]
    assert residual["eval_set"] == "data/eval/m8_query_rewrite_eval.yaml"


def test_the_open_items_are_recorded_as_decisions_not_omissions(versions):
    not_done = versions["not_done"]
    assert not_done["plain_language_rule_copies"]["decision"] == "REJECTED."
    assert "provenance" in not_done["plain_language_rule_copies"]["why"]
    assert "v1.3" in not_done["recall_at_k_target_defect"]["decision"]
    assert "unaffected" in not_done["recall_at_k_target_defect"]["impact_on_this_freeze"]


# --- the M8 evaluation set -----------------------------------------------------------------------


def test_the_m8_eval_set_is_filed_and_distinct_from_the_retrieval_set(eval_set):
    meta = eval_set["meta"]
    assert meta["version"] == "m8-rewrite-eval-v1"
    assert "retrieval_queries.yaml" in meta["distinct_from"]
    assert meta["case_count"] == len(eval_set["cases"]) == 8


def test_every_m8_case_carries_the_four_fields_the_task_needs(eval_set):
    """Lay query, domain target, source row, reason code - the mapping and its evidence."""
    for case in eval_set["cases"]:
        assert case["lay_query"].strip(), case["id"]
        assert case["domain_target"].strip(), case["id"]
        assert case["source_row"].strip(), case["id"]
        assert case["source"].strip(), case["id"]
        assert case["required_domain_terms"], case["id"]
        assert "code" in case, case["id"]
        assert case["gap"].strip(), case["id"]


def test_the_m8_cases_are_exactly_the_gap_rows_from_the_freeze(versions, eval_set):
    assert [c["id"] for c in eval_set["cases"]] == versions["residual_finding"]["rows"]


def test_the_borderline_case_is_kept_out_of_the_evidence(eval_set):
    """h10 is an encoder-disagreement case. Counting it as a vocabulary gap would overstate."""
    borderline = eval_set["borderline"]
    assert [c["id"] for c in borderline] == ["h10"]
    assert "not a vocabulary gap" in " ".join(borderline[0]["note"].split())


def test_forbidden_terms_name_the_sibling_each_case_must_not_collapse_onto(eval_set):
    """A hallucinated code routes retrieval to a confidently wrong rule - worse than nothing."""
    forbidden = eval_set["forbidden_terms"]
    ids = {c["id"] for c in eval_set["cases"]}
    assert set(forbidden) <= ids
    assert set(forbidden["h32"]) == {"1134", "1135"}
    assert "ND1" in forbidden["h01"]
    for case_id, terms in forbidden.items():
        case = next(c for c in eval_set["cases"] if c["id"] == case_id)
        assert case["code"] not in terms, f"{case_id} forbids its own answer"


def test_the_guardrail_keeps_the_rewrite_on_the_query_side(eval_set):
    """The rewrite steers the query; the retriever still returns verbatim source."""
    header = (project_path("data") / "eval" / "m8_query_rewrite_eval.yaml").read_text(
        encoding="utf-8"
    )
    assert "steers the QUERY only" in header
    assert "VERBATIM SOURCE" in header
    assert "never becomes the returned text" in header
    assert re.search(r"rewrite itself must be EVALUATED", header)
