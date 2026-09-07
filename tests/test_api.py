"""Smoke tests for the FastAPI serving surface."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.data.seed_loader import load_seed_cases
from src.serving.api import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def cases_by_id():
    return {case.case_id: case for case in load_seed_cases()}


def payload_for(case) -> dict:
    """Build a /decide request body carrying only what the engine may see."""
    return {
        "dispute": json.loads(case.dispute.model_dump_json()),
        "evidence": [json.loads(a.model_dump_json()) for a in case.evidence],
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "engine": "B0"}


@pytest.mark.parametrize(
    ("case_id", "expected_decision"),
    [("seed_001", "FILE"), ("seed_007", "CONCEDE"), ("seed_033", "RGNB")],
)
def test_decide_returns_the_same_outcome_as_the_engine(
    client, cases_by_id, case_id, expected_decision
):
    response = client.post("/decide", json=payload_for(cases_by_id[case_id]))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == expected_decision
    assert body["dispute_id"] == cases_by_id[case_id].dispute.dispute_id
    assert body["reason"]


def test_decide_returns_the_audit_trail(client, cases_by_id):
    response = client.post("/decide", json=payload_for(cases_by_id["seed_030"]))
    audit = response.json()["audit"]
    assert audit["engine_version"] == "B0"
    assert audit["decision"] == "FILE"
    rules = {rule["rule_id"]: rule for rule in audit["rules_fired"]}
    assert "merchant_type:declaration_substitution" in rules
    assert all(rule["source"] for rule in rules.values())
    # The section sign survives the JSON round trip rather than being mangled.
    assert any("§" in rule["source"] for rule in rules.values())


def test_decide_rejects_a_body_carrying_test_metadata(client, cases_by_id):
    """The request model forbids extra fields, so ground truth cannot be smuggled in."""
    body = payload_for(cases_by_id["seed_001"])
    body["hard_case_class"] = "STRONG"
    response = client.post("/decide", json=body)
    assert response.status_code == 422


def test_decide_rejects_an_unknown_reason_code(client, cases_by_id):
    body = payload_for(cases_by_id["seed_001"])
    body["dispute"]["reason_code"] = "RC_9999"
    response = client.post("/decide", json=body)
    assert response.status_code == 422
