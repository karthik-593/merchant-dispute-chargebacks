"""M0 smoke test: the sufficiency rulebook parses and every reason code is provenance-tagged."""

from __future__ import annotations

import pytest
import yaml

from src.config import project_path

RULEBOOK = project_path("rulebook") / "reason_code_evidence.yaml"


@pytest.fixture(scope="module")
def rulebook() -> dict:
    with RULEBOOK.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_rulebook_file_exists():
    assert RULEBOOK.is_file(), f"missing rulebook config: {RULEBOOK}"


def test_rulebook_parses_to_mapping(rulebook):
    assert isinstance(rulebook, dict)
    assert isinstance(rulebook.get("reason_codes"), dict)
    assert rulebook["reason_codes"], "rulebook declares no reason codes"


def test_every_reason_code_has_required_evidence_and_source(rulebook):
    for rc_key, entry in rulebook["reason_codes"].items():
        assert isinstance(entry, dict), f"{rc_key}: entry is not a mapping"

        required = entry.get("required_evidence_any_of")
        assert isinstance(required, list), f"{rc_key}: required_evidence_any_of must be a list"
        assert required, f"{rc_key}: required_evidence_any_of is empty"
        assert all(isinstance(item, str) and item.strip() for item in required), (
            f"{rc_key}: required_evidence_any_of holds a blank or non-string entry"
        )

        source = entry.get("source")
        assert isinstance(source, str) and source.strip(), f"{rc_key}: missing a `source` tag"


def test_required_evidence_uses_the_declared_vocabulary(rulebook):
    known = set(rulebook["evidence_types"])
    for rc_key, entry in rulebook["reason_codes"].items():
        unknown = set(entry["required_evidence_any_of"]) - known
        assert not unknown, f"{rc_key}: evidence types not in evidence_types: {sorted(unknown)}"
