"""Read-only accessors for the controlled vocabularies declared in the rulebook config.

Vocabulary lookup only: this module reports what the rulebook *declares* so the seed fixtures can
be validated against it. Interpreting those rules — the L0 sufficiency engine — is later work.
"""

from __future__ import annotations

import functools
from typing import Any

import yaml

from src.config import project_path

RULEBOOK_FILE = "reason_code_evidence.yaml"
CAPS_FILE = "caps.yaml"
REJECT_TAXONOMY_FILE = "reject_taxonomy.yaml"


@functools.cache
def load_rulebook() -> dict[str, Any]:
    """Read and cache the reason-code/evidence rulebook config."""
    path = project_path("rulebook") / RULEBOOK_FILE
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@functools.cache
def evidence_type_ids() -> frozenset[str]:
    """Controlled evidence-type vocabulary; an artifact type outside this set is invalid."""
    return frozenset(load_rulebook()["evidence_types"])


@functools.cache
def reason_code_entries() -> dict[str, dict[str, Any]]:
    """Rulebook entries by declared key (RC_108_U2 and RC_108_U3 are separate entries)."""
    return dict(load_rulebook()["reason_codes"])


@functools.cache
def reason_code_ids() -> frozenset[str]:
    """Wire reason codes: an entry's explicit `code` where present, else its key."""
    return frozenset(entry.get("code", key) for key, entry in reason_code_entries().items())


@functools.cache
def txn_sub_type_ids() -> frozenset[str]:
    """Transaction sub-types the rulebook declares (U2 / U3 / UC)."""
    return frozenset(load_rulebook()["meta"]["txn_sub_types"])


@functools.cache
def merchant_type_branch_ids() -> frozenset[str]:
    """Merchant-type branch keys the rulebook declares for the declaration-letter rule."""
    return frozenset(load_rulebook()["merchant_type_rule"])


@functools.cache
def merchant_type_rule() -> dict[str, Any]:
    """The declaration-letter substitution rule, by merchant-type branch."""
    return dict(load_rulebook()["merchant_type_rule"])


def resolve_reason_code_entry(reason_code: str, txn_sub_type: str) -> tuple[str, dict[str, Any]]:
    """Find the one rulebook entry governing a (reason code, transaction sub-type) pair.

    Pure lookup: reason codes 108 and 121 have separate P2M and P2P entries because they take
    different evidence, so the sub-type is part of the key.

    Raises:
        KeyError: if the pair matches no entry, or more than one.
    """
    matches = {
        key: entry
        for key, entry in reason_code_entries().items()
        if entry.get("code", key) == reason_code and txn_sub_type in entry.get("txn_sub_type", [])
    }
    if len(matches) != 1:
        raise KeyError(
            f"{reason_code}/{txn_sub_type} resolved to {sorted(matches)}; expected exactly one entry"
        )
    return next(iter(matches.items()))


@functools.cache
def load_caps() -> dict[str, Any]:
    """Read and cache the chargeback caps config."""
    path = project_path("rulebook") / CAPS_FILE
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@functools.cache
def load_reject_taxonomy() -> dict[str, Any]:
    """Read and cache the NRP verdict reason codes transcribed from OC 208A Annexure A."""
    path = project_path("rulebook") / REJECT_TAXONOMY_FILE
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@functools.cache
def nrp_verdict_reason_codes() -> dict[str, dict[str, Any]]:
    """NRP verdict reason codes by code, as declared by the reject taxonomy."""
    return {entry["code"]: entry for entry in load_reject_taxonomy()["nrp_verdict_reason_codes"]}
