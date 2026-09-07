"""L0 sufficiency engine: are the required evidence types present for this reason code?

Deterministic and rulebook-driven. This layer answers a narrow question — does the evidence set
contain an artifact of a type the reason code accepts — and delegates the separate question of
whether that artifact proves anything to the verifier. Nothing here re-derives the rulebook: the
accepted types, the declaration substitution and their source tags all come from
`configs/rulebook/reason_code_evidence.yaml`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.data.rulebook_vocab import merchant_type_rule, resolve_reason_code_entry
from src.data.schemas import DisputeRecord, EvidenceArtifact, MerchantType, Sufficiency
from src.decision.verifier import VerificationResult, Verifier

# The record's merchant_type maps onto the rulebook's declaration-rule branches.
MERCHANT_TYPE_BRANCH = {
    MerchantType.SMALL_OFFLINE: "small_or_offline",
    MerchantType.LARGE: "large",
}


class SufficiencyResult(BaseModel):
    """What the type-level check found, and the rule it applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Sufficiency
    reason_code: str
    rulebook_entry: str = Field(description="the rulebook key that governed the check")
    required_any_of: list[str]
    matched_requirement: str | None = Field(
        default=None, description="the accepted evidence type that carried the case, if any"
    )
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    valid_evidence_ids: list[str] = Field(default_factory=list)
    declaration_substituted: bool = False
    reason: str = Field(min_length=1)
    source: str = Field(min_length=1)


def accepted_evidence_types(
    reason_code: str, txn_sub_type: str, merchant_type: MerchantType
) -> tuple[list[str], str, str, bool]:
    """Resolve which evidence types satisfy this reason code for this merchant.

    Returns the accepted types, the rulebook entry key, its source tag, and whether the acquirer
    declaration letter was added as a substitute.

    A small or offline merchant cannot raise an invoice over the counter, so the acquirer's
    declaration letter substitutes for documentary fulfilment evidence. For a large merchant it
    does not substitute, and a declaration on its own leaves the case with nothing accepted.
    """
    entry_key, entry = resolve_reason_code_entry(reason_code, txn_sub_type)
    accepted = list(entry["required_evidence_any_of"])
    source = entry["source"]
    substituted = False

    branch = merchant_type_rule()[MERCHANT_TYPE_BRANCH[merchant_type]]
    substitute = branch.get("substitute_evidence")
    if substitute is not None and substitute not in accepted:
        accepted.append(substitute)
        substituted = True
        source = f"{source}; {branch['source']}"
    return accepted, entry_key, source, substituted


def assess(
    dispute: DisputeRecord,
    evidence: list[EvidenceArtifact],
    verifier: Verifier,
    reason_code: str | None = None,
) -> tuple[SufficiencyResult, list[VerificationResult]]:
    """Grade the evidence set for a dispute, returning the result and every verification made.

    `reason_code` overrides the record's own code so the classifier's output can be used once a
    real one exists; it defaults to the code on the record.
    """
    code = reason_code or dispute.reason_code
    accepted_types, entry_key, source, substituted = accepted_evidence_types(
        code, dispute.txn_sub_type.value, dispute.merchant_type
    )

    accepted = [artifact for artifact in evidence if artifact.type in accepted_types]
    verifications = [verifier.verify(dispute, artifact) for artifact in accepted]
    valid_ids = [check.evidence_id for check in verifications if check.is_valid]
    valid_accepted = [a for a in accepted if a.evidence_id in set(valid_ids)]

    if valid_accepted:
        level = Sufficiency.STRONG
        matched = valid_accepted[0].type
        reason = (
            f"{valid_accepted[0].evidence_id} is a {matched}, an accepted type for {code}, "
            f"and the verifier accepted it."
        )
    elif accepted:
        level = Sufficiency.WEAK
        matched = None
        reason = (
            f"{len(accepted)} artifact(s) of an accepted type are present for {code}, but the "
            f"verifier rejected every one of them."
        )
    else:
        level = Sufficiency.ABSENT
        matched = None
        held = sorted({artifact.type for artifact in evidence})
        reason = f"no artifact of a type {code} accepts is present" + (
            f"; the set holds {held} instead." if held else "; the evidence set is empty."
        )

    result = SufficiencyResult(
        level=level,
        reason_code=code,
        rulebook_entry=entry_key,
        required_any_of=accepted_types,
        matched_requirement=matched,
        accepted_evidence_ids=[a.evidence_id for a in accepted],
        valid_evidence_ids=valid_ids,
        declaration_substituted=substituted,
        reason=reason,
        source=source,
    )
    return result, verifications
