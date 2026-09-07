"""The why-trail: a structured record of how a decision was reached.

Every decision the thread makes is reconstructable from one of these. It carries the inputs it
saw, the sufficiency finding, each rule that fired with the circular it came from, the caps
evaluation, and the outcome — so a reviewer can check the reasoning without rerunning anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import Decision, DisputeRecord, EvidenceArtifact
from src.decision.caps import CapsEvaluation
from src.decision.sufficiency import SufficiencyResult
from src.decision.verifier import Contradiction, VerificationResult
from src.llm.classifier import ClassificationResult


class RuleFired(BaseModel):
    """One rule the engine applied, and the document it traces to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    source: str = Field(min_length=1, description="governing circular and section")


class AuditInputs(BaseModel):
    """The fields the engine was allowed to see. Ground truth is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dispute_id: str
    reason_code: str
    txn_sub_type: str
    merchant_type: str
    amount: Decimal
    fraud_flag: bool
    chargeback_count_ifsc_acct: int
    chargeback_count_vpa_pair: int
    acquiring_psp_is_merchant_bank: bool
    evidence_ids: list[str]
    evidence_types: list[str]


class AuditRecord(BaseModel):
    """A complete, self-contained account of one decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dispute_id: str
    decision: Decision
    decided_at: datetime
    engine_version: str
    inputs: AuditInputs
    classification: ClassificationResult
    sufficiency: SufficiencyResult
    verifications: list[VerificationResult]
    caps: CapsEvaluation
    contradictions: list[Contradiction]
    rules_fired: list[RuleFired]
    reason: str = Field(min_length=1)


def build_audit_record(
    *,
    dispute: DisputeRecord,
    evidence: list[EvidenceArtifact],
    classification: ClassificationResult,
    sufficiency: SufficiencyResult,
    verifications: list[VerificationResult],
    caps: CapsEvaluation,
    contradictions: list[Contradiction],
    rules_fired: list[RuleFired],
    decision: Decision,
    reason: str,
    engine_version: str,
) -> AuditRecord:
    """Assemble the audit record for one decision."""
    return AuditRecord(
        dispute_id=dispute.dispute_id,
        decision=decision,
        decided_at=datetime.now(UTC),
        engine_version=engine_version,
        inputs=AuditInputs(
            dispute_id=dispute.dispute_id,
            reason_code=dispute.reason_code,
            txn_sub_type=dispute.txn_sub_type.value,
            merchant_type=dispute.merchant_type.value,
            amount=dispute.amount,
            fraud_flag=dispute.fraud_flag,
            chargeback_count_ifsc_acct=dispute.chargeback_count_ifsc_acct,
            chargeback_count_vpa_pair=dispute.chargeback_count_vpa_pair,
            acquiring_psp_is_merchant_bank=dispute.acquiring_psp_is_merchant_bank,
            evidence_ids=[a.evidence_id for a in evidence],
            evidence_types=[a.type for a in evidence],
        ),
        classification=classification,
        sufficiency=sufficiency,
        verifications=verifications,
        caps=caps,
        contradictions=contradictions,
        rules_fired=rules_fired,
        reason=reason,
    )
