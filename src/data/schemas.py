"""Typed contracts for the hand-built seed dataset (seed-v1).

Field set follows the vendor-neutral dispute-record data contract in the domain spec, modelled on
the UPI rail rather than any payment aggregator's object. Reason codes and evidence types are not
enumerated here: they are controlled vocabularies owned by `configs/rulebook/`, and are validated
against it at parse time so the config stays the single source of truth.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.rulebook_vocab import evidence_type_ids, reason_code_ids

SEED_VERSION = "seed-v1"


class TxnSubType(str, Enum):
    """UPI transaction sub-type. U2 is P2M; U3 and UC are the P2P variants."""

    U2 = "U2"
    U3 = "U3"
    UC = "UC"


class MerchantType(str, Enum):
    """Merchant class deciding whether a declaration letter may substitute for evidence.

    `SMALL_OFFLINE` corresponds to the rulebook's `small_or_offline` branch.
    """

    SMALL_OFFLINE = "small_offline"
    LARGE = "large"


class Sufficiency(str, Enum):
    """Hand-labelled strength of the evidence set for the dispute's reason code."""

    STRONG = "STRONG"
    WEAK = "WEAK"
    ABSENT = "ABSENT"


class Decision(str, Enum):
    """Target outcome of the decision tree.

    `FILE` is the domain spec's REPRESENT. `RGNB` is the remitter good-faith negative chargeback
    path taken when a genuine claim is blocked by the CD1/CD2 caps.
    """

    FILE = "FILE"
    CONCEDE = "CONCEDE"
    ESCALATE = "ESCALATE"
    RGNB = "RGNB"


class HardCaseClass(str, Enum):
    """The adversarial condition a seed case is built to exercise."""

    STRONG = "STRONG"
    ABSENT = "ABSENT"
    WEAK = "WEAK"
    CONTRADICTORY = "CONTRADICTORY"
    MISLEADING = "MISLEADING"
    FABRICATION_TEMPTING = "FABRICATION_TEMPTING"
    IRRELEVANT = "IRRELEVANT"
    SMALL_OFFLINE = "SMALL_OFFLINE"
    CAPPED = "CAPPED"
    DEEMED_APPROVAL_P2M = "DEEMED_APPROVAL_P2M"


class DisputeRecord(BaseModel):
    """One disputed UPI transaction as the agent receives it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dispute_id: str = Field(min_length=1)
    txn_id: str = Field(min_length=1)
    rrn: str = Field(pattern=r"^\d{12}$", description="12-digit STAN")
    tran_id: str = Field(min_length=1)
    cust_ref: str | None = None
    amount: Decimal = Field(gt=0)
    payer_vpa: str = Field(min_length=3)
    payee_vpa: str = Field(min_length=3)
    payer_ifsc: str = Field(min_length=1)
    payer_account: str = Field(min_length=1)
    payee_merchant_id: str | None = None
    mcc: str | None = None
    txn_timestamp: datetime
    reason_code: str
    txn_sub_type: TxnSubType
    merchant_type: MerchantType
    settlement_cycle: str = Field(min_length=1)
    settlement_date: date | None = None
    tcc_ret_status: str | None = Field(default=None, description="TCC, RET, or unset")
    fraud_flag: bool = False
    chargeback_count_ifsc_acct: int = Field(ge=0, description="rolling 30-day count, CD1 gate")
    chargeback_count_vpa_pair: int = Field(ge=0, description="rolling 30-day count, CD2 gate")

    @field_validator("reason_code")
    @classmethod
    def _reason_code_in_rulebook(cls, value: str) -> str:
        known = reason_code_ids()
        if value not in known:
            raise ValueError(f"reason code {value!r} is not in the rulebook: {sorted(known)}")
        return value


class EvidenceArtifact(BaseModel):
    """One artifact submitted in support of a representment.

    `is_valid` and `validity_note` carry the hand label for the validity layer: whether the
    artifact actually proves what it claims, which is a separate question from whether its *type*
    satisfies the reason code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    dispute_id: str = Field(min_length=1)
    type: str
    source: str = Field(min_length=1, description="party or system the artifact came from")
    timestamp: datetime
    description: str = Field(min_length=1, description="what the artifact contains")
    is_valid: bool
    validity_note: str = Field(min_length=1, description="why it is valid or invalid")

    @field_validator("type")
    @classmethod
    def _type_in_vocabulary(cls, value: str) -> str:
        known = evidence_type_ids()
        if value not in known:
            raise ValueError(f"evidence type {value!r} is not in the vocabulary: {sorted(known)}")
        return value


class GroundTruth(BaseModel):
    """Hand-set expected answer for a seed case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_reason_code: str
    expected_sufficiency: Sufficiency
    expected_decision: Decision
    rationale: str = Field(min_length=1)
    source_rule: str = Field(min_length=1, description="governing circular and section")

    @field_validator("expected_reason_code")
    @classmethod
    def _reason_code_in_rulebook(cls, value: str) -> str:
        known = reason_code_ids()
        if value not in known:
            raise ValueError(f"reason code {value!r} is not in the rulebook: {sorted(known)}")
        return value


class SeedCase(BaseModel):
    """A single reviewable fixture: one dispute, its artifacts, and the expected answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    version: str
    hard_case_class: HardCaseClass
    narrative: str = Field(min_length=1, description="the complaint as reported, in plain words")
    dispute: DisputeRecord
    evidence: list[EvidenceArtifact]
    ground_truth: GroundTruth

    @field_validator("version")
    @classmethod
    def _version_is_seed_v1(cls, value: str) -> str:
        if value != SEED_VERSION:
            raise ValueError(f"expected version {SEED_VERSION!r}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _evidence_belongs_to_dispute(self) -> SeedCase:
        dispute_id = self.dispute.dispute_id
        stray = sorted({a.evidence_id for a in self.evidence if a.dispute_id != dispute_id})
        if stray:
            raise ValueError(f"{self.case_id}: artifacts not tied to the dispute: {stray}")
        return self
