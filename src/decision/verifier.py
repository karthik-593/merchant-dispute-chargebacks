"""Evidence-validity verification (L4) and the stub standing in for it.

Sufficiency asks whether the required evidence *types* are present. Validity asks whether an
artifact actually proves what it claims — a refund that completed rather than one merely
initiated, an invoice whose amount matches the disputed transaction. That judgement needs to read
document content, so the real verifier is later work.

`StubVerifier` reads back the hand label already on the artifact. It is a fixture reader, not a
verifier, and it is deliberately incapable of detecting contradictions.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import DisputeRecord, EvidenceArtifact


class VerificationResult(BaseModel):
    """Whether one artifact proves what a representment would need it to prove."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    is_valid: bool
    reason: str = Field(min_length=1)
    method: str = Field(min_length=1)


class Contradiction(BaseModel):
    """Two or more artifacts that cannot both be true as a defence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ids: list[str] = Field(min_length=2)
    detail: str = Field(min_length=1)
    critical: bool = True


class Verifier(Protocol):
    """Judges artifact validity and spots conflicts across an evidence set."""

    def verify(self, dispute: DisputeRecord, artifact: EvidenceArtifact) -> VerificationResult:
        """Judge whether one artifact proves what it claims."""
        ...

    def find_contradictions(
        self, dispute: DisputeRecord, evidence: list[EvidenceArtifact]
    ) -> list[Contradiction]:
        """Find artifacts that cannot both be true as a defence."""
        ...


class StubVerifier:
    """Reads back the hand-set `is_valid` label instead of inspecting the artifact.

    It never reads `hard_case_class` or any ground-truth field, and it never reports a
    contradiction: cross-artifact reasoning is exactly the capability the stub lacks. The
    ESCALATE branch downstream stays wired and unreachable rather than being faked here.
    """

    method = "stub-passthrough"

    def verify(self, dispute: DisputeRecord, artifact: EvidenceArtifact) -> VerificationResult:
        """Return the artifact's own validity label."""
        return VerificationResult(
            evidence_id=artifact.evidence_id,
            is_valid=artifact.is_valid,
            reason=artifact.validity_note,
            method=self.method,
        )

    def find_contradictions(
        self, dispute: DisputeRecord, evidence: list[EvidenceArtifact]
    ) -> list[Contradiction]:
        """Always empty: this stub cannot compare artifacts against each other."""
        return []
