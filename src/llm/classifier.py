"""Narrative to reason-code classification (L2) and the stub standing in for it.

The real classifier reads the dispute narrative and infers the reason code. That is later work;
until then `PassthroughClassifier` takes the reason code the rail already supplied, so the rest of
the thread can be built and measured against a known-good input.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import DisputeRecord


class ClassificationResult(BaseModel):
    """The reason code the thread will reason with, and where it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method: str = Field(min_length=1, description="how the reason code was arrived at")


class Classifier(Protocol):
    """Turns a dispute into the reason code the sufficiency check should use."""

    def classify(self, dispute: DisputeRecord) -> ClassificationResult:
        """Return the reason code governing this dispute."""
        ...


class PassthroughClassifier:
    """Returns the reason code already on the record, without looking at the narrative.

    This is the honest baseline: it asserts no inference and claims no confidence, so any
    classification error in the thread is attributable to the record rather than hidden here.
    """

    method = "passthrough"

    def classify(self, dispute: DisputeRecord) -> ClassificationResult:
        """Hand back the record's own reason code."""
        return ClassificationResult(
            reason_code=dispute.reason_code,
            confidence=None,
            method=self.method,
        )
