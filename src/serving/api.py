"""FastAPI surface over the B0 thread.

One endpoint: post a dispute and its evidence, get back the decision and the audit trail that
justifies it. The request body carries only what the engine is allowed to see — there is no field
for ground truth or hard-case class, so the serving path cannot leak test metadata into a decision.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import Decision, DisputeRecord, EvidenceArtifact
from src.decision.audit import AuditRecord
from src.decision.engine import ENGINE_VERSION, DecisionEngine
from src.decision.sufficiency import SufficiencyResult


class DecideRequest(BaseModel):
    """A dispute and the artifacts filed with it."""

    model_config = ConfigDict(extra="forbid")

    dispute: DisputeRecord
    evidence: list[EvidenceArtifact] = Field(default_factory=list)


class DecideResponse(BaseModel):
    """The outcome, why it was reached, and the full audit record."""

    model_config = ConfigDict(extra="forbid")

    dispute_id: str
    decision: Decision
    reason: str
    sufficiency: SufficiencyResult
    audit: AuditRecord


def create_app(engine: DecisionEngine | None = None) -> FastAPI:
    """Build the application, optionally against a supplied engine."""
    decision_engine = engine or DecisionEngine()
    application = FastAPI(
        title="UPI merchant-dispute agent",
        version=ENGINE_VERSION,
        description="Rules-grounded chargeback representment decisions with an audit trail.",
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "engine": ENGINE_VERSION}

    @application.post("/decide", response_model=DecideResponse)
    def decide(request: DecideRequest) -> DecideResponse:
        """Decide one dispute and return the reasoning behind the outcome."""
        outcome = decision_engine.decide(request.dispute, request.evidence)
        return DecideResponse(
            dispute_id=request.dispute.dispute_id,
            decision=outcome.decision,
            reason=outcome.reason,
            sufficiency=outcome.sufficiency,
            audit=outcome.audit,
        )

    return application


app = create_app()
