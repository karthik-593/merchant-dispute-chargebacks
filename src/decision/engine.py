"""L5 decision engine: the deterministic policy that turns findings into an outcome.

This is baseline B0 — the rules-only thread. It reads the dispute, its evidence and the rulebook,
and nothing else: `hard_case_class` and every ground-truth field are test metadata and are not
available to it, which is why `decide` takes a record and artifacts rather than a seed case.

Policy order, which matters:

1. the caps gate short-circuits everything — a blocked chargeback cannot be filed on any evidence;
2. a critical contradiction escalates to a human rather than being adjudicated here;
3. otherwise the sufficiency finding decides: STRONG files, WEAK and ABSENT concede.

The expected-value policy the project is aiming at needs a calibrated P(win) and the fee schedule,
neither of which exists yet, so this deterministic mapping stands in for it as the baseline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import Decision, DisputeRecord, EvidenceArtifact, Sufficiency
from src.decision.audit import AuditRecord, RuleFired, build_audit_record
from src.decision.caps import CapsEvaluation, evaluate_caps
from src.decision.sufficiency import SufficiencyResult, assess
from src.decision.verifier import Contradiction, StubVerifier, VerificationResult, Verifier
from src.llm.classifier import ClassificationResult, Classifier, PassthroughClassifier

ENGINE_VERSION = "B0"

# Outcome for each sufficiency level once the caps and contradiction gates have passed.
SUFFICIENCY_POLICY = {
    Sufficiency.STRONG: Decision.FILE,
    Sufficiency.WEAK: Decision.CONCEDE,
    Sufficiency.ABSENT: Decision.CONCEDE,
}
POLICY_SOURCE = "OC 208 §C (evidence map); OC 208 §C(b) (auto-loss conditions)"
CLASSIFICATION_SOURCE = "OC 208 §C (reason-code evidence map)"
DEEMED_APPROVAL_SOURCE = "OC 39A (deemed approval, acquiring PSP = merchant bank)"


class DecisionOutcome(BaseModel):
    """The decision plus everything that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Decision
    reason: str = Field(min_length=1)
    classification: ClassificationResult
    sufficiency: SufficiencyResult
    verifications: list[VerificationResult]
    caps: CapsEvaluation
    contradictions: list[Contradiction]
    rules_fired: list[RuleFired]
    audit: AuditRecord


class DecisionEngine:
    """Runs the thread: classify, grade the evidence, gate, decide, and record why."""

    def __init__(
        self, classifier: Classifier | None = None, verifier: Verifier | None = None
    ) -> None:
        """Wire the thread, defaulting to the passthrough classifier and the stub verifier."""
        self.classifier = classifier or PassthroughClassifier()
        self.verifier = verifier or StubVerifier()

    def decide(self, dispute: DisputeRecord, evidence: list[EvidenceArtifact]) -> DecisionOutcome:
        """Decide one dispute from its record, its artifacts and the rulebook."""
        classification = self.classifier.classify(dispute)
        sufficiency, verifications = assess(
            dispute, evidence, self.verifier, classification.reason_code
        )
        caps = evaluate_caps(dispute)
        contradictions = self.verifier.find_contradictions(dispute, evidence)

        rules = [
            RuleFired(
                rule_id="classification",
                outcome=classification.reason_code,
                detail=f"reason code determined by {classification.method}",
                source=CLASSIFICATION_SOURCE,
            ),
            RuleFired(
                rule_id=f"sufficiency:{sufficiency.rulebook_entry}",
                outcome=sufficiency.level.value,
                detail=sufficiency.reason,
                source=sufficiency.source,
            ),
            RuleFired(
                rule_id="caps_gate",
                outcome="TRIGGERED" if caps.triggered else "NOT_TRIGGERED",
                detail=caps.reason,
                source=caps.source,
            ),
            self._deemed_approval_rule(dispute),
        ]

        if sufficiency.declaration_substituted:
            rules.append(
                RuleFired(
                    rule_id="merchant_type:declaration_substitution",
                    outcome="SUBSTITUTE_ALLOWED",
                    detail=(
                        "small or offline merchant, so an acquirer declaration letter is an "
                        "accepted substitute for documentary fulfilment evidence"
                    ),
                    source=sufficiency.source,
                )
            )

        decision, reason, final_rule = self._apply_policy(sufficiency, caps, contradictions)
        rules.append(final_rule)

        audit = build_audit_record(
            dispute=dispute,
            evidence=evidence,
            classification=classification,
            sufficiency=sufficiency,
            verifications=verifications,
            caps=caps,
            contradictions=contradictions,
            rules_fired=rules,
            decision=decision,
            reason=reason,
            engine_version=ENGINE_VERSION,
        )
        return DecisionOutcome(
            decision=decision,
            reason=reason,
            classification=classification,
            sufficiency=sufficiency,
            verifications=verifications,
            caps=caps,
            contradictions=contradictions,
            rules_fired=rules,
            audit=audit,
        )

    @staticmethod
    def _deemed_approval_rule(dispute: DisputeRecord) -> RuleFired:
        """Record the deemed-approval signal. Wired for later; it changes no outcome in B0."""
        applies = dispute.acquiring_psp_is_merchant_bank
        return RuleFired(
            rule_id="deemed_approval_p2m",
            outcome="PRESUMPTION_APPLIES" if applies else "NOT_APPLICABLE",
            detail=(
                "acquiring PSP and merchant bank are the same institution, so the transaction was "
                "returned SUCCESS and delivery is presumed; the presumption carries no weight in "
                "the B0 policy and does not change this outcome"
                if applies
                else "acquiring PSP and merchant bank are not known to be the same institution"
            ),
            source=DEEMED_APPROVAL_SOURCE,
        )

    @staticmethod
    def _apply_policy(
        sufficiency: SufficiencyResult,
        caps: CapsEvaluation,
        contradictions: list[Contradiction],
    ) -> tuple[Decision, str, RuleFired]:
        """Apply the outcome policy in order: caps, then contradiction, then sufficiency."""
        if caps.triggered:
            return (
                Decision.RGNB,
                f"caps gate fired before the evidence question: {caps.reason}",
                RuleFired(
                    rule_id="policy:caps_short_circuit",
                    outcome=Decision.RGNB.value,
                    detail=(
                        "a chargeback blocked by a volume cap takes the remitter good-faith "
                        "negative chargeback route regardless of the evidence"
                    ),
                    source=caps.source,
                ),
            )

        critical = [c for c in contradictions if c.critical]
        if critical:
            ids = sorted({eid for c in critical for eid in c.evidence_ids})
            return (
                Decision.ESCALATE,
                f"critical contradiction across {ids}; a human must resolve it before filing",
                RuleFired(
                    rule_id="policy:critical_contradiction",
                    outcome=Decision.ESCALATE.value,
                    detail="; ".join(c.detail for c in critical),
                    source=POLICY_SOURCE,
                ),
            )

        decision = SUFFICIENCY_POLICY[sufficiency.level]
        return (
            decision,
            f"{sufficiency.level.value} evidence: {sufficiency.reason}",
            RuleFired(
                rule_id="policy:sufficiency",
                outcome=decision.value,
                detail=f"sufficiency {sufficiency.level.value} maps to {decision.value}",
                source=POLICY_SOURCE,
            ),
        )
