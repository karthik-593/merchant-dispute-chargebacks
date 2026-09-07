"""L0 caps gate: has this customer or VPA pair already used up its chargeback allowance?

Runs before the evidence question. A claim blocked by a cap cannot be raised as a chargeback
however strong its evidence, so the gate short-circuits the rest of the policy. Limits, counter
bindings, the fraud exemption and the breach path all come from `configs/rulebook/caps.yaml`;
none of them are written into this module.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.data.rulebook_vocab import load_caps
from src.data.schemas import DisputeRecord


class CapBreach(BaseModel):
    """One cap whose limit has been reached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cap_id: str
    decline_code: str
    counter_field: str
    counter_value: int
    limit: int
    window_days: int
    source: str = Field(min_length=1)


class CapsEvaluation(BaseModel):
    """Whether the caps gate fires for this dispute, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    triggered: bool
    fraud_exempt: bool
    breaches: list[CapBreach] = Field(default_factory=list)
    breach_path: str | None = Field(default=None, description="route taken when a cap fires")
    reason: str = Field(min_length=1)
    source: str = Field(min_length=1)


def evaluate_caps(dispute: DisputeRecord) -> CapsEvaluation:
    """Decide whether a chargeback on this dispute would be declined under a volume cap.

    A counter holds the chargebacks already raised in the rolling window, so reaching the limit
    means the next one is declined. Fraud transactions sit outside the cap regime entirely, so a
    fraud-flagged dispute is evaluated for the record but never gated.
    """
    config = load_caps()
    exemption = config["exemptions"]["fraud"]
    breach_path = config["on_breach"]

    breaches = [
        CapBreach(
            cap_id=cap_id,
            decline_code=cap["decline_code"],
            counter_field=cap["counter_field"],
            counter_value=getattr(dispute, cap["counter_field"]),
            limit=cap["limit"],
            window_days=cap["window_days"],
            source=cap["source"],
        )
        for cap_id, cap in config["caps"].items()
        if getattr(dispute, cap["counter_field"]) >= cap["limit"]
    ]

    fraud_exempt = bool(getattr(dispute, exemption["field"]))
    triggered = bool(breaches) and not fraud_exempt

    if fraud_exempt and breaches:
        reason = (
            f"{', '.join(b.decline_code for b in breaches)} would fire, but the transaction is "
            f"flagged as fraud and fraud transactions are exempt from the caps."
        )
        source = exemption["source"]
    elif triggered:
        detail = ", ".join(
            f"{b.decline_code} ({b.counter_field}={b.counter_value} >= {b.limit} "
            f"in {b.window_days} days)"
            for b in breaches
        )
        reason = f"cap reached: {detail}; the next chargeback is declined."
        source = "; ".join(dict.fromkeys([b.source for b in breaches] + [breach_path["source"]]))
    else:
        below = ", ".join(
            f"{cap['counter_field']}={getattr(dispute, cap['counter_field'])} < {cap['limit']}"
            for cap in config["caps"].values()
        )
        reason = f"no cap reached ({below}); the chargeback is still allowed."
        source = config["meta"]["primary_source"]

    return CapsEvaluation(
        triggered=triggered,
        fraud_exempt=fraud_exempt,
        breaches=breaches,
        breach_path=breach_path["path"] if triggered else None,
        reason=reason,
        source=source,
    )
