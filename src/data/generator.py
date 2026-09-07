"""Synthetic dispute generator (dataset-v1.0).

Sampling choices live in `configs/generator.yaml`; domain rules do not. Reason codes, accepted
evidence types, the declaration substitution and the cap limits are read from
`configs/rulebook/` at generation time.

Labelling works by running the decision engine itself rather than by re-implementing its rules.
The generator knows what it planted, so it wires the engine with an oracle verifier that reports
those planted contradictions; the B0 baseline wires the same engine with the stub verifier, which
cannot. The two therefore agree by construction everywhere except on contradiction detection,
which is exactly the capability gap being measured — there is no second copy of the rules to
drift.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import yaml

from src.config import project_path
from src.data.rulebook_vocab import (
    evidence_type_ids,
    load_caps,
    merchant_type_rule,
    reason_code_entries,
)
from src.data.schemas import (
    Decision,
    DisputeRecord,
    EvidenceArtifact,
    GroundTruth,
    HardCaseClass,
    MerchantType,
    SYNTHETIC_VERSION,
    RealizedOutcome,
    SeedCase,
    Sufficiency,
    TxnSubType,
)
from src.decision.engine import DecisionEngine
from src.decision.verifier import Contradiction, StubVerifier

CONFIG_FILE = "generator.yaml"
DATASET_VERSION = SYNTHETIC_VERSION

# Classes whose shape only makes sense for a person-to-merchant transaction.
P2M_ONLY_CLASSES = {HardCaseClass.SMALL_OFFLINE, HardCaseClass.DEEMED_APPROVAL_P2M}


@functools.cache
def load_generator_config() -> dict[str, Any]:
    """Read, cache and check the generator config."""
    path = project_path("configs") / CONFIG_FILE
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_generator_config(config)
    return config


def validate_generator_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject a config that cannot produce valid data.

    Called both when loading from disk and when a config is injected, so an in-memory config
    cannot slip past the vocabulary guard and coin an evidence type the rulebook has never heard
    of.
    """
    for key in ("hard_case_class_shares", "txn_sub_type_shares"):
        total = sum(config[key].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{key} must sum to 1, got {total}")
    split_total = sum(config["split"][k] for k in ("train", "val", "test"))
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {split_total}")

    vocabulary = evidence_type_ids()
    unknown = set(config["evidence"]["refund_types"]) - vocabulary
    if unknown:
        raise ValueError(f"refund_types names evidence outside the vocabulary: {sorted(unknown)}")
    unknown_classes = set(config["hard_case_class_shares"]) - {c.value for c in HardCaseClass}
    if unknown_classes:
        raise ValueError(f"unknown hard case classes: {sorted(unknown_classes)}")
    return config


class OracleVerifier(StubVerifier):
    """A verifier that knows what the generator planted.

    Validity is read back exactly as the stub does. The difference is contradictions: this one
    reports the conflicts the generator built into a case, so the engine can reach ESCALATE. It
    exists only to label generated data and is never used to serve a decision.
    """

    method = "oracle"

    def __init__(self, contradictions: dict[str, list[Contradiction]] | None = None) -> None:
        """Hold the planted contradictions, keyed by dispute id."""
        self._planted = contradictions or {}

    def find_contradictions(
        self, dispute: DisputeRecord, evidence: list[EvidenceArtifact]
    ) -> list[Contradiction]:
        """Report the conflicts planted in this dispute, if any."""
        return list(self._planted.get(dispute.dispute_id, []))


@dataclass(frozen=True)
class GeneratedCase:
    """One synthetic case plus the planted facts a checker needs."""

    case: SeedCase
    contradictions: list[Contradiction]


def _rulebook_codes_for(sub_type: str) -> list[tuple[str, dict[str, Any]]]:
    """Rulebook entries valid for a transaction sub-type, as (wire code, entry) pairs."""
    return [
        (entry.get("code", key), entry)
        for key, entry in reason_code_entries().items()
        if sub_type in entry.get("txn_sub_type", [])
    ]


def _non_accepted_types(accepted: list[str]) -> list[str]:
    """Vocabulary entries this reason code does not accept, for building irrelevant evidence."""
    return sorted(evidence_type_ids() - set(accepted))


class SyntheticGenerator:
    """Builds synthetic cases and labels them with the engine."""

    def __init__(self, config: dict[str, Any] | None = None, seed: int | None = None) -> None:
        """Seed the RNG and cache the rulebook facts the builder needs."""
        self.config = validate_generator_config(config) if config else load_generator_config()
        self.seed = self.config["seed"] if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.caps = load_caps()["caps"]
        self.declaration = merchant_type_rule()["small_or_offline"]["substitute_evidence"]
        self.refund_types = set(self.config["evidence"]["refund_types"])

    # -- sampling helpers ----------------------------------------------------------------

    def _choice(self, options: list[str], weights: list[float] | None = None) -> str:
        if weights is None:
            return str(self.rng.choice(options))
        total = sum(weights)
        return str(self.rng.choice(options, p=[w / total for w in weights]))

    def _sample_class(self) -> HardCaseClass:
        shares = self.config["hard_case_class_shares"]
        return HardCaseClass(self._choice(list(shares), list(shares.values())))

    def _feasible_sub_types(self, hard_class: HardCaseClass) -> list[TxnSubType]:
        """Sub-types that can host this class, worked out from the rulebook rather than assumed.

        A refund that never completed needs a reason code accepting refund evidence, and only
        some sub-types have one; the P2P codes accept a beneficiary credit screenshot and nothing
        else, so they cannot host a MISLEADING case.
        """
        if hard_class in P2M_ONLY_CLASSES:
            return [TxnSubType.U2]
        if hard_class is not HardCaseClass.MISLEADING:
            return list(TxnSubType)
        return [
            sub_type
            for sub_type in TxnSubType
            if any(
                self.refund_types & set(entry["required_evidence_any_of"])
                for _, entry in _rulebook_codes_for(sub_type.value)
            )
        ]

    def _sample_sub_type(self, hard_class: HardCaseClass) -> TxnSubType:
        feasible = self._feasible_sub_types(hard_class)
        if not feasible:
            raise ValueError(f"no transaction sub-type can host {hard_class.value}")
        if len(feasible) == 1:
            return feasible[0]
        shares = self.config["txn_sub_type_shares"]
        names = [s.value for s in feasible]
        return TxnSubType(self._choice(names, [shares[n] for n in names]))

    def _sample_reason_code(
        self, sub_type: TxnSubType, hard_class: HardCaseClass
    ) -> tuple[str, list[str]]:
        """Pick a reason code valid for the sub-type and able to host this class."""
        weights = self.config["reason_code_weights"]
        candidates = _rulebook_codes_for(sub_type.value)
        if hard_class is HardCaseClass.MISLEADING:
            # A refund that never completed needs a reason code that accepts refund evidence.
            candidates = [
                (code, entry)
                for code, entry in candidates
                if self.refund_types & set(entry["required_evidence_any_of"])
            ]
        codes = [code for code, _ in candidates]
        entries = {code: entry for code, entry in candidates}
        chosen = self._choice(codes, [weights.get(code, 0.01) for code in codes])
        return chosen, list(entries[chosen]["required_evidence_any_of"])

    def _sample_timestamp(self) -> datetime:
        window = self.config["window"]
        start = date.fromisoformat(window["start"])
        end = date.fromisoformat(window["end"])
        span = (end - start).days
        day = start + timedelta(days=int(self.rng.integers(0, span + 1)))
        moment = time(
            hour=int(self.rng.integers(0, 24)),
            minute=int(self.rng.integers(0, 60)),
            second=int(self.rng.integers(0, 60)),
        )
        return datetime.combine(day, moment)

    def _sample_amount(self) -> Decimal:
        cfg = self.config["amount"]
        raw = float(self.rng.lognormal(cfg["log_mean"], cfg["log_sigma"]))
        raw = min(max(raw, cfg["min"]), cfg["max"])
        return Decimal(raw).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # -- case construction ---------------------------------------------------------------

    def _build_dispute(
        self, index: int, hard_class: HardCaseClass, sub_type: TxnSubType, reason_code: str
    ) -> DisputeRecord:
        cfg = self.config
        is_p2p = sub_type is not TxnSubType.U2

        merchant_type = MerchantType.LARGE
        if hard_class is HardCaseClass.SMALL_OFFLINE:
            negative = self.rng.random() < cfg["merchant"]["small_offline_negative_share"]
            merchant_type = MerchantType.LARGE if negative else MerchantType.SMALL_OFFLINE
        elif not is_p2p and self.rng.random() < cfg["merchant"]["small_offline_share"]:
            merchant_type = MerchantType.SMALL_OFFLINE

        fraud = False
        acct_count = int(self.rng.integers(0, cfg["caps"]["background_max_ifsc_acct"] + 1))
        vpa_count = int(self.rng.integers(0, cfg["caps"]["background_max_vpa_pair"] + 1))

        if hard_class is HardCaseClass.CAPPED:
            acct_count, vpa_count, fraud = self._capped_counters()
        elif self.rng.random() < cfg["fraud"]["background_share"]:
            fraud = True

        acquiring_psp, beneficiary_bank = self._institutions(hard_class, is_p2p)

        timestamp = self._sample_timestamp()
        return DisputeRecord(
            dispute_id=f"SYN-{index:06d}",
            txn_id=f"SYNTXN{index:010d}",
            rrn=f"{900000000000 + index}",
            tran_id=f"SYNTRN{index:08d}",
            cust_ref=f"{700000000000 + index}",
            amount=self._sample_amount(),
            payer_vpa=f"payer{index:06d}@okbank",
            payee_vpa=(f"payee{index:06d}@okbank" if is_p2p else f"merchant{index:06d}@okpsp"),
            payer_ifsc=f"HDFC{int(self.rng.integers(0, 10000)):07d}",
            payer_account=f"5010{index:08d}",
            payee_merchant_id=None if is_p2p else f"MID{index:08d}",
            mcc=None if is_p2p else self._choice(["5651", "5812", "5411", "5251", "4900"]),
            acquiring_psp=acquiring_psp,
            beneficiary_bank=beneficiary_bank,
            txn_timestamp=timestamp,
            reason_code=reason_code,
            txn_sub_type=sub_type,
            merchant_type=merchant_type,
            settlement_cycle=self._choice(["DC1", "DC2"]),
            settlement_date=(timestamp + timedelta(days=1)).date(),
            tcc_ret_status="TCC" if reason_code == "RC_121" else None,
            fraud_flag=fraud,
            chargeback_count_ifsc_acct=acct_count,
            chargeback_count_vpa_pair=vpa_count,
        )

    def _capped_counters(self) -> tuple[int, int, bool]:
        """Counters for a CAPPED case: at the boundary, over a limit, or over but fraud-exempt."""
        cfg = self.config["caps"]
        cd1_limit = self.caps["CD1"]["limit"]
        cd2_limit = self.caps["CD2"]["limit"]
        overshoot = int(self.rng.integers(0, cfg["breach_overshoot_max"] + 1))

        if self.rng.random() < cfg["boundary_share"]:
            # One below each limit: the last chargeback the caps still allow.
            return cd1_limit - 1, cd2_limit - 1, False

        fraud = self.rng.random() < self.config["fraud"]["capped_exempt_share"]
        if self.rng.random() < 0.5:
            return cd1_limit + overshoot, int(self.rng.integers(0, cd2_limit)), fraud
        return int(self.rng.integers(0, cd1_limit)), cd2_limit + overshoot, fraud

    def _institutions(
        self, hard_class: HardCaseClass, is_p2p: bool
    ) -> tuple[str | None, str | None]:
        cfg = self.config["deemed_approval"]
        banks = cfg["institutions"]
        if hard_class is HardCaseClass.DEEMED_APPROVAL_P2M:
            if self.rng.random() < cfg["matching_share"]:
                same = self._choice(banks)
                return same, same
            first, second = self.rng.choice(banks, size=2, replace=False)
            return str(first), str(second)
        if not is_p2p and self.rng.random() < cfg["background_populated_share"]:
            first, second = self.rng.choice(banks, size=2, replace=False)
            return str(first), str(second)
        return None, None

    def _build_evidence(
        self,
        dispute: DisputeRecord,
        hard_class: HardCaseClass,
        accepted: list[str],
    ) -> tuple[list[EvidenceArtifact], list[Contradiction]]:
        """Build the artifact set for a case, plus any contradiction planted in it."""
        did = dispute.dispute_id
        stamp = dispute.txn_timestamp + timedelta(days=1)

        def artifact(n: int, etype: str, valid: bool, note: str) -> EvidenceArtifact:
            return EvidenceArtifact(
                evidence_id=f"EV-{did}-{n}",
                dispute_id=did,
                type=etype,
                source=self._choice(
                    ["merchant order system", "acquiring bank CBS", "merchant finance ledger",
                     "beneficiary bank CBS", "merchant back office"]
                ),
                timestamp=stamp,
                description=f"{etype.replace('_', ' ')} filed against the disputed transaction",
                is_valid=valid,
                validity_note=note,
            )

        weak_notes = self.config["invalid_reasons"]["weak"]
        misleading_notes = self.config["invalid_reasons"]["misleading"]
        good = "legible, internally consistent, and matched to the disputed transaction"

        if hard_class in (HardCaseClass.ABSENT, HardCaseClass.FABRICATION_TEMPTING):
            return [], []

        if hard_class is HardCaseClass.IRRELEVANT:
            pool = _non_accepted_types(accepted)
            count = int(self.rng.integers(1, 3))
            picks = self.rng.choice(pool, size=min(count, len(pool)), replace=False)
            return [
                artifact(i, str(t), True, "genuine and legible, but not a type this reason code "
                                          "accepts")
                for i, t in enumerate(picks, start=1)
            ], []

        if hard_class is HardCaseClass.SMALL_OFFLINE:
            note = (
                "issued by the acquirer in the prescribed format and tied to the disputed "
                "transaction"
            )
            return [artifact(1, self.declaration, True, note)], []

        if hard_class is HardCaseClass.WEAK:
            count = int(self.rng.integers(1, 3))
            return [
                artifact(i, self._choice(accepted), False, self._choice(weak_notes))
                for i in range(1, count + 1)
            ], []

        if hard_class is HardCaseClass.MISLEADING:
            refund_type = self._choice(sorted(self.refund_types & set(accepted)))
            return [artifact(1, refund_type, False, self._choice(misleading_notes))], []

        if hard_class is HardCaseClass.CONTRADICTORY:
            first = self._choice(accepted)
            second = self._choice([t for t in accepted if t != first] or accepted)
            artifacts = [artifact(1, first, True, good), artifact(2, second, True, good)]
            planted = Contradiction(
                evidence_ids=[a.evidence_id for a in artifacts],
                detail=(
                    f"{first} and {second} are each valid on their own but cannot both hold as a "
                    f"defence of this transaction"
                ),
                critical=True,
            )
            return artifacts, [planted]

        # STRONG, CAPPED and DEEMED_APPROVAL_P2M all rest on valid accepted evidence.
        if hard_class is HardCaseClass.DEEMED_APPROVAL_P2M and self.rng.random() < 0.35:
            return [], []

        artifacts = [artifact(1, self._choice(accepted), True, good)]
        if len(accepted) > 1 and self.rng.random() < self.config["evidence"]["corroboration_share"]:
            other = self._choice([t for t in accepted if t != artifacts[0].type])
            artifacts.append(artifact(2, other, True, good))
        return artifacts, []

    def _narrative(self, hard_class: HardCaseClass, dispute: DisputeRecord) -> str:
        templates = self.config["narratives"][hard_class.value]
        template = self._choice(templates)
        return template.format(
            amount=f"{dispute.amount:,.2f}",
            item=self._choice(self.config["items"]),
            days=int(self.rng.integers(2, 40)),
        )

    # -- labelling -----------------------------------------------------------------------

    def _win_probability(
        self,
        sufficiency: Sufficiency,
        valid_accepted: int,
        declaration_only: bool,
        contradictory: bool,
    ) -> float:
        """P(the representment succeeds if filed), from the documented noise model."""
        cfg = self.config["noise_model"]
        logit = cfg["base_logit"][sufficiency.value]
        logit += cfg["corroboration_bonus"] * max(valid_accepted - 1, 0)
        if declaration_only:
            logit += cfg["declaration_penalty"]
        if contradictory:
            logit += cfg["contradiction_penalty"]
        logit += float(self.rng.normal(0.0, cfg["adjudicator_sigma"]))
        probability = 1.0 / (1.0 + math.exp(-logit))
        return float(min(max(probability, cfg["floor"]), cfg["ceiling"]))

    def build_case(self, index: int) -> GeneratedCase:
        """Construct one synthetic case and label it by running the engine over it."""
        hard_class = self._sample_class()
        sub_type = self._sample_sub_type(hard_class)
        reason_code, accepted = self._sample_reason_code(sub_type, hard_class)
        dispute = self._build_dispute(index, hard_class, sub_type, reason_code)
        evidence, contradictions = self._build_evidence(dispute, hard_class, accepted)
        narrative = self._narrative(hard_class, dispute)

        sufficiency, decision, rationale, source = label(dispute, evidence, contradictions)

        declaration_only = bool(evidence) and all(a.type == self.declaration for a in evidence)
        probability = self._win_probability(
            sufficiency=sufficiency,
            valid_accepted=len([a for a in evidence if a.is_valid]),
            declaration_only=declaration_only,
            contradictory=bool(contradictions),
        )
        outcome = (
            RealizedOutcome.WON if self.rng.random() < probability else RealizedOutcome.LOST
        )

        case = SeedCase(
            case_id=f"syn_{index:06d}",
            version=DATASET_VERSION,
            hard_case_class=hard_class,
            narrative=narrative,
            dispute=dispute,
            evidence=evidence,
            ground_truth=GroundTruth(
                expected_reason_code=reason_code,
                expected_sufficiency=sufficiency,
                expected_decision=decision,
                rationale=rationale,
                source_rule=source,
                realized_outcome=outcome,
                win_probability=round(probability, 4),
            ),
        )
        return GeneratedCase(case=case, contradictions=contradictions)

    def generate(self, n: int | None = None) -> list[GeneratedCase]:
        """Generate the whole dataset."""
        count = n if n is not None else self.config["n_cases"]
        return [self.build_case(i) for i in range(1, count + 1)]


def label(
    dispute: DisputeRecord,
    evidence: list[EvidenceArtifact],
    contradictions: list[Contradiction] | None = None,
) -> tuple[Sufficiency, Decision, str, str]:
    """Label a case by running the decision engine with an oracle verifier.

    This is the single labelling path: the generator uses it, and the seed-reproduction check
    uses it against the hand-written labels. It does not restate the evidence map, the caps
    convention or the declaration rule — the engine reads all of those from the rulebook.
    """
    planted = {dispute.dispute_id: list(contradictions or [])}
    engine = DecisionEngine(verifier=OracleVerifier(planted))
    outcome = engine.decide(dispute, evidence)
    return (
        outcome.sufficiency.level,
        outcome.decision,
        outcome.reason,
        outcome.sufficiency.source,
    )
