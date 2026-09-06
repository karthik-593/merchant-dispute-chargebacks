# merchant-dispute-chargebacks

An evidence-grounded, cost-sensitive multi-agent system for **UPI merchant disputes**. Per
dispute it decides whether winning evidence exists, assembles a grounded representment when it
does, and cleanly concedes or escalates when it does not.

The differentiator is not text generation. It is the pair of layers around it: a deterministic
**evidence-sufficiency engine** and a **grounding/quality verifier**.

## Problem

When a customer disputes a UPI transaction, the merchant's side has a narrow, hard-deadlined
window to file a representment backed by the specific evidence types the reason code demands.
Filing without sufficient evidence loses and costs money; conceding a winnable dispute also
costs money. The decision is an expected-value one, not a coin flip on a model's confidence.

Two questions are kept strictly separate:

- **Sufficiency (deterministic, L0)** — are the required evidence *types* present for this reason
  code? A pure lookup against `configs/rulebook/`. No LLM. This is baseline B0 and the safety
  backbone.
- **Validity (LLM, L2)** — does an artifact actually *prove* what it claims? Is a "refund"
  completed or only initiated? Does the invoice amount match the disputed transaction?

Every domain fact — reason code, required evidence, TAT, fee, cap — traces to a specific NPCI or
RBI circular and section. Rules live in `configs/rulebook/*.yaml` as data with a `source` tag,
never in Python branches.

## Scope

- **In scope:** UPI merchant-dispute chargebacks, the evidence-sufficiency family (goods /
  fulfilment / refund / duplicate). Reason codes RC 1061–1085, 108, 1081, 1084, 1085, 121.
- **Out of scope for now:** pure payment-failure (debited-not-credited as a rail failure), and
  the card networks (Visa / Mastercard / RuPay). Card rails plug into the same agent core only
  after UPI v1 is solid.
- **Vendor-neutral** — no payment-aggregator-specific coupling.

## Architecture

Seven layers, each independently testable, with the multi-agent graph as the integrating
layer on top:

| Layer | Responsibility | LLM? |
|---|---|---|
| L0 | Deterministic core: reason-code → required-evidence lookup, caps gate, TAT clock, fee/EV math, hard safety gates | No |
| L1 | Retrieval over the circulars corpus — governing rule, evidence spec, reject taxonomy | Embeddings |
| L2 | Narrative → reason-code classification, artifact-validity verification, contradiction detection, grounded drafting, abstention | Yes |
| L3 | Calibrated `P(successful defense)` from structured features | No |
| L4 | Grounding verifier: claim → evidence support, fabrication interception, bounded redraft | Mixed |
| L5 | Expected-value FILE / CONCEDE / ESCALATE decision plus hard gates | No |
| L6 | Multi-agent orchestration: Triage → Evidence → Sufficiency → Draft → Verify → Decide | — |

Decisions are expected-value, never a threshold on model confidence:
`EV(FILE) = P(win)·recovery − filing_cost − expected_failure_cost`, compared against conceding
or escalating. Critical evidence absent or a critical grounding failure means no filing at all.

## Repository layout

```
configs/          runtime config + rulebook/ (domain rules as data, each with a source tag)
data/             DVC-tracked datasets (contents not in git)
refs/circulars/   source NPCI/RBI circulars; see refs/README.md for the index and OCR status
src/              data · retrieval · llm · decision · agents · evaluation · serving
experiments/      experiment records (EXP-*), MLflow-linked
tests/            pytest; the deterministic core is test-first
app/ docker/      serving surfaces and containerisation
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # create .venv and install runtime + dev dependencies
uv run ruff check .      # lint
uv run pytest            # tests
```

Datasets are tracked with DVC (`dvc pull` once a remote is configured); experiment runs are
logged to MLflow at the tracking URI in `configs/config.yaml` (`mlflow ui` to browse).

## Status

**M0 — project foundation.** Repository skeleton, tooling, configuration, and the source-circular
corpus are in place. No domain, agent, or ML logic is implemented yet; `configs/rulebook/`
contains the ported sufficiency map plus stubs for TAT, fees, caps, and the reject taxonomy.

Next: **M1** — port the remaining domain rules (TAT, fees, caps, reject taxonomy) into
`configs/rulebook/`, each entry carrying its circular and section, then **M2** seed dataset and
**M3** the vertical slice: a dispute goes in, a rules-grounded decision plus audit trail comes
out.
