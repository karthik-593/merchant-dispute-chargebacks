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
logged to MLflow at the tracking URI in `configs/config.yaml`
(`mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse).

Run the thread:

```bash
uv run python -m src.evaluation.run_seed        # whole seed set, scored, logged to MLflow
uv run python -m src.serving.cli --case seed_001 --json   # one case, full audit record
uv run uvicorn app.main:app --reload            # POST /decide, docs at /docs
```

## Status

**M3 — vertical slice.** A dispute goes in and a rules-grounded FILE / CONCEDE / ESCALATE / RGNB
decision comes out with an audit trail naming every rule that fired and the circular it came
from. The thread runs end to end on the 40 hand-built `seed-v1` fixtures, served over HTTP and a
CLI.

Everything in it is deterministic. This is baseline **B0**: the sufficiency engine and caps gate
read `configs/rulebook/`, while the two LLM-shaped seams — narrative classification and evidence
validity — are stubs behind interfaces (`PassthroughClassifier`, `StubVerifier`). No model is
called, and no ML dependency is installed.

Against the seed ground truth: **sufficiency 40/40, decision 36/40**. The four misses are the
`CONTRADICTORY` cases, which need cross-artifact reasoning the stub verifier cannot do, so the
ESCALATE branch is wired but unreachable. That gap is deliberate and is pinned by a test.

`configs/rulebook/` now holds the sufficiency map and the caps config; `tat.yaml`, `fees.yaml`
and `reject_taxonomy.yaml` are still stubs. Next: the retrieval corpus and the real classifier
and verifier, after which ESCALATE becomes reachable and the decision policy can move from this
deterministic mapping to the expected-value one.
