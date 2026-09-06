# Dataset card — `seed-v1`

## Name and status

**`seed-v1`** — the hand-built seed set for the UPI merchant-dispute agent.

This is **not** "Dataset v1.0". That name is reserved for the synthetic set produced by the data
generating process in a later milestone. `seed-v1` is a small, fully hand-authored fixture set and
is versioned separately for that reason.

## Provenance

**Hand-authored, not generated.** Every case — narrative, evidence set, validity label and ground
truth — was written by hand against the rulebook. No sampler, template engine or model produced
any field. The cases are fictional: identifiers, VPAs, amounts, merchant names and account numbers
are invented and correspond to no real transaction, customer or merchant.

Case content is constrained by `configs/rulebook/reason_code_evidence.yaml`:

- every `reason_code` is one the rulebook declares;
- every evidence `type` is drawn from the rulebook's controlled `evidence_types` vocabulary — no
  case invents an evidence type, a reason code, or a rule;
- every `ground_truth.source_rule` names the circular and section that governs the decision.

Tracked in **git**, not DVC: the set is small and its value is that the ground truth is reviewable
in a diff.

## Size and shape

**38 cases**, one YAML file each, named `seed_<n>_<reason-code>_<what-it-tests>.yaml`.
35 evidence artifacts in total (27 hand-labelled valid, 8 invalid); 9 cases carry no evidence at
all, which is itself the condition under test.

Each file holds a `case_id`, the `seed-v1` version tag, a `hard_case_class`, a plain-language
`narrative`, one `DisputeRecord`, zero or more `EvidenceArtifact`s, and one `GroundTruth`.

### By rulebook entry

Reason codes 108 and 121 appear twice in the rulebook — once for P2M and once for P2P — because
they take different evidence. They are counted here as the separate entries they are. **All 12
entries are exercised.**

| Rulebook entry | Reason code | Sub-type | Cases |
|---|---|---|---|
| `RC_1061` | RC 1061 | U2 | 3 |
| `RC_1062` | RC 1062 | U2 | 2 |
| `RC_1063` | RC 1063 | U2 | 2 |
| `RC_1064` | RC 1064 | U2 | 6 |
| `RC_1065` | RC 1065 | U2 | 4 |
| `RC_108_U2` | RC 108 | U2 | 6 |
| `RC_1081` | RC 1081 | U2 | 2 |
| `RC_1084` | RC 1084 | U2 | 2 |
| `RC_1085` | RC 1085 | U2 | 4 |
| `RC_121_U2` | RC 121 | U2 | 2 |
| `RC_108_U3` | RC 108 | U3 / UC | 3 |
| `RC_121_U3` | RC 121 | U3 / UC | 2 |

### By hard-case class

| Class | Cases | What it exercises |
|---|---|---|
| `STRONG` | 6 | An accepted evidence type is present, valid, and matched to the disputed transaction |
| `ABSENT` | 5 | No accepted evidence type is on file at all |
| `WEAK` | 4 | Accepted type present, but illegible, partial, or not pertaining to the transaction |
| `CONTRADICTORY` | 4 | Two individually valid artifacts assert incompatible facts |
| `MISLEADING` | 4 | Refund initiated, requested, returned unpaid, or partial — shown as completed |
| `FABRICATION_TEMPTING` | 3 | Narrative invites asserting an artifact that does not exist |
| `IRRELEVANT` | 3 | Genuine, legible artifacts of a type the reason code does not accept |
| `SMALL_OFFLINE` | 3 | The acquirer declaration-letter substitution, including its negative case |
| `CAPPED` | 3 | CD1 and CD2 chargeback caps, plus the fraud exemption from them |
| `DEEMED_APPROVAL_P2M` | 3 | Acquiring PSP = merchant bank, so delivery is presumed and must be rebutted — plus the contrast where the two institutions differ |

### By ground truth

| Sufficiency | Cases | Decision | Cases |
|---|---|---|---|
| `STRONG` | 17 | `FILE` | 11 |
| `WEAK` | 8 | `CONCEDE` | 21 |
| `ABSENT` | 13 | `ESCALATE` | 4 |
| | | `RGNB` | 2 |

Other splits: 33 P2M (U2), 3 U3 and 2 UC (P2P); 36 large merchants and 2 small/offline.

## Ground-truth methodology

### Sufficiency — `STRONG` / `WEAK` / `ABSENT`

Sufficiency here is a **hand label over both layers**, not the output of the deterministic
type-presence check alone. The distinction the labels encode is exactly the one the project turns
on: type presence is one question, whether the artifact proves anything is another.

- **`ABSENT`** — no artifact whose *type* appears in the reason code's `required_evidence_any_of`.
  The deterministic check fails outright. A case can be `ABSENT` while holding perfectly genuine
  documents, if those documents are of the wrong type for the reason code (the `IRRELEVANT` class
  is built on this).
- **`WEAK`** — an accepted evidence *type* is present, so a type-only check passes, but the
  artifact fails on content: illegible or partial, not pertaining to the disputed transaction, a
  bare generic statement, or a refund that was initiated rather than completed.
- **`STRONG`** — an accepted type is present, and the artifact is legible, internally consistent,
  and tied to the disputed transaction by reference and amount.

**Small and offline merchants:** an acquirer declaration letter substitutes for documentary
fulfilment evidence and can carry a case to `STRONG`. For a large merchant it does not substitute,
and a case holding only a declaration is `ABSENT`. One case of each is included so the branch is
tested in both directions.

### Decision — `FILE` / `CONCEDE` / `ESCALATE` / `RGNB`

- **`FILE`** — sufficiency is `STRONG`, no auto-loss condition is present, and the caps gate does
  not fire. This is the domain spec's REPRESENT.
- **`CONCEDE`** — sufficiency is `ABSENT` or `WEAK`, or an auto-loss condition is present, or a
  deemed-approval delivery presumption cannot be backed by fulfilment evidence.
- **`ESCALATE`** — the evidence is individually valid but *critically contradictory*. The agent
  does not adjudicate a conflict in the merchant's own record; it routes it to a human.
- **`RGNB`** — the claim is genuine but blocked by a chargeback cap, so it takes the remitter
  good-faith negative chargeback path rather than a chargeback.

`RGNB` is a fourth decision value because the domain spec's outcome tree has four outcomes and the
caps path resolves to none of the other three. Labelling a capped case `CONCEDE` or `ESCALATE`
would assert something the rulebook contradicts.

Ordering matters and the fixtures encode it: the **caps gate runs before the evidence question**,
which is why two cases carry `STRONG` evidence and still resolve to `RGNB`. Fraud transactions are
exempt from the caps, so one case sits far above both limits and still decides on its evidence.

### Evidence validity labels

Each artifact carries `is_valid` plus a `validity_note` saying why. `is_valid: false` never means
the document is fake — it means the document does not prove what the representment would need it
to prove. The eight invalid artifacts cover: transaction details not matching the invoice, generic
statements with no record, illegible and partial captures, refunds that were initiated but not
completed, a refund returned unpaid, a refund request ticket with no payment, and a partial refund
with the balance still disputed.

## How this set is meant to be used

- **Reference fixture set for the deterministic L0 engine.** These cases are the expected answers
  the sufficiency engine, caps gate and decision policy are written against. The `ABSENT` and
  `IRRELEVANT` cases in particular pin down that type presence is judged against the reason code,
  not against whether a document looks impressive.
- **Sanity check for the synthetic generator.** When the DGP arrives, its output should reproduce
  the behaviour these cases describe. A generator that cannot produce this distribution of hard
  cases — or that disagrees with these labels — is wrong.
- **Not a training set and not a benchmark.** 38 cases is far too few, and the distribution is
  deliberately weighted toward adversarial conditions. The `CONCEDE`-heavy split reflects that
  weighting, **not** any real-world base rate. Nothing here should be read as a measurement of how
  often disputes are winnable.

## Known limitations

- Deemed approval is now represented **structurally**, not only through `hard_case_class` and the
  narrative. `DisputeRecord` carries optional `acquiring_psp` and `beneficiary_bank` identifiers,
  and the derived property `acquiring_psp_is_merchant_bank` reads the condition off them: true
  only for a P2M transaction where both are set and equal. It is computed, never stored, so the
  fixtures cannot assert a condition their own identifiers contradict. Three cases exercise it —
  two where the institutions match and one contrast where they differ — and the remaining 35
  leave both fields unset, which reads as false.
- Sufficiency labels fold in validity judgements. Once the L0 engine and the validity verifier are
  separate running components, it may be worth splitting the label into a deterministic
  type-presence result and a separate validity verdict.
- Two small/offline cases only. The declaration path deserves wider coverage in the synthetic set.
