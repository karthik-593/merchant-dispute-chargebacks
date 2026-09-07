# Dataset card — `dataset-v1.0`

## Name and status

**`dataset-v1.0`** — the synthetic dispute set for the UPI merchant-dispute agent.

Distinct from `seed-v1`, which is the 40-case hand-written fixture set. That one is the human ground truth; this one is generated, far larger, and carries a stochastic outcome label the seed set does not have.

## Generation method

Produced by `src/data/generator.py` from `configs/generator.yaml`, seeded with **42**. Regenerating with the same seed reproduces the files byte for byte.

The generator holds **no domain rules of its own**. Reason codes, accepted evidence types, the small/offline declaration substitution and the cap limits are all read from `configs/rulebook/` at generation time. Labelling runs the decision engine itself rather than a second copy of the rules: the generator wires that engine with an oracle verifier that reports the contradictions it planted, while the B0 baseline wires the same engine with the stub verifier that cannot detect them. The two therefore agree by construction everywhere except on contradiction detection.

Each case is built as a chain: transaction fields, then the dispute and a natural-language narrative drawn from several surface variants per case class, then the evidence artifacts with their validity labels, then the ground truth.

- **Cases**: 2000
- **Seed**: 42
- **Transaction window**: 2025-07-01 to 2026-06-30
- **Format**: one JSON object per line, one file per split

## Distributions

### By reason code

| reason code | cases | share |
|---|---:|---:|
| `RC_108` | 531 | 26.6% |
| `RC_1064` | 387 | 19.4% |
| `RC_1065` | 231 | 11.6% |
| `RC_1061` | 210 | 10.5% |
| `RC_1081` | 127 | 6.3% |
| `RC_1062` | 118 | 5.9% |
| `RC_1085` | 113 | 5.7% |
| `RC_1063` | 105 | 5.2% |
| `RC_1084` | 102 | 5.1% |
| `RC_121` | 76 | 3.8% |

### By transaction sub-type

| sub-type | cases | share |
|---|---:|---:|
| `U2` | 1718 | 85.9% |
| `U3` | 209 | 10.4% |
| `UC` | 73 | 3.6% |

### By hard-case class

| class | cases | share |
|---|---:|---:|
| `STRONG` | 620 | 31.0% |
| `ABSENT` | 246 | 12.3% |
| `WEAK` | 222 | 11.1% |
| `MISLEADING` | 192 | 9.6% |
| `CONTRADICTORY` | 173 | 8.6% |
| `IRRELEVANT` | 153 | 7.6% |
| `FABRICATION_TEMPTING` | 126 | 6.3% |
| `CAPPED` | 106 | 5.3% |
| `SMALL_OFFLINE` | 98 | 4.9% |
| `DEEMED_APPROVAL_P2M` | 64 | 3.2% |

### By sufficiency

| sufficiency | cases | share |
|---|---:|---:|
| `STRONG` | 1008 | 50.4% |
| `ABSENT` | 578 | 28.9% |
| `WEAK` | 414 | 20.7% |

### By decision

| decision | cases | share |
|---|---:|---:|
| `CONCEDE` | 992 | 49.6% |
| `FILE` | 779 | 39.0% |
| `ESCALATE` | 173 | 8.6% |
| `RGNB` | 56 | 2.8% |

### By realized outcome

| outcome | cases | share |
|---|---:|---:|
| `LOST` | 1083 | 54.1% |
| `WON` | 917 | 45.9% |

Other splits of interest: 185 small/offline merchants, 78 fraud-flagged, 47 where the acquiring PSP and merchant bank are the same institution.

## The two labels

Every case carries both, and they answer different questions.

### 1. Deterministic rule label

`expected_sufficiency` and `expected_decision`, computed by the decision engine from the rulebook. Policy order is the engine's: the caps gate short-circuits first, then a critical contradiction escalates, then sufficiency decides — STRONG files, WEAK and ABSENT concede.

### 2. Stochastic realized outcome

`realized_outcome` (WON/LOST) with its `win_probability`. This is the **counterfactual** question: would the representment have succeeded, had it been filed? It is defined for every case regardless of the decision taken, it is drawn rather than derived, and it **never feeds back into the deterministic decision**. It is the target the calibrated scorer will learn.

The model:

```
logit(p) = base[sufficiency]
         + 0.35 * (valid accepted artifacts - 1)
         + -0.45   if the case rests on a declaration alone
         + -1.1   if the bundle contradicts itself
         + Normal(0, 0.85)   adjudicator variance

p = clip(sigmoid(logit), 0.02, 0.97)
realized_outcome ~ Bernoulli(p)
```

Base logits: STRONG 1.55, WEAK -1.3, ABSENT -2.6.

Three deliberate properties. **Adjudicator variance** means two identical evidence bundles can land differently, because two panels do. **Corroboration** rewards a second valid artifact, so evidence strength is graded rather than binary. **The floor and ceiling** keep any case from being a certainty, so a scorer cannot reach a perfect AUC by memorising the sufficiency label — which is the whole point of injecting noise at all. Realised win rate across the set is 917/2000 = 45.9%, with modelled probabilities spanning 0.02 to 0.97 (mean 0.47).

## Splits

**Temporal, by `txn_timestamp`.** Cases are ordered by transaction time and cut, so the test set is strictly the latest slice — trained on the past, judged on what came after. A shuffled split would leak future information; this cannot.

| split | cases | share |
|---|---:|---:|
| train | 1400 | 70.0% |
| val | 300 | 15.0% |
| test | 300 | 15.0% |
| dev (subset of train) | 50 | — |

The **test split is locked**: `TEST_SET_LOCK.json` records its SHA-256 and case count. It must not be used to select a model, a prompt or a threshold. A changed hash means the held-out set moved and anything measured on it is void.

The dev subset is drawn from train only, so fast iteration can never touch held-out data.

## Consistency checks

These are what make the dataset trustworthy: they test that the generator and the engine agree about the rules, and that both agree with a human.

```
Check A - B0 engine vs generated labels
  cases checked        : 2000
  sufficiency agreement: 2000/2000
  decision agreement   : 1827/2000
  expected gap         : 173 contradiction case(s) B0 cannot detect
  UNEXPECTED mismatches: none
  result               : PASS

Check B - generator labelling vs hand-written seed-v1 labels
  cases checked        : 40
  sufficiency agreement: 40/40
  decision agreement   : 40/40
  UNEXPECTED mismatches: none
  result               : PASS
```

**Check A** confirms the only disagreement between generated labels and the B0 engine is the 173 contradiction case(s) B0 structurally cannot detect. Any other divergence would mean the two had drifted apart.

**Check B** is the stronger one. It applies the generator's labelling to the 40 seed-v1 inputs and compares against labels a human wrote by hand, ESCALATE cases included. It passing means the encoded rules agree with human judgement on every hand-reviewed case; had it failed, the finding would be reported rather than the check relaxed.

## Known limitations

- Narratives are template-drawn, not written. They carry enough surface variation to exercise a classifier's robustness a little, but they are not a substitute for real complaint text and should not be used to claim anything about natural-language performance.
- Contradictions are planted structurally — two individually valid artifacts of accepted types marked as conflicting — rather than being emergent from artifact content. A real verifier will face subtler conflicts than these.
- The noise model is a stated assumption, not an estimate from observed outcomes. Its base rates are plausible rather than measured, so absolute win rates here carry no external meaning; only relative ordering is intended to.
- Deemed approval is carried as metadata and deliberately changes no outcome, matching the engine. When the presumption is given weight in the policy, this dataset will need regenerating.
