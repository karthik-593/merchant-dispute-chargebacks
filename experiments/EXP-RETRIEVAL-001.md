# EXP-RETRIEVAL-001 — M7 retriever × embedding grid

**Status: FROZEN (M7 closed, 2026-09-09).** `CHUNKER_VERSION = structure_aware/row-grain`,
`RETRIEVER_VERSION = dense`, `EMBEDDING_VERSION = bge-small-en-v1.5`, `RERANKER_VERSION = NONE
(rejected, scoped)`. Recorded with provenance in `configs/versions.yaml`; one MLflow run under
`arm=freeze`. Corpus `corpus-v1.4`, query set `retrieval-v1.2.2`.

**Every lever was tested and priced before anything was frozen:**

| lever | configuration | outcome |
| --- | --- | --- |
| encoder | 6 models, 2 sizes, 3 recipes | **0 hard rows**; ensemble union = best single (3/11) |
| depth | K_retrieve 25 → 100 | **1 row** (h02) |
| reranker | cross-encoder over row-grain chunks | **negative** — 1 win, 4–9 regressions |
| reranker | cross-encoder over ±2-row parents | **negative** — h02 and h32 both worse |
| expansion | parent = ±2 rows | **0 rows** recovered |
| expansion | parent = full table | **0 rows**; the +0.12 all-50 rise is the B3 ruler artefact at 25% of the corpus per query |

**`bge-small` was kept on per-row evidence, not on the aggregate that hid the failures earlier.**
Six encoders were tested; nothing is better on either axis. It has the best aggregate (0.640), the
best reachability for any future second stage (8/11 within depth 100 against e5-small's 4), and 8×
lower VRAM than any large candidate. The one argument against it is a single row of hard-set top-5
coverage, inside the noise floor at n=11.

**The residual finding is why retrieval closes here.** Eight rows (h01–h05, h09, h32, h35) fail on
a **query–corpus vocabulary gap**, not a pipeline defect: the corpus states rules in domain
language and the queries ask in lay language, and the bridge between them is knowledge the corpus
never writes down. No similarity function crosses a bridge the text does not contain. Filed as the
M8 query-rewrite benchmark at `data/eval/m8_query_rewrite_eval.yaml`.
**Revision 2 (2026-09-08).** Supersedes revision 1. Re-run end to end after two changes to the
inputs, neither of which was a retrieval change:

1. **Corpus reconciliation.** OC 208A Annexure A was reconciled against the verified rulebook and
   a curated record now supersedes the OCR'd scan — 15 of its 28 reject codes were damaged.
   Corpus 35 → **36** documents. Log: `configs/corpus/oc_208a_reconciliation.yaml`.
2. **Query set amended and re-frozen** to `retrieval-v1.1`. One query (q25) gained a target; the
   other 25 are hash-identical to v1.

Revision 1's numbers were taken on 35 documents and `retrieval-v1`. **They are stale — do not
quote them.** The comparison between the two is recorded under "What moved, and why" below.

**MLflow:** `M7-retrieval`, 36 runs on v1.1 (21 `grid=main`, 15 `grid=alpha_sweep`);
`M6-chunking`, 4 runs on v1.1. **Harness:** `src/evaluation/eval_retrieval.py`.
**Scoring set:** `data/corpus/retrieval_queries.yaml`, `retrieval-v1.1`, 26 queries,
`queries_sha256 = 257b38f5…97a4151`. **Corpus:** `circulars.jsonl`, 36 documents.
**Hardware:** RTX 4060 Laptop 8 GB, CUDA 12.6, torch 2.14.0+cu126.

```bash
uv run python -m src.evaluation.eval_retrieval --alpha-sweep bge-small-en-v1.5
```

## Revision 3 (2026-09-09) — source-verification pass, `retrieval-v1.2.1`

All hard-query ground truth was checked against the OC 208, OC 184B and OC 208A **source PDFs**.
This is the provenance spine for M18: §1 evidence map (OC 208 Table C, all 12 rows), §2/§3
branches, §5 RGNB (OC 184B), reject codes (OC 208A), fees (OC 208 p1) — all confirmed.

### The one behavioural error found

**The §3 NVB/NVR category boundary.** Annexure A's Adj Flag column carries three categories, not
one flat list. The catalog was scoped "1130–1157" with every code treated as a fail — but
**1130 ("Beneficiary customer account credited successfully") and 1131 are NVB: verdict FOR the
beneficiary, meaning the representment WON.** A verifier reading them as auto-loss would have
rejected valid representments on the strength of the codes that say they succeeded.

| band | codes | meaning | treatment |
| --- | --- | --- | --- |
| NVB | 1126–1131 | verdict FOR — representment succeeded | **never a fail** |
| NVR | 1132–1157 | verdict AGAINST | the auto-loss catalog |
| ACC | 1105–1107 | arbitration continuation | out of scope |

Fixed: the two bands are separate structures (`nrp_verdict_for_codes`, `nrp_verdict_against_codes`),
`nrp_fail_codes()` returns NVR only, and `tests/test_reject_taxonomy.py` fails if NVB is ever
folded back in. 1126–1129 belong to NVB but are **not** listed — OCR could not resolve their rows,
and inventing a description for a code the verifier acts on is the failure this file exists to
prevent.

### The correctness fix carried a retrieval cost

Rendering the verdict band on every row added a line identical across 26 reject codes. The
reject-taxonomy curated record grew 4,983 → **7,016 chars**, RGNB 1,745 → 2,113, and
`structure_aware` went 326 → 334 chunks. On the best cell, `structure_aware / dense / bge`,
**rule@5 fell 0.700 → 0.620** — four queries (h15, h22, h32, q25), two of them reject-taxonomy.

**Attribution, both directions.** One query at n=50 moves a rate by at most 0.020, so a 0.080 drop
**cannot** be the h34 anchor edit. h34 itself is answerable at rank 2 and appears in neither miss
list. The movement is the **corpus text change**, not the query change: adding boilerplate to every
row makes near-duplicate rows *more* similar, which is precisely the signal the near-duplicate
queries exist to test. The fix is behaviourally necessary and the retrieval cost is real; both are
recorded rather than netted off. A remedy exists — carry the band as record-level metadata rather
than a per-row line — and is **not applied**, pending review.

### Stage 1 follow-up (corpus-v1.2): the band moved, and the fix was only half a fix

The verdict band moved from per-row text to record-level metadata — carried on `source_section`,
stated once in the header. Per-row text is now the code and its reason string only; the per-row
`Source:` line and reconciliation marker went with it, being equally uniform. The categorisation
is untouched: `nrp_verdict_for_codes` / `nrp_verdict_against_codes` and the NVB guard test all
stand. The reject record halved, 7,016 → **3,511 chars**.

**`structure_aware / dense / bge` rule@5: 0.620 → 0.640. Pre-band it was 0.700.** Two of the four
regressed queries recovered (**h15, h32**); two did not (**h22, q25**), and **h30 newly
regressed**. Net against pre-band: three regressions, no new hits.

**Why it only half worked, and it is worth knowing.** Stripping the uniform boilerplate made each
row so short that it fell under `structure_aware`'s 24-token merge floor. The reject record now
chunks into **13 chunks, every one carrying 2–3 codes** — against 20 chunks pre-band and 26 with
it. So the fix removed one source of near-duplicate confusion and created another: instead of 26
rows sharing an identical verdict line, adjacent near-duplicate rows now share a *chunk*. h30
(1157) regressed for exactly this reason — it sits in a chunk with 1156.

h22 is a separate cause and should not be attributed here: it anchors the OC 208 §C evidence map,
not the reject taxonomy, and its regression traces to the RC 1085 verbatim change (Fix 4) altering
that record's text.

**The generalisation stands and is worth more than the number.** Metadata rendered into
retrievable text is a retrieval hazard even when correct — the same defect class as the B3
table-vs-row artifact and the `RC_1064` tokenizer bug. But so is text that is *too* terse for the
chunker's floor. Not fixed here: the remedy is a chunker parameter (`STRUCTURE_MIN_TOKENS`), which
would move every M6 number, and that is a separate decision.

### Stage 2 (retrieval-v1.2.2 / corpus-v1.3): the verbatim-anchor pass, completed

All nine RGNB row descriptions, the Annexure 1 trigger and deemed-acceptance prose, and the
NA1/NA2 penalty cells were read from the OC 184B page 3 table image — the rotated table OCR
destroyed — and now sit in the curated record verbatim. `caps.yaml` keeps the gloss as `meaning`;
`source_text` carries the quoted cell.

Four anchors moved off a gloss: **q14** (case only — matching is case-insensitive, so no metric
can move from it), **h03** → `Yes/No (As chosen by Bank)`, **h09** → `otherwise the adjustment
window will be closed on deemed acceptance`, **h10** → `when URCS declines the normal chargeback
with CD1 & CD2 reason code`. h04 and h05 needed no change — their fragments survive verbatim
inside the longer source strings. 48 queries hash identically.

**A transcription rule, logged not applied silently.** ND2's source cell renders the transaction
type as `P2m`. Normalised to `P2M` and recorded in `caps.yaml` under
`rgnb_responses.transcription_rules`: every other row in that column reads P2M, and P2M is the
sub-type identifier used throughout the rulebook, so this is a scan rendering artefact rather than
a distinct value.

**One field is still a gloss and is named as such:** the ND1/ND2 penalty cells. No query anchors
them, and they were **not** inferred from NA1's verbatim `Yes (Mandatory)`.

**Result: 14 of 21 cells moved, all by one or two queries (0.02–0.04 at n=50).** The diagnosis
cell — `structure_aware / dense / bge` — held at rule@5 **0.640 with an identical miss set**, so
the anchor work changed nothing there. Cause is split and neither half is large: the RGNB record's
text changed (different words, different embeddings) and four anchors changed. Both are inputs,
not retrieval improvements.

**Three of the four amended queries still miss**, and the reason is worth carrying into Stage 3:
h03 and h09 are unfound at any depth, h10 sits at rank 33. h09 in particular got *harder* — its
verbatim Annexure 1 sentence also appears in the OC 206 generic good-faith circulars, so it went
from one target to four by containment and stopped being RGNB-specific. Moving an anchor onto
source wording is correct and made two queries harder, which is the point of a ruler rather than a
score.

### Stage A (corpus-v1.4): the metadata-in-row-text defect, third instance and closed

The Stage 3 diagnosis found a defect I introduced in Stage 2. A corpus-wide sweep of row chunks
turned up **14 contaminated rows** across all three curated records:

| record | contaminated rows | what leaked in |
| --- | ---: | --- |
| OC 208 §C evidence map | 12 | an identical per-row `Source: OC 208 §C` line |
| OC 208A reject taxonomy | 1 | the record's trailing `Source:` line, merged into 1157 |
| OC 184B RGNB | 1 | 87 tokens of transcription note + trailing prose + Source, merged into ND2 |

**One defect, three appearances.** Stage 1 moved the verdict band out of row text; Stage 1b fixed
the fusion that caused; Stage 2's transcription note then went straight back in. The metadata was
correct every time — that is the point. **Trailing record-level text is the specific trap**,
because it merges into the last row. Substantive prose now sits *before* the rows; provenance
moved to `source_section`, where the chunk already carried it. The P2m→P2M rule is not deleted,
only re-rendered.

**Guarded permanently.** `tests/test_chunking.py` now fails if any row chunk contains
`Transcription rule`, `Source:`, `Verdict:`, `reconciliation`, `normalised` or `OCR-corrected`,
and a second test fails if one row chunk exceeds four times the median — which is what trailing
text merging always looks like. That closes all three instances at once. **0 of 50 row chunks are
contaminated.**

### h01: the corrected verdict is worse, and it inverts the reading

ND2's chunk fell 113 → **27 tokens**. h01's ranks went from a contaminated 126/157/129 to a clean
**>200 under all three embeddings.**

The contamination had been *helping*. The 87-token note carried the prose *"otherwise the
adjustment window will be closed on deemed acceptance"* — the semantic bridge to a question asking
when silence counts as agreement. The clean row says *"Deemed acceptance of P2M Generic good faith
chargeback"* and nothing about silence or timeouts. **h01 is a genuine CHUNK case and a stronger
one than the contaminated measurement suggested**, not the marginal one it appeared to be.

**Tally, corrected:** 3 FOUND (h10, h34, h36) · 1 DEPTH (h02) · 5 EMBEDDING (h03, h04, h09, h32,
h35) · 2 CHUNK (h01, h05). h10 moved to FOUND — e5 now ranks it 1 — because the trigger prose it
anchors moved into the header and out of a row's shadow.

### Stage B: the embedding was never the bottleneck

Six encoders on the verified hard set — three incumbents (~22–33M, 384-dim) and three larger
candidates (~335M, 1024-dim): `bge-large-en-v1.5`, `e5-large-v2`, and `gte-large`, the last chosen
as a third *recipe* rather than a third size, since BGE and E5 were already represented.

| encoder | dim | top-5 | past 50 | reach@100 | rule@5 hard | rule@5 all-50 | ms/q | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **bge-small-en-v1.5** | 384 | 2 | 6 | **8** | 0.182 | **0.640** | 10.1 | **0.45** |
| e5-small-v2 | 384 | **3** | 7 | 4 | **0.273** | 0.560 | 9.9 | 0.59 |
| all-MiniLM-L6-v2 | 384 | 1 | 6 | **8** | 0.091 | 0.520 | **6.0** | 0.52 |
| bge-large-en-v1.5 | 1024 | 2 | 7 | 6 | 0.182 | 0.600 | 16.7 | 2.52 |
| e5-large-v2 | 1024 | **3** | 6 | 6 | **0.273** | 0.580 | 19.9 | 3.86 |
| gte-large | 1024 | 2 | 7 | 6 | 0.182 | 0.580 | 18.7 | 4.13 |

**Scaling 8× the parameters and 2.7× the dimensions buys nothing.** `e5-large` ties `e5-small` at
3/11. Every large model is *worse* than `bge-small` on the aggregate. And the decisive number:

**The per-row ensemble ceiling is 3/11 — a gain of ZERO over the best single encoder.** All six
encoders succeed on the same three rows (h10, h34, h36) and fail on the same eight. An ensemble
would triple encode cost and index size to buy nothing at all.

**Four rows are buried past rank 50 by every one of the six**: h01, h02, h03, h09. No encoder
reaches them; a different encoder cannot invent signal a 26-token row does not carry.

**This retires the Stage 3 "EMBEDDING" verdict.** Those five rows looked like an embedding problem
because weak encoders disagreed about *where* to bury them. Given six encoders spanning two size
classes and three training recipes, none solves any of them. The disagreement was noise between
models that all fail.

**Recommendation: keep `bge-small-en-v1.5`.** Best aggregate (0.640), best reachability for a
future reranker (8/11 within depth 100), and 8× cheaper in VRAM than any large candidate. The only
argument against it is one row of hard-set top-5 coverage, which is inside the noise floor. The
earlier choice was made on aggregate evidence and happens to survive per-row evidence too — not
because the aggregate was right, but because nothing else is better on either.

**The real residue is the chunk.** Eight of eleven hard rows are unreachable at top-5 by any
encoder, and the two clean CHUNK cases (h01 at 27 tokens, h05 at 26) show why. That is
parent-expansion work, and it is now the only lever left with evidence behind it.

### The B3 ruler defect (unchanged)

`whole_document` and `sentence` still inflate on near-duplicate queries: the entire 9-row RGNB
table is one 260-token chunk and the whole reject taxonomy one 759-token chunk, so those chunkers
score a rule-hit by retrieving *the table* without ever isolating the row. `structure_aware` is
the only chunker where a hit means the row was found. Unfixed, and it qualifies every cross-chunker
comparison in this document.

### Verbatim-anchor pass, partially complete

Six anchors key on a rulebook gloss or a curated construction rather than source wording: q14, h03,
h04, h05, h09, h10. Only NA1 and NR1 have verified source strings so far; h04 survives NR1's switch
because "amount already credited" appears in both wordings. The remaining rows are recorded in
`meta.pending_source_strings` and were **not** inferred by symmetry — "Accepting P2P chargeback"
makes "Accepting P2M chargeback" an obvious guess for NA2, and page 3 of the OC 184B scan offers
nothing to check a guess against.

---

## What varied, and what did not

M6 varied the chunker with BM25 held constant. M7 varies the retriever and the embedding over the
chunkers M6 left standing, scoring on the same frozen query set with the same metrics —
`src/evaluation/retrieval_metrics.py` is shared by both harnesses, and a test asserts that M7's
`sentence / bm25` cell reproduces M6's `sentence` row exactly.

**21 cells:** 3 chunkers × { BM25, dense × 3 embeddings, hybrid × 3 embeddings }.

### Pruned before running

| Dropped | Why |
| --- | --- |
| `fixed_size` chunker | M6 rule@1 = **0.654**, worst of the four, with no axis where it redeemed that: smaller chunks than `sentence` yet fewer hits at every K, and it missed q15 outright. Dominated, not merely behind. |
| Reranker | Out of this pass by design — a cross-encoder reranks whatever the first stage hands it, so it is only meaningful once the first stage is chosen. The q13/q15 evidence below makes the case for it specific. |

### Embeddings

Three ~22–33M-parameter encoders, all 384-dim. Each is fed the prefixes it was **trained** with —
`query:`/`passage:` for e5, an instruction on the query for bge, none for MiniLM. Note the
asymmetry the table depends on: bge and e5 read **512** tokens, MiniLM reads **256**.

## The grid (retrieval-v1.1, 36 documents)

`trunc` = share of chunks longer than the encoder's window; `seen` = share of the corpus's subword
tokens the encoder actually read. Both in the model's own tokenizer, not whitespace words.

| chunker | retriever | embedding | α | chunks | trunc | seen | R@1 | R@3 | R@5 | MRR | rule@1 | rule@3 | rule@5 | hit tok | build s | idx MB | ms/q |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sentence | bm25 | – | – | 117 | 0.00 | 1.00 | 0.923 | 0.962 | 1.000 | 0.950 | 0.731 | 0.923 | **0.962** | 342 | 0.02 | 0.26 | **0.36** |
| sentence | dense | bge | – | 117 | 0.68 | 0.88 | 0.846 | 0.846 | 0.962 | 0.869 | 0.615 | 0.654 | 0.731 | 333 | 0.83 | 0.18 | 11.23 |
| sentence | hybrid | bge | 0.5 | 117 | 0.68 | 0.88 | 0.923 | 1.000 | 1.000 | 0.955 | 0.769 | 0.962 | **0.962** | 345 | 0.84 | 0.44 | 11.60 |
| sentence | dense | e5 | – | 117 | 0.68 | 0.87 | 0.731 | 0.885 | 0.923 | 0.809 | 0.615 | 0.769 | 0.808 | 345 | 0.89 | 0.18 | 10.96 |
| sentence | hybrid | e5 | 0.5 | 117 | 0.68 | 0.87 | 0.962 | 1.000 | 1.000 | **0.974** | **0.808** | 0.962 | **0.962** | 345 | 0.91 | 0.44 | 10.64 |
| sentence | dense | MiniLM | – | 117 | 0.85 | 0.47 | 0.769 | 1.000 | 1.000 | 0.865 | 0.500 | 0.808 | 0.808 | 338 | 0.31 | 0.18 | 6.56 |
| sentence | hybrid | MiniLM | 0.5 | 117 | 0.85 | 0.47 | 0.923 | 1.000 | 1.000 | 0.962 | 0.731 | 0.885 | **0.962** | 345 | 0.33 | 0.44 | 6.56 |
| structure_aware | bm25 | – | – | 326 | 0.00 | 1.00 | 0.769 | 0.923 | 1.000 | 0.865 | 0.462 | 0.808 | 0.885 | 153 | 0.02 | 0.30 | 0.67 |
| structure_aware | dense | bge | – | 326 | 0.07 | 0.97 | 0.769 | 1.000 | 1.000 | 0.878 | 0.385 | 0.692 | 0.885 | **98** | 1.03 | 0.50 | 12.04 |
| structure_aware | hybrid | bge | 0.5 | 326 | 0.07 | 0.97 | 0.923 | 1.000 | 1.000 | 0.955 | 0.577 | 0.885 | 0.923 | 113 | 1.05 | 0.80 | 11.73 |
| structure_aware | dense | e5 | – | 326 | 0.07 | 0.97 | 0.846 | 0.923 | 1.000 | 0.904 | 0.500 | 0.654 | 0.846 | 117 | 1.17 | 0.50 | 10.97 |
| structure_aware | hybrid | e5 | 0.5 | 326 | 0.07 | 0.97 | 0.923 | 1.000 | 1.000 | 0.962 | 0.538 | 0.846 | 0.885 | 117 | 1.18 | 0.80 | 11.47 |
| structure_aware | dense | MiniLM | – | 326 | 0.25 | 0.76 | 0.808 | 0.923 | 1.000 | 0.881 | 0.462 | 0.654 | 0.769 | 106 | 0.49 | 0.50 | 6.33 |
| structure_aware | hybrid | MiniLM | 0.5 | 326 | 0.25 | 0.76 | 0.885 | 1.000 | 1.000 | 0.936 | 0.462 | 0.885 | 0.923 | 113 | 0.50 | 0.80 | 7.16 |
| whole_document | bm25 | – | – | 36 | 0.00 | 1.00 | 0.846 | 0.962 | 0.962 | 0.904 | 0.808 | 0.962 | 0.962 | 882 | 0.02 | 0.19 | 0.21 |
| whole_document | dense | bge | – | 36 | 0.78 | 0.29 | 0.808 | 0.846 | 0.962 | 0.854 | 0.808 | 0.846 | 0.962 | 850 | 0.28 | **0.06** | 10.20 |
| whole_document | hybrid | bge | 0.5 | 36 | 0.78 | 0.29 | **0.962** | 1.000 | 1.000 | **0.974** | **0.962** | 1.000 | **1.000** | 866 | 0.30 | 0.24 | 10.41 |
| whole_document | dense | e5 | – | 36 | 0.78 | 0.29 | 0.769 | 0.808 | 0.885 | 0.806 | 0.769 | 0.808 | 0.885 | 861 | 0.41 | **0.06** | 10.08 |
| whole_document | hybrid | e5 | 0.5 | 36 | 0.78 | 0.29 | 0.923 | 1.000 | 1.000 | 0.949 | 0.923 | 1.000 | **1.000** | 866 | 0.43 | 0.24 | 10.26 |
| whole_document | dense | MiniLM | – | 36 | 0.94 | 0.15 | 0.731 | 0.885 | 0.923 | 0.804 | 0.692 | 0.846 | 0.923 | 814 | 0.15 | **0.06** | 6.01 |
| whole_document | hybrid | MiniLM | 0.5 | 36 | 0.94 | 0.15 | 0.885 | 1.000 | 1.000 | 0.942 | 0.885 | 1.000 | **1.000** | 866 | 0.17 | 0.24 | 6.34 |

M6 on the same corpus and query set, BM25 held constant: `whole_document` 0.808/0.962/0.962,
`fixed_size` 0.615/0.846/0.923, `sentence` 0.731/0.923/0.962, `structure_aware` 0.462/0.808/0.885
(rule@1/@3/@5).

### What the grid says

1. **Dense alone never beats BM25 on rule-hit-rate, in 9 cells out of 9.** Largest gap is
   `sentence`: 0.962 → 0.731. This corpus is dense with exact identifiers — `RC_1064`, `ND1`,
   `CD1`, `1142` — which is what a lexical index is good at and a 384-dim embedding blurs.
2. **Hybrid ≥ both of its ends at rule@5, in all 9 cells.** The one unanimous result in the grid,
   and the one that does not depend on which 26 queries were written. *It does not hold at every
   K*: `sentence`/hybrid/MiniLM scores rule@3 = 0.885 against BM25's 0.923.
3. **Truncation is severe and mostly does not matter.** `whole_document` + MiniLM reads 15% of the
   corpus and still scores rule@5 = 0.923 dense-only. These circulars front-load their identity,
   so the opening tokens are unusually diagnostic. A property of *this* corpus; it will not
   survive a longer document set, and it is a reason to distrust `whole_document` + dense as a
   design rather than to accept it.
4. **e5 is the strongest embedding on `sentence`, not MiniLM.** rule@1 0.808 vs 0.731, rule@3
   0.962 vs 0.885, for 4 ms more per query. Revision 1 recommended MiniLM on latency; on the
   finer metrics that was wrong, and v1.1 widens the gap.
5. **Index cost is not a constraint at this scale.** Every index is under 1 MB and builds in under
   1.2 s. Recorded because it will matter when the corpus grows past 36 documents.

## The two M6 failures, per cell

**true rank** = where the answering chunk actually sits, from retrieving 50 deep and scoring 5.
A miss at rank 7 and a miss at rank 81 are the same `MISS` and different diagnoses.

### q13 — RGNB, "does not respond" vs "no response"

| chunker | retriever | rule rank | true rank |
| --- | --- | ---: | ---: |
| sentence | bm25 and all dense/hybrid | **1** | 1 |
| structure_aware | bm25 | MISS | **>50** |
| structure_aware | dense bge / e5 / MiniLM | MISS | 8 / 10 / 11 |
| structure_aware | hybrid bge / e5 / MiniLM | MISS | 25 / 27 / 23 |
| whole_document | bm25 | **2** | 2 |
| whole_document | all dense/hybrid | **1** | 1 |

**Fixed at top-5 in 14/21 cells**; 2 more have it inside the top 10.

The M6 diagnosis was wrong in an interesting way: this was never a vocabulary problem dense
retrieval had to solve — `sentence` + BM25 answers it at rank 1. Where it is still unfixed
(`structure_aware`) the cause is the chunker, which splits the RGNB table so the ND1 row and the
"3 calendar days" value land in different chunks. No retriever can fix a chunk that lacks the
answer. Note that **fusion at α=0.5 is worse than dense alone here** (25 vs 8).

### q15 — caps, "how many chargebacks before declined"

| chunker | retriever | rule rank | true rank |
| --- | --- | ---: | ---: |
| sentence | bm25 | MISS | **>50** |
| sentence | dense bge / e5 / MiniLM | MISS | **7 / 7 / 9** |
| sentence | hybrid bge / e5 / MiniLM | MISS | 21 / 26 / 17 |
| structure_aware | bm25 | MISS | >50 |
| structure_aware | dense bge / e5 / MiniLM | MISS | 15 / 9 / 15 |
| whole_document | all retrievers | **1** | 1 |

**Fixed at top-5 in 7/21 cells** — every one a `whole_document` cell; 4 more inside the top 10.

The most useful result in the experiment, and invisible in any average. The question never says
"CD1"; BM25 has no way in and leaves the answering chunk below rank 50. **Dense moves it to rank
7.** The top-5 cutoff is the only reason that is not a fix — and fusion at α=0.5 *undoes* it
(7 → 21). `whole_document` "fixes" q15 only because with 36 chunks there is nothing to rank.

## Fusion-weight sensitivity (bge), rule@5

| chunker | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sentence | 0.962 | 0.962 | **0.962** | 0.923 | 0.769 |
| structure_aware | 0.885 | 0.923 | **0.923** | 0.923 | 0.885 |
| whole_document | 0.962 | **1.000** | **1.000** | **1.000** | 0.962 |

Every chunker holds its best across a **band** of low-to-mid weights and degrades only as α
approaches pure dense. α=0.5 is not a lucky point, and the hybrid result is not a fitting artefact.

## Reading of the top cluster

Best rule@5 = 1.000. One query out of 26 is 0.038, so the cluster is everything within that —
**nine cells**. Ranked on tiebreakers that do **not** depend on which 26 queries were written.

| # | cell | rule@5 | hit tok | ms/q | idx MB | q13 | q15 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | sentence / bm25 | 0.962 | 342 | **0.36** | 0.26 | yes | no |
| 2 | sentence / hybrid / MiniLM / α0.5 | 0.962 | 345 | 6.56 | 0.44 | yes | no |
| 3 | sentence / hybrid / e5 / α0.5 | 0.962 | 345 | 10.64 | 0.44 | yes | no |
| 4 | sentence / hybrid / bge / α0.5 | 0.962 | 345 | 11.60 | 0.44 | yes | no |
| 5 | whole_document / dense / bge | 0.962 | 850 | 10.20 | **0.06** | yes | yes |
| 6 | whole_document / hybrid / MiniLM / α0.5 | **1.000** | 866 | 6.34 | 0.24 | yes | yes |
| 7 | whole_document / hybrid / e5 / α0.5 | **1.000** | 866 | 10.26 | 0.24 | yes | yes |
| 8 | whole_document / hybrid / bge / α0.5 | **1.000** | 866 | 10.41 | 0.24 | yes | yes |
| 9 | whole_document / bm25 | 0.962 | 882 | **0.21** | 0.19 | yes | yes |

Two real candidates, separated by one query:

**A — `sentence` + hybrid + e5, α=0.5.** 345 tokens per hit, 10.6 ms, rule@5 0.962, and the best
rank-1 performance anywhere on `sentence` (rule@1 0.808, rule@3 0.962). Costs **2.5× less context
per answer** than any `whole_document` cell.

**B — `whole_document` + hybrid + bge, α=0.5.** rule@5 1.000, rule@1 0.962, both watched queries
fixed. Costs 866 tokens per hit.

1. **Tokens per hit → A, decisively.** 345 vs 866, paid on every dispute by every downstream
   consumer of retrieval.
2. **Latency → A marginally** (10.6 vs 10.4 ms — a tie in practice). MiniLM is ~4 ms faster in
   both but gives up rule@1 and rule@3; `sentence`/bm25 at 0.36 ms is 30× faster than either and
   worth remembering if retrieval latency ever becomes the constraint.
3. **Index size → B**, 0.24 vs 0.44 MB. Not a real cost at 36 documents.
4. **q13/q15 → B.** But B's q15 "fix" is an artefact of having only 36 chunks to rank, and A's
   q15 answer sits at rank 7 — a cutoff away, not a capability away.
5. **Robustness → A.** B depends on encoders reading 15–29% of each document and on this corpus
   front-loading its identity. Neither generalises, and the corpus is growing.

**Reading: A**, with one caveat that should not be buried. **The K=5 cut is doing more damage than
the retriever choice.** Dense moves both watched answers from below rank 50 to ranks 7–15 and the
cutoff discards it. That is precisely the gap a cross-encoder over the top ~25 closes: it would
convert A's 0.962 to a likely 1.000 without paying B's 866 tokens. Recommend freezing A and
opening the reranker as **M7a**, rather than buying one query with 2.5× the context forever.

## What moved, and why — revision 1 → revision 2

Two input changes, measured separately so neither is credited to the other.

### Corpus reconciliation (35 → 36 documents): 11 of 21 cells moved, in both directions

Adding a 36th document changes BM25's corpus statistics and adds a competing chunk, which on a
36-chunk index is enough to reorder results. Notable moves, all attributable to that and **not** to
better text:

- `whole_document / bm25`: rule@1 0.885 → 0.769, rule@3 and @5 0.923 → 0.962
- `sentence / dense / MiniLM`: rule@5 0.808 → 0.769
- `structure_aware / hybrid / bge` and `/ e5`: rule@5 0.923 → 0.885
- `whole_document / hybrid / MiniLM`: rule@1 0.923 → 0.846
- The three leaders held at rule@5 = 1.000

**The q13 shift, annotated in both directions.** `whole_document` gained q13 at top-5, taking M6's
rule@5 from 0.923 → 0.962. This is **a 36th-document BM25 corpus-statistics reordering on a tiny
index, not a reconciliation improvement** — q13 is an OC 184B/RGNB query and the reconciliation
touched OC 208A only. The *same* reordering **cost** that cell rule@1, 0.885 → 0.769. It is one
reordering with a gain at K=5 and a loss at K=1, and reporting only the gain would misrepresent it.

**A side-effect worth recording for M18.** Correcting `Ilagible` → `Illegible` created a new
lexical competitor for q09, whose anchor is `illegible`: the corrected record now outranks q09's
real target in 3 cells. The correction was right, and it demonstrably made a **non-target** more
retrievable. Cleaning OCR is not metric-neutral.

### Query set v1 → v1.1 (q25 gains a target): 0 regressions, 23 metric improvements

Measured in isolation on the already-reconciled corpus. Of 21 cells × 3 rule-hit metrics: **no
metric regressed**, 23 improved, 6 cells unchanged. q25's rank-1 result was the curated record —
the corrected source — which the frozen v1 query did not list, so the query had been penalising
the system for retrieving the right thing in 11 of 21 configurations.

`hit tok` for `whole_document` fell 914 → 866 as a direct consequence: q25 now scores against the
compact curated record instead of the OCR'd body.

**q19, q21 and q22 were deliberately not amended.** They also name a circular that now has a
curated record, but those records cover a *different section* and do not contain those queries'
anchors (`5 calendar days`, `500`, `3,000`). Listing them would let Recall@K credit retrieval of a
document that provably cannot answer the question. Asserted in
`tests/test_query_freeze.py::test_the_queries_that_were_deliberately_not_amended_stayed_that_way`.

## Provenance

Every `RetrievalHit` carries the `Chunk`, so `doc_id`, `source_section`, `page_start`/`page_end`,
`source_path` and `extraction_method` survive retrieval; `RetrievalHit.citation()` renders them.
Asserted in `tests/test_retrieval.py::test_every_hit_carries_its_provenance`.

The query set carries content hashes over the loaded queries and a changelog naming every amended
query with before/after fingerprints; `tests/test_query_freeze.py` recomputes them, so the set
cannot drift without a test going red.

## Threats to validity

- **26 queries.** One query is 0.038. Nothing should be decided on a margin thinner than that,
  which is why the decision above rests on tiebreakers rather than rule@5.
- **Anchor matching is literal.** A chunk that paraphrases the answer perfectly scores zero. This
  makes rule-hit-rate conservative and biased toward chunks quoting the circular verbatim.
- **36 documents.** Recall@K is generous at this scale, index size is not yet informative, and —
  as the reconciliation showed — adding a single document can reorder a whole column.
- **The truncation result is corpus-specific.** See grid finding 3.
- **Latency is single-query, warm, batch-of-one, on one GPU.** Fine for ranking cells against each
  other; not a serving benchmark.

## Not done in this pass

- **Reranker.** See the recommendation — the case is now specific and quantified.
- **nDCG.** The master plan lists it; the query set has binary targets and anchors, no graded
  relevance, so nDCG would be MRR with extra steps. Needs a graded query set first.
- **Downstream check** (does the retrieved rule let the system pick the correct required-evidence
  set). Needs the L0 sufficiency engine wired to retrieval output; belongs with M3's vertical
  slice.
- **Hard-query expansion.** 26 queries is too few for the margins being compared; ~25 new
  near-duplicate queries are drafted and awaiting review.
