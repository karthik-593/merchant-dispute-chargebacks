# EXP-RETRIEVAL-001 — M7 retriever × embedding grid

**Status:** results in, nothing frozen. `RETRIEVER_VERSION`, `EMBEDDING_VERSION` and
`CHUNKER_VERSION` are set together by review.
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
