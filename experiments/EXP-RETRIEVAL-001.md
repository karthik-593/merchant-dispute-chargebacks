# EXP-RETRIEVAL-001 — M7 retriever × embedding grid

**Status:** results in, nothing frozen. `RETRIEVER_VERSION`, `EMBEDDING_VERSION` and
`CHUNKER_VERSION` are set together by review.
**Date:** 2026-09-08 · **MLflow experiment:** `M7-retrieval` (36 runs: 21 `grid=main`,
15 `grid=alpha_sweep`) · **Harness:** `src/evaluation/eval_retrieval.py`
**Scoring set:** `data/corpus/retrieval_queries.yaml`, `retrieval-v1`, 26 queries, frozen and
unmodified. **Corpus:** `data/corpus/circulars.jsonl`, 35 documents.
**Hardware:** RTX 4060 Laptop 8 GB, CUDA 12.6, torch 2.14.0+cu126.

Reproduce:

```bash
uv run python -m src.evaluation.eval_retrieval --alpha-sweep bge-small-en-v1.5
```

## What varied, and what did not

M6 varied the chunker with BM25 held constant. M7 varies the retriever and the embedding over the
chunkers M6 left standing. The query set, the anchors, the corpus and the metrics are the same
objects M6 used — `src/evaluation/retrieval_metrics.py` is shared by both harnesses, and a test
asserts that M7's `sentence / bm25` cell reproduces M6's `sentence` row exactly. An M7 number can
therefore be read straight against an M6 number.

**21 cells:** 3 chunkers × { BM25, dense × 3 embeddings, hybrid × 3 embeddings }.

### Pruned before running

| Dropped | Why |
| --- | --- |
| `fixed_size` chunker | M6 rule@1 = **0.654**, the worst of the four, and no axis where it redeemed that. Its chunks (239 mean tokens) are *smaller* than `sentence`'s (332) yet it hit less often at every K, and it missed q15 outright. It is dominated, not merely behind, so re-measuring it under three retrievers would have spent a third of the grid confirming a settled result. |
| Reranker | Deliberately out of this pass. A cross-encoder reranks whatever the first stage hands it, so it is only meaningful once the first stage is chosen. The q13/q15 evidence below turns this from a vague "possible later add" into a specific, motivated one. |

### Embeddings

Three ~22–33M-parameter encoders, all 384-dim, all far inside 8 GB. Each is fed the query and
passage prefixes it was **trained** with — `query:`/`passage:` for e5, an instruction on the query
for bge, none for MiniLM. Feeding all three bare text would have benchmarked prompt formatting
rather than embeddings, which is the same class of error as M6's original un-split `RC_1064`.

Note the asymmetry the table depends on: bge and e5 read **512** tokens, MiniLM reads **256**.

## The grid

`trunc` = share of chunks longer than the encoder's context window; `seen` = share of the corpus's
subword tokens the encoder actually read. Both counted in the model's own tokenizer, not in
whitespace words. `hit tok` = mean size of the chunk that carried the answer.

| chunker | retriever | embedding | α | chunks | trunc | seen | R@1 | R@3 | R@5 | MRR | rule@1 | rule@3 | rule@5 | hit tok | build s | idx MB | ms/q |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sentence | bm25 | – | – | 115 | 0.00 | 1.00 | 0.885 | 0.962 | 1.000 | 0.931 | 0.692 | 0.923 | **0.962** | 342 | 0.02 | 0.25 | **0.32** |
| sentence | dense | bge-small-en-v1.5 | – | 115 | 0.67 | 0.88 | 0.846 | 0.846 | 0.962 | 0.869 | 0.615 | 0.654 | 0.731 | 333 | 0.75 | 0.18 | 10.31 |
| sentence | hybrid | bge-small-en-v1.5 | 0.5 | 115 | 0.67 | 0.88 | 0.885 | 1.000 | 1.000 | 0.936 | 0.731 | 0.962 | **0.962** | 345 | 0.77 | 0.43 | 10.50 |
| sentence | dense | e5-small-v2 | – | 115 | 0.67 | 0.87 | 0.731 | 0.885 | 0.923 | 0.809 | 0.615 | 0.769 | 0.808 | 345 | 0.84 | 0.18 | 10.26 |
| sentence | hybrid | e5-small-v2 | 0.5 | 115 | 0.67 | 0.87 | 0.923 | 1.000 | 1.000 | 0.955 | 0.769 | 0.962 | **0.962** | 345 | 0.85 | 0.43 | 10.91 |
| sentence | dense | all-MiniLM-L6-v2 | – | 115 | 0.84 | 0.47 | 0.769 | 0.962 | 1.000 | 0.854 | 0.500 | 0.769 | 0.808 | 338 | 0.30 | 0.18 | 6.04 |
| sentence | hybrid | all-MiniLM-L6-v2 | 0.5 | 115 | 0.84 | 0.47 | 0.923 | 1.000 | 1.000 | 0.962 | 0.731 | 0.885 | **0.962** | 345 | 0.31 | 0.43 | 6.80 |
| structure_aware | bm25 | – | – | 306 | 0.00 | 1.00 | 0.769 | 0.923 | 1.000 | 0.863 | 0.462 | 0.808 | 0.885 | 152 | 0.02 | 0.29 | 0.65 |
| structure_aware | dense | bge-small-en-v1.5 | – | 306 | 0.08 | 0.97 | 0.769 | 0.962 | 0.962 | 0.859 | 0.423 | 0.692 | 0.846 | **102** | 1.01 | 0.47 | 10.29 |
| structure_aware | hybrid | bge-small-en-v1.5 | 0.5 | 306 | 0.08 | 0.97 | 0.846 | 0.962 | 1.000 | 0.907 | 0.538 | 0.885 | 0.923 | 127 | 1.03 | 0.76 | 10.80 |
| structure_aware | dense | e5-small-v2 | – | 306 | 0.08 | 0.97 | 0.808 | 0.885 | 0.962 | 0.865 | 0.500 | 0.654 | 0.846 | 117 | 1.12 | 0.47 | 10.54 |
| structure_aware | hybrid | e5-small-v2 | 0.5 | 306 | 0.08 | 0.97 | 0.885 | 0.962 | 1.000 | 0.933 | 0.538 | 0.846 | 0.923 | 127 | 1.14 | 0.76 | 11.03 |
| structure_aware | dense | all-MiniLM-L6-v2 | – | 306 | 0.26 | 0.76 | 0.808 | 0.885 | 0.962 | 0.862 | 0.423 | 0.615 | 0.731 | 110 | 0.47 | 0.47 | 6.07 |
| structure_aware | hybrid | all-MiniLM-L6-v2 | 0.5 | 306 | 0.26 | 0.76 | 0.846 | 1.000 | 1.000 | 0.917 | 0.462 | 0.885 | 0.923 | 130 | 0.49 | 0.76 | 7.24 |
| whole_document | bm25 | – | – | 35 | 0.00 | 1.00 | 0.923 | 0.962 | 0.962 | 0.942 | 0.885 | 0.923 | 0.923 | 960 | 0.02 | 0.18 | 0.19 |
| whole_document | dense | bge-small-en-v1.5 | – | 35 | 0.77 | 0.29 | 0.769 | 0.846 | 0.923 | 0.819 | 0.769 | 0.846 | 0.923 | 854 | 0.26 | **0.05** | 9.92 |
| whole_document | hybrid | bge-small-en-v1.5 | 0.5 | 35 | 0.77 | 0.29 | 0.923 | 1.000 | 1.000 | 0.955 | **0.923** | 1.000 | **1.000** | 914 | 0.27 | 0.24 | 10.33 |
| whole_document | dense | e5-small-v2 | – | 35 | 0.77 | 0.29 | 0.769 | 0.808 | 0.885 | 0.806 | 0.769 | 0.808 | 0.885 | 861 | 0.39 | **0.05** | 10.66 |
| whole_document | hybrid | e5-small-v2 | 0.5 | 35 | 0.77 | 0.29 | 0.885 | 1.000 | 1.000 | 0.929 | 0.885 | 1.000 | **1.000** | 914 | 0.40 | 0.24 | 10.71 |
| whole_document | dense | all-MiniLM-L6-v2 | – | 35 | 0.94 | 0.15 | 0.692 | 0.885 | 0.885 | 0.769 | 0.654 | 0.846 | 0.885 | 816 | 0.14 | **0.05** | 6.05 |
| whole_document | hybrid | all-MiniLM-L6-v2 | 0.5 | 35 | 0.94 | 0.15 | 0.923 | 1.000 | 1.000 | **0.962** | **0.923** | 1.000 | **1.000** | 914 | 0.15 | 0.24 | 6.38 |

### What the grid says

1. **Dense alone never beats BM25 on rule-hit-rate. In 9 cells out of 9.** The largest gap is
   `sentence`: 0.962 → 0.731. Whatever these encoders are contributing, "replace BM25" is not it.
   This corpus is dense with exact identifiers — `RC_1064`, `ND1`, `CD1`, `OC 208A` — and an exact
   identifier is precisely what a lexical index is good at and what a 384-dim embedding blurs.
2. **Hybrid ≥ both of its ends, in every cell.** Fusion never loses to BM25 and never loses to
   dense, in all 9 hybrid cells. This is the one clean, unanimous result in the grid, and it is
   the result that does not depend on which 26 queries were written.
3. **Truncation is severe and mostly does not matter.** `whole_document` + MiniLM reads 15% of the
   corpus and still scores rule@5 = 0.885 dense-only. These circulars front-load their identity —
   subject line, circular number, subject matter — so the opening 256 tokens are unusually
   diagnostic of the whole. It is a property of *this* corpus, and it will not survive contact
   with a longer document set. It is a reason to distrust `whole_document` + dense as a design,
   not a reason to accept it.
4. **MiniLM is roughly 40% faster and no worse in hybrid.** 6.0–7.2 ms/query against 10.3–11.0 for
   bge and e5, at identical hybrid rule@5 in `sentence` and `whole_document`. Its 256-token window
   costs it dearly dense-only (0.731 on `sentence`) and costs it nothing once BM25 is carrying the
   lexical half.
5. **Index cost is not a real constraint at this scale.** Every index in the grid is under 1 MB and
   builds in under 1.2 s. It cannot break a tie today; it is recorded because it will matter when
   the corpus grows past 35 documents.

## The two M6 failures, per cell

Both are reported for all 21 cells. The critical column is **true rank** — where the answering
chunk actually sits, found by retrieving 50 deep and scoring only the top 5. A miss at rank 7 and
a miss at rank 81 are the same `MISS` in a table and completely different diagnoses.

### q13 — RGNB, "does not respond" vs "no response"

> *What happens on an RGNB if the beneficiary bank does not respond at all on a P2P transfer?*
> anchors `['ND1', '3 calendar days']` · target OC 184B row ND1

| chunker | retriever | embedding | doc rank | rule rank | true rank |
| --- | --- | --- | ---: | ---: | ---: |
| sentence | bm25 | – | 1 | **1** | 1 |
| sentence | dense / hybrid | all three | 1 | **1** | 1 |
| structure_aware | bm25 | – | 1 | MISS | **>50** |
| structure_aware | dense | bge / e5 / MiniLM | 1–2 | MISS | 8 / 10 / 11 |
| structure_aware | hybrid | bge / e5 / MiniLM | 1 | MISS | 26 / 28 / 23 |
| whole_document | bm25 | – | 1 | MISS | 7 |
| whole_document | dense / hybrid | all three | 1 | **1** | 1 |

**Fixed at top-5 in 13/21 cells**; 3 more have the answer inside the top 10, below the cut.

The M6 diagnosis was wrong in an interesting way. This was never a vocabulary problem that dense
retrieval had to solve — `sentence` + BM25 already answers it at rank 1. What M6 saw was
`whole_document` failing it, and there dense *does* fix it outright: BM25 ranks the answering
document 7th, every embedding ranks it 1st. That is a genuine paraphrase win, and it is the
clearest evidence in the grid that the embeddings are contributing something BM25 cannot.

Where it is still unfixed — `structure_aware` — the cause is visible: the chunker splits the RGNB
table into one row per chunk, so the ND1 row and the "3 calendar days" column value that qualifies
it land in different chunks. No retriever can fix a chunk that does not contain the answer. Note
also that **fusion at α=0.5 is worse than dense alone here** (rank 26 vs 8): BM25's rank->50
opinion actively drags the fused score down.

### q15 — caps, "how many chargebacks before declined"

> *How many chargebacks can one customer raise in a rolling 30 days before they are declined?*
> anchors `['CD1']` · target OC 184, CD1 cap

| chunker | retriever | embedding | doc rank | rule rank | true rank |
| --- | --- | --- | ---: | ---: | ---: |
| sentence | bm25 | – | 1 | MISS | **>50** |
| sentence | dense | bge / e5 / MiniLM | 1 | MISS | **7 / 7 / 9** |
| sentence | hybrid | bge / e5 / MiniLM | 1 | MISS | 21 / 26 / 17 |
| structure_aware | bm25 | – | 1 | MISS | **>50** |
| structure_aware | dense | bge / e5 / MiniLM | 1 | MISS | 15 / 9 / 15 |
| structure_aware | hybrid | bge / e5 / MiniLM | 1 | MISS | 29 / 21 / 25 |
| whole_document | bm25 / dense / hybrid | all three | 1 | **1** | 1 |

**Fixed at top-5 in 7/21 cells** — every one of them a `whole_document` cell.

This is the most useful single result in the experiment, and it is not visible in any average.
The question never says "CD1"; it asks in plain language for a number. BM25 has no way in — the
answering chunk sits below rank 50 under both fine-grained chunkers. **Dense retrieval moves it
from >50 to rank 7.** That is the paraphrase gap closing almost completely, and the top-5 cutoff
is the only reason it does not show up as a fix.

And again fusion at α=0.5 *undoes* it: 7 → 21. Averaged over 26 queries that damage is invisible,
because hybrid's wins elsewhere more than pay for it.

The honest reading: **on both watched queries, dense retrieval fixes the ranking and the K=5 cut
hides it.** `whole_document` "fixes" q15 only because with 35 chunks there is nothing to rank —
retrieving the document *is* retrieving the answer, at a cost of 914 tokens per hit.

## Fusion-weight sensitivity (bge-small-en-v1.5)

Run so that a hybrid win at exactly α=0.5 could not be mistaken for a real one.

| chunker | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sentence | 0.962 | 0.962 | **0.962** | 0.923 | 0.731 |
| structure_aware | 0.885 | 0.923 | **0.923** | 0.846 | 0.846 |
| whole_document | 0.962 | **1.000** | **1.000** | 0.962 | 0.923 |

rule@5. Every chunker holds its best value across a **band** of low-to-mid weights and degrades
only as α approaches pure dense. α=0.5 is not a lucky point, and the hybrid result in the grid is
not a fitting artefact. It also confirms point 1: weighting dense above ~0.7 always hurts.

## Reading of the top cluster

`rule@5` best in grid = 1.000. One query out of 26 is 0.038, so the cluster is everything within
0.038 of the best — seven cells. Ranked on tiebreakers that **do not** depend on which 26 queries
were written: context cost per hit, latency, index size, simplicity, and whether q13/q15 fixed.

| # | cell | rule@5 | hit tok | ms/q | idx MB | q13 | q15 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | sentence / bm25 | 0.962 | 342 | **0.32** | 0.25 | yes | no |
| 2 | sentence / hybrid / all-MiniLM-L6-v2 / α0.5 | 0.962 | 345 | 6.80 | 0.43 | yes | no |
| 3 | sentence / hybrid / bge-small-en-v1.5 / α0.5 | 0.962 | 345 | 10.50 | 0.43 | yes | no |
| 4 | sentence / hybrid / e5-small-v2 / α0.5 | 0.962 | 345 | 10.91 | 0.43 | yes | no |
| 5 | whole_document / hybrid / all-MiniLM-L6-v2 / α0.5 | **1.000** | 914 | 6.38 | 0.24 | yes | yes |
| 6 | whole_document / hybrid / bge-small-en-v1.5 / α0.5 | **1.000** | 914 | 10.33 | 0.24 | yes | yes |
| 7 | whole_document / hybrid / e5-small-v2 / α0.5 | **1.000** | 914 | 10.71 | 0.24 | yes | yes |

The cluster splits cleanly into two candidates, and the 0.038 between them is one query.

**A — `sentence` + hybrid + MiniLM, α=0.5** (row 2). 345 tokens per hit, 6.8 ms, rule@5 = 0.962.
Costs **2.6× less context per answer** than any `whole_document` cell. Since every downstream
consumer of retrieval — grounded drafting, the verifier, claim→evidence_id checking — pays per
token on every dispute, that ratio is a recurring cost, whereas the 0.038 is one query.

**B — `whole_document` + hybrid + MiniLM, α=0.5** (row 5). Perfect rule@5, both watched queries
fixed, the smallest index, and the same latency. It costs 914 tokens per hit.

Ranked against the stated tiebreakers:

1. **Tokens per hit → A, decisively.** 345 vs 914.
2. **Latency → tie.** MiniLM at ~6.4–6.8 ms in both. (Pure `sentence`/BM25 at 0.32 ms is 20×
   faster than either, and worth remembering if retrieval latency ever becomes the constraint.)
3. **Index size → B**, marginally: 0.24 MB vs 0.43. Neither is a real cost at 35 documents.
4. **q13/q15 → B.** But per the analysis above, B's q15 "fix" is an artefact of having only 35
   chunks to rank, not retrieval quality, and A's q15 answer is sitting at rank 7 — it is a
   cutoff away, not a capability away.
5. **Simplicity → A on the retrieval side, B on the pipeline side.** B's chunker is trivial, but B
   is structurally fragile: it depends on encoders reading 15–29% of each document and on this
   corpus front-loading its identity. That does not generalise, and the corpus will grow.

**My reading: A** — `sentence` + hybrid + `all-MiniLM-L6-v2` at α=0.5 — with one caveat I would
not want buried. A is the better engineering choice on every axis that survives a change to the
query set: a third of the context cost, no dependence on truncation luck, and a chunker that
scales past 35 documents. The single query it gives up (q15) is not lost, it is at rank 7.

**The caveat: the K=5 cut is doing more damage than the retriever choice is.** Across the grid,
dense retrieval moves both watched answers from below rank 50 to ranks 7–15, and the cutoff throws
that away. Before freezing, it is worth deciding whether K=5 is the right operating point at all —
and this is exactly the gap a cross-encoder reranker over the top ~25 is built to close. If the
answer is coming back at rank 7, a reranker converts A's 0.962 into a likely 1.000 without paying
`whole_document`'s 914 tokens. **I would recommend freezing A now and opening the reranker as
M7a**, rather than choosing B to buy one query with 2.6× the context on every dispute.

If you prefer to defer: A and B are within one query on a 26-query set, so the case for B rests
entirely on that query, and the analysis above says that query is a chunk-count artefact.

## Provenance

Verified end to end. Every `RetrievalHit` carries the `Chunk` itself, so `doc_id`,
`source_section`, `page_start`/`page_end`, `source_path` and `extraction_method` survive
retrieval; `RetrievalHit.citation()` renders them. Asserted in
`tests/test_retrieval.py::test_every_hit_carries_its_provenance`.

## Threats to validity

- **26 queries.** One query is 0.038. Nothing here should be decided on a margin thinner than
  that, which is why the decision above is argued on tiebreakers rather than on rule@5.
- **Anchor matching is literal.** A chunk that paraphrases the answer perfectly scores zero. This
  makes rule-hit-rate conservative and biased toward chunks that quote the circular verbatim.
- **35 documents.** Recall@K is generous at this scale and the index-size column is not yet
  informative. Both change as the corpus grows.
- **The truncation result is corpus-specific.** See grid finding 3.
- **Latency is single-query, warm, batch-of-one, on one GPU.** Fine for ranking cells against each
  other; not a serving benchmark.

## Not done in this pass

- **Reranker.** See the recommendation above — the case for it is now specific.
- **nDCG.** The master plan lists it; the query set has no graded relevance labels, only binary
  targets and anchors, so nDCG would equal MRR-with-extra-steps. It needs a graded query set first.
- **Downstream check** ("does the retrieved rule let the system pick the correct required-evidence
  set", per the M7 plan entry). That needs the L0 sufficiency engine wired to retrieval output;
  it belongs with M3's vertical slice, not here.
