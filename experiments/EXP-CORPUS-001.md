# EXP-CORPUS-001 — the M7 corpus pointer drift: diagnosis, adjudication, and the guard

**Status:** resolved
**Milestone:** M7 freeze integrity (bookkeeping only — no corpus content changed, no result changed)
**Date:** 2026-09-13
**Incident window:** 2026-09-09 → 2026-09-13
**Artifact:** `data/corpus/circulars.jsonl`, corpus-v1.4
**H_true:** `34a085ab04488672b9d8d79a582655ba`, 551,876 B

---

## 1. What was wrong

`configs/versions.yaml` stated that every M7 number was measured on corpus-v1.4.
`data/corpus/circulars.jsonl.dvc` pointed at a **different, older** artifact. The working copy was
a **third** hash. So `dvc checkout` from a clean clone would have restored a corpus that cannot
reproduce the M7 freeze, while the freeze went on claiming it could.

Both files were internally consistent. Both were committed. Nothing compared them.

---

## 2. Timeline, as found

The pointer was last updated at **`0f2555d`** (2026-09-08), which is **corpus-v1.1**. Three
content versions then landed without touching it:

| commit | change | pointer updated |
|---|---|---|
| `8164310` | corpus-v1.2 — verdict band out of per-row text | **no** ← **drift originates here** |
| `25e21b6` | corpus-v1.3 — verbatim OC 184B RGNB wording | **no** |
| `27d07f6` | corpus-v1.4 — metadata swept out of row text | **no** — *inherited* the drift |

The pointer was **three versions stale**, not one. `27d07f6` is the last commit to carry the
problem forward, not its cause — worth stating because the obvious reading blames the commit that
happened to bump the version label.

**The authentic v1.4 was one command from being lost.** DVC cache mtimes show every other corpus
artifact entering the cache on 2026-09-07/08; `34a085ab` entered at **2026-09-09 21:35:05**, which
was an *accidental* `dvc add` during unrelated M8-pre work (immediately reverted at the pointer
level, but the cache write stood). Before that accident corpus-v1.4 existed **only as an
uncommitted working file**. A `dvc checkout` at any point in the preceding four days would have
silently overwritten it with the v1.1 corpus, and v1.4 would have survived only if a rebuild
happened to be byte-reproducible.

---

## 3. Adjudication — and why MLflow could not do it

**MLflow recorded no corpus hash. Not on the freeze run, not on any run.** Checked exhaustively
across 349 runs in experiments 1–3:

- 38 distinct param keys across all M7 runs; **none** mentioning hash/sha/md5/digest/fingerprint.
- No non-MLflow tags. No hash-bearing metrics.
- **No run logged the corpus as an artifact.** The freeze run `267bf5c8…` carries only
  `freeze_report.txt` and `selection_table.txt`; neither contains a hash.
- `configs/corpus/corpus_version.yaml` recorded no hashes either.

`freeze_m7.py` logged `corpus_version` straight out of `versions.yaml` — so the run asserted the
label the config asserted, and **neither was checkable against any bytes**. The freeze was
self-certifying. That is the root cause, and it is fixed in §5.

**H_true was established instead by metric reproduction.** `versions.yaml` records the frozen
score — `rule-hit-rate@5 = 0.640 over 50 discriminators` — and that functions as a fingerprint.
Re-running the frozen cell (structure_aware row-grain / dense / bge-small-en-v1.5, no reranker)
against each live candidate:

```
CURRENT POINTER a4ed9bf : rule_hit@5 = 0.600   does NOT reproduce
WORKING COPY   34a085a : rule_hit@5 = 0.640   EXACT MATCH
```

**Corroborated independently** by the v1.4 entry's own acceptance criteria — *"0 of 50 row chunks
contaminated"*, *"ND2 back to 27 tokens from 113"*. Applying the frozen chunker to every candidate:

| md5 | bytes | origin | docs | row chunks | contaminated | ND2 tok |
|---|---:|---|---:|---:|---:|---:|
| `3991ac7f` | 526,686 | pointer @ `527f121` | 33 | 1 | 0 | – |
| `985fc345` | 526,682 | orphan — cached, never in any pointer | 33 | 1 | 0 | – |
| `bd3685c4` | 544,013 | pointer @ `3e2c957` | 35 | 22 | 13 | 38 |
| `a4ed9bf4` | 554,913 | **stale pointer** @ `0f2555d` (v1.1) | 36 | 50 | **41** | 38 |
| `34a085ab` | 551,876 | **working copy** | 36 | 50 | **0** | **27** |

Only `34a085ab` satisfies them. Two independent adjudicators, same answer. **Case B**: the
authentic v1.4 was the working copy and had never been deliberately committed to DVC.

---

## 4. Resolution

1. `dvc add data/corpus/circulars.jsonl` — pointer stamped to `34a085ab`. Cache already held the
   artifact, so this was a **pointer write only; no content changed**.
2. Verified by destroying the working copy and running `dvc checkout`: restored
   `34a085ab04488672b9d8d79a582655ba`, 551,876 B. **A clean clone can now restore the corpus M7
   was frozen on.**
3. `content_md5` + `content_bytes` recorded on the corpus-v1.4 entry in
   `configs/corpus/corpus_version.yaml`, which owns corpus identity; `configs/versions.yaml`
   restates it as `corpus_content_md5` and points at that record.

**All four now name one artifact:**

```
.dvc pointer                      34a085ab04488672b9d8d79a582655ba
corpus_version.yaml content_md5   34a085ab04488672b9d8d79a582655ba
versions.yaml corpus_content_md5  34a085ab04488672b9d8d79a582655ba
file on disk                      34a085ab04488672b9d8d79a582655ba
```

### Provenance of the recorded hash — read this before trusting it

> `content_md5` for corpus-v1.4 was **reconstructed post-hoc, not logged at freeze time**. No
> corpus hash was ever recorded in MLflow (`freeze_m7.py` self-certified the version label).
> H_true was established by **metric reproduction** — the frozen `rule_hit@5 = 0.640` reproduces
> on `34a085ab` and **not** on the stale pointer `a4ed9bf`, which gives 0.600 — and corroborated
> by the v1.4 content criteria. **The hash's authority traces to the metric reproduction, not to
> a live capture.**

---

## 5. Root cause fixed: a freeze now records what it ran on

`freeze_m7.py` hashes every versioned input **at run time** and logs it as MLflow params
(`input_md5__*`) **and** into `freeze_report.txt`: corpus, query set, M8 eval set, rulebook
evidence map, and both version records.

The freeze was re-logged. The contrast is the whole point:

| run | when | corpus_version | input hashes |
|---|---|---|---|
| `267bf5c8…` original | 2026-09-09 14:38 | corpus-v1.4 | **0** — self-certifying, unadjudicable |
| `b4dd5bfe…` re-logged | 2026-09-13 11:28 | corpus-v1.4 | **6** — corpus = `34a085ab…` |

The original run is kept. It is the evidence for why this happened.

**Requirement carried forward to M8 Block 1:** the BASE_LLM freeze harness must log content hashes
of its frozen inputs — dataset splits, prompt, task definitions, output schema, evaluator — from
the start. M8's standing rules already say that if any fixed input changes after a model has run,
every prior model must be re-run; that rule is unenforceable if the freeze records only labels,
because nobody can tell afterwards whether an input moved. Do not rediscover this.

---

## 6. The guard

`src/evaluation/version_integrity.py` + `tests/test_version_integrity.py` (10 tests, all passing).
Also runnable as a CI check: `python -m src.evaluation.version_integrity` exits non-zero on drift.

It asserts the recorded `content_md5`, the `.dvc` pointer, `versions.yaml`, and (optionally) the
file on disk all name the same artifact.

**Negative controls**, so the guard is not vacuous — the same discipline as the leak-probe
controls in `tests/test_leak_probes.py`. Each rebuilds a drift in a temp tree and asserts the
guard **fails**:

- the exact incident: pointer at `a4ed9bf` while the records say `34a085ab`
- the pre-incident state: a version label with no hash behind it
- `versions.yaml` naming a different hash than the corpus record
- `versions.yaml` naming a different version than the corpus record
- a corpus on disk that is not the recorded artifact

**What the guard deliberately does NOT establish.** It checks *consistency* between records. It
cannot check *correctness*: a wrong hash written into all three places passes, and there is a test
(`test_guard_is_not_self_referential`) that pins exactly that and documents why. The hash's
correctness is anchored by §3's metric reproduction, not by the guard. A guard that treated "a
hash is recorded" as proof the hash is right would be self-referential and would have passed on
the very drift it exists to catch.

---

## 7. Cost to results: **none**

The corpus **content** was correct throughout — `34a085ab` was the right bytes on disk the whole
time, and every M7 measurement was taken on it. What was broken was only the bookkeeping that
lets DVC *restore* it.

- **No M7 number changes.** The frozen values stand exactly as recorded.
- **No re-run needed.** The freeze was measured on the right corpus; it simply could not be
  reconstructed from a clean clone until now.
- **The reproduction re-run in §3 confirms it**, and is itself the evidence: 0.640 on the frozen
  cell, matching `versions.yaml` to three decimals.

The exposure was real but latent — the risk was that a `dvc checkout` would have destroyed the
only copy of the authentic corpus, not that any published number was ever wrong.
