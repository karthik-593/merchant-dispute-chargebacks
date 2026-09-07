# Corpus card — NPCI/RBI circulars

## What this is

The **retrieval target**. These are the source circulars the rulebook is derived from, normalised into clean per-document text with page boundaries intact so a chunk can cite `doc_id` plus page.

**This is not the evidence store.** Finding the rule that governs a dispute is a search problem over this corpus; finding a dispute's own artifacts is an exact keyed join, handled by `src/data/evidence_store.py` against the datasets. The two are kept strictly apart — a similarity score is the wrong tool for a question that has one right answer.

Nothing here is chunked or embedded yet. That is later work.

## Extraction

- **Records**: 35 over 33 source circulars
- **Pages**: 143 (58 recognised by OCR)
- **Characters**: 247,519
- **OCR engine**: tesseract 5.4.0.20240606 at C:\Program Files\Tesseract-OCR\tesseract.exe
- **Rasterisation**: 300 dpi, via PyMuPDF (no poppler dependency)
- **Text-layer threshold**: a page yielding under 120 characters is treated as having no text layer and is sent to OCR

Most of these circulars are image scans, so extraction is two-stage and per page: try the PDF text layer first, rasterise and OCR whatever it cannot read. The method is recorded per page and rolled up per document, because whether a quoted rule came from a text layer or from a character recogniser changes how much it should be trusted.

### Text vs OCR

| method | documents | share | meaning |
|---|---:|---:|---|
| `text` | 9 | 25.7% | every page had a usable text layer |
| `ocr` | 22 | 62.9% | image scan; every page was recognised |
| `mixed` | 2 | 5.7% | text body with scanned pages (typically a cover or an annexure) |
| `curated` | 2 | 5.7% | a table rebuilt from its verified structured form, because OCR cannot render it usefully |

Records cover 33 source circulars: 33 extracted bodies plus 2 curated table(s) attached to circulars that also have a body record.

### Curated tables

OCR flattens a table into running text, so the OC 208 §C evidence map and the OC 184B RGNB response table both lose the association between a row and its columns. A chunk of either reads as a stream of codes with no reliable link between a reason code and the evidence that answers it — which retrieves as though it were an answer while being unusable. Both are therefore also carried as curated records, rebuilt from the verified structured form held in `configs/rulebook/`, so retrieval gets clean rows.

**The OCR'd bodies are kept.** Nothing is deleted: the flattened table text is still in the corpus and still cites its page. It is simply superseded for retrieval by the curated record, which names what it replaces in `supersedes_doc_id` so the relationship is machine-readable rather than a note in prose.

| curated record | section | supersedes |
|---|---|---|
| `oc_184b_rgnb_response_table_curated` | RGNB response table | `upi_oc_no_184_b_fy_2025_26_addendum_to_oc_184_modification_in_upi_chargeback_rul` |
| `oc_208_sec_c_evidence_map_curated` | §C (Type of Evidence) | `upi_oc_no_208_fy_24_25_implementation_of_nrp_prd_process_arbitration_guidelines` |

Only these two are curated, and only because they are verified against source. Every other table stays as OCR'd text until someone checks it: a curated record asserts that a human confirmed it, and producing them casually would empty the label of meaning.

## Per-document yield

| method | ref (parsed from content) | pages | chars | chars/page | doc_id |
|---|---|---:|---:|---:|---|
| `curated` | — | 1 | 5,313 | 5,313 | `oc_208_sec_c_evidence_map_curated` |
| `curated` | — | 1 | 1,745 | 1,745 | `oc_184b_rgnb_response_table_curated` |
| `mixed` | — | 28 | 38,817 | 1,386 | `osdt31012019` |
| `mixed` | — | 4 | 4,086 | 1,021 | `upi_settlement_process_256f73e1df` |
| `ocr` | — | 8 | 16,944 | 2,118 | `upi_oc_no_208_fy_24_25_implementation_of_nrp_prd_process_arbitration_guidelines` |
| `ocr` | OC 206 | 6 | 15,129 | 2,521 | `upi_oc_no_206_fy_24_25_implementation_of_generic_good_faith_debit_credit_adjustm` |
| `ocr` | OC 208 | 6 | 11,824 | 1,970 | `upi_oc_no_208_a_fy_2025_26_addendum_to_oc_208_implementation_of_nrp_prd_process_` |
| `ocr` | OC 208C | 6 | 11,141 | 1,856 | `upi_oc_no_208c_fy_2026_27_addendum_to_oc_208_208a_208b_implementation_of_nrp_prd` |
| `ocr` | OC 184 | 3 | 5,694 | 1,898 | `upi_oc_no_184_b_fy_2025_26_addendum_to_oc_184_modification_in_upi_chargeback_rul` |
| `ocr` | OC 184 | 3 | 5,448 | 1,816 | `upi_oc_no_184_fy_23_24_modification_in_upi_chargeback_rules_and_procedures_bdfd9` |
| `ocr` | OC 152 | 2 | 3,860 | 1,930 | `upi_oc_157_upi_raw_data_version_3_0_1edb076c72` |
| `ocr` | OC 741 | 2 | 3,643 | 1,821 | `oc_no_141_7e958ee3b1` |
| `ocr` | OC 165 | 2 | 3,446 | 1,723 | `upi_oc_165_fy_23_24_implementation_of_advanced_refund_api_as_part_of_upi_help_ud` |
| `ocr` | — | 1 | 2,935 | 2,935 | `upi_oc_193_fy_23_24_non_compliance_to_txnid_b76ac4cdb9` |
| `ocr` | OC 197 | 2 | 2,904 | 1,452 | `upi_oc_no_197_fy_24_25_implementation_of_10_settlement_cycles_and_revised_busine` |
| `ocr` | — | 2 | 2,884 | 1,442 | `upi_oc_no_222_fy_2025_26_segregation_of_upi_settlement_cycles_for_auth_and_dispu` |
| `ocr` | OC 198 | 2 | 2,841 | 1,420 | `upi_oc_no_198_fy_24_25_revision_of_disputes_tat_7863285dd6` |
| `ocr` | — | 2 | 2,694 | 1,347 | `upi_oc_no_235_fy_2026_27_revision_of_turnaround_time_tat_for_responding_to_fraud` |
| `ocr` | OC 39 | 1 | 2,535 | 2,535 | `upi_oc_no_39_a_fy_2025_26_handling_of_deemed_approval_deemed_acceptance_or_da_st` |
| `ocr` | OC 70 | 2 | 2,417 | 1,208 | `npci_upi_oc_181_compliance_to_merchant_onboarding_in_upi_and_usage_limits_90abfb` |
| `ocr` | OC 160 | 1 | 2,103 | 2,103 | `upi_oc_160_disabling_interchange_fee_movement_in_dispute_and_adjustment_life_cyc` |
| `ocr` | OC 145 | 1 | 1,959 | 1,959 | `upi_oc_145_a_reminder_on_final_implementation_timelines_for_odr_enhancing_compla` |
| `ocr` | OC 107 | 1 | 1,952 | 1,952 | `npci_upi_oc_107_a_fy_23_24_addendum_to_circular_for_revision_of_rrn_in_upi_to_av` |
| `ocr` | OC 148 | 1 | 1,937 | 1,937 | `upi_oc_148_processing_of_refunds` |
| `ocr` | — | 1 | 1,673 | 1,673 | `upi_oc_172_evidence_towords_upi_dispute_on_small_and_offline_merchants_5e0d80669` |
| `ocr` | OC 208B | 1 | 784 | 784 | `upi_oc_no_208_b_fy_2025_26_addendum_to_oc_208a_implementation_of_nrp_prd_process` |
| `text` | RBI/DPSS/2025-26/141 | 26 | 44,736 | 1,720 | `141md7d7f25debf1f48449e20d685e4b014e5` |
| `text` | RBI/2017-18/15 | 9 | 16,109 | 1,789 | `noti1506072017` |
| `text` | RBI/2019-20/67 | 5 | 8,085 | 1,617 | `circular677ec931a7a65e4d99aa957d8e85bc0a2a` |
| `text` | OC 206A | 4 | 7,987 | 1,996 | `upi_oc_no_206_a_fy_2024_25_addendum_to_oc_206_implementation_of_generic_good_fai` |
| `text` | RBI/2020-21/21 | 4 | 5,848 | 1,462 | `21odr602f0a579eb246aa885776a76122db0c` |
| `text` | — | 2 | 4,033 | 2,016 | `pr1184eng121121` |
| `text` | OC 213 | 1 | 1,638 | 1,638 | `upi_oc_no_213_fy_2024_25_auto_acceptance_rejection_of_chargeback_7e15e36905` |
| `text` | OC 141B | 1 | 1,601 | 1,601 | `npci_upi_oc_141_b_addendum_to_npci_oc_141_safeguarding_users_on_upi_addendum_4ed` |
| `text` | OC 184A | 1 | 774 | 774 | `upi_oc_no_184_a_fy_2024_25_addendum_to_oc_184_modification_in_upi_chargeback_rul` |

## Provenance

Every record carries `source_path` and `source_sha256`, so a chunk traces to an exact file, and `extraction_method` at both document and page level. Page text is stored separately as well as joined into `text` with `[[page N]]` markers, so page attribution survives chunking.

`circular_ref` and `title` are parsed from the **document's own text**, not from the filename. Filenames are download artefacts and sometimes disagree with the document.

## Flagged for review

17 item(s). None of these block ingestion; they are recorded so a human can check them.

- npci_upi_oc_107_a_fy_23_24_addendum_to_circular_for_revision_of_rrn_in_upi_to_av: parsed reference 'OC 107' disagrees with the filename hint 'OC 107A'; OCR misreads digits, so confirm by eye
- npci_upi_oc_181_compliance_to_merchant_onboarding_in_upi_and_usage_limits_90abfb: parsed reference 'OC 70' disagrees with the filename hint 'OC 181C'; OCR misreads digits, so confirm by eye
- oc_no_141_7e958ee3b1: parsed reference 'OC 741' disagrees with the filename hint 'OC 141'; OCR misreads digits, so confirm by eye
- osdt31012019: page(s) [4, 6] are blank in the source document (no ink), so they carry no text by design
- osdt31012019: no circular reference could be parsed from the document text
- pr1184eng121121: no circular reference could be parsed from the document text
- upi_oc_no_184_b_fy_2025_26_addendum_to_oc_184_modification_in_upi_chargeback_rul: parsed reference 'OC 184' disagrees with the filename hint 'OC 184B'; OCR misreads digits, so confirm by eye
- upi_oc_no_208_fy_24_25_implementation_of_nrp_prd_process_arbitration_guidelines: no circular reference could be parsed from the document text
- upi_oc_no_208_a_fy_2025_26_addendum_to_oc_208_implementation_of_nrp_prd_process_: parsed reference 'OC 208' disagrees with the filename hint 'OC 208A'; OCR misreads digits, so confirm by eye
- upi_oc_145_a_reminder_on_final_implementation_timelines_for_odr_enhancing_compla: parsed reference 'OC 145' disagrees with the filename hint 'OC 145A'; OCR misreads digits, so confirm by eye
- upi_oc_157_upi_raw_data_version_3_0_1edb076c72: parsed reference 'OC 152' disagrees with the filename hint 'OC 157'; OCR misreads digits, so confirm by eye
- upi_oc_172_evidence_towords_upi_dispute_on_small_and_offline_merchants_5e0d80669: no circular reference could be parsed from the document text
- upi_oc_193_fy_23_24_non_compliance_to_txnid_b76ac4cdb9: no circular reference could be parsed from the document text
- upi_oc_no_222_fy_2025_26_segregation_of_upi_settlement_cycles_for_auth_and_dispu: no circular reference could be parsed from the document text
- upi_oc_no_235_fy_2026_27_revision_of_turnaround_time_tat_for_responding_to_fraud: no circular reference could be parsed from the document text
- upi_oc_no_39_a_fy_2025_26_handling_of_deemed_approval_deemed_acceptance_or_da_st: parsed reference 'OC 39' disagrees with the filename hint 'OC 39A'; OCR misreads digits, so confirm by eye
- upi_settlement_process_256f73e1df: no circular reference could be parsed from the document text

**On the reference mismatches.** OCR misreads digits — a `1` scanned as a `7`, a trailing `A`/`B`/`C` lost to a smudge — so a reference parsed out of recognised text can be wrong. These are surfaced rather than silently overwritten from the filename, because the filename is not authoritative either and quietly substituting it would hide the OCR quality problem instead of exposing it. Resolve them by eye against the PDF before anything downstream relies on `circular_ref` as a key.

## Known limitations

- OCR quality is unmeasured. No page has been checked against a human transcription, so the character error rate is unknown. The evidence for it being adequate is indirect: the scanned documents average around 1,730 characters per page, which is in the range of the born-digital ones, and no document fell under the 400-character review threshold.
- Tables are flattened by OCR. The two that retrieval most depends on — OC 208 §C and the OC 184B RGNB table — are addressed by the curated records above, but every other table in the corpus is still running text. A chunker that splits mid-table will produce misleading fragments; this needs attention when chunking is designed.
- Recognition is English-only. Several RBI circulars carry a Hindi header, which is recognised as noise. It sits at the top of page 1 and does not affect the operative English text below it.
- Tesseract is a system binary, not a Python dependency, so `uv sync` alone does not make ingestion reproducible on another machine. The corpus output is DVC-tracked so it does not have to be regenerated to be used.
