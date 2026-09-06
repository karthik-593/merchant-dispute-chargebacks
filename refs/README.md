# refs/ — source circulars

Primary-source corpus for the project. **Documents win over model priors**: every domain fact
encoded in `configs/rulebook/` — reason code, required evidence, TAT, fee, cap — must cite a
circular and section from this set. A rule with no traceable source does not exist, and no rule
is ever invented from model priors. This directory is the evidence behind the rulebook.

Files live in `refs/circulars/` under their original download names — names are kept verbatim so
each file can be traced back to its NPCI/RBI source page.

## OCR status

**Most NPCI circulars here are image scans with no text layer** — 22 of the 33 PDFs return zero
extractable characters. They must be OCR'd before they can be chunked into the retrieval corpus
(L1). The RBI documents are mostly born-digital and extract cleanly.

The "Text" column below records what a plain PDF text extraction yields today:

- `digital` — usable text layer, ingest directly
- `OCR` — image scan, **OCR required**
- `OCR (noisy)` — a low-quality OCR layer is embedded; re-OCR rather than trust it

Ingestion (chunking, embedding, retriever) is milestone M6 — not built yet.

## NPCI — UPI Operating Circulars

| OC | Title | Text | File |
|---|---|---|---|
| OC 39A / 2025-26 | Deemed Approval / Deemed Acceptance (DA) for P2M where acquiring PSP = merchant bank | OCR | `UPI_OC_No_39_A_FY_2025_26_Handling_of_Deemed_Approval_Deemed_Acceptance_or_DA_status_in_UPI_for_P2_M_transaction_where_Acquiring_PSP_and_Merchant_Bank_are_the_same_482ed0c0f6.pdf` |
| OC 107A / 2023-24 | Revision of RRN in UPI to avoid duplicates (addendum) | OCR | `NPCI_UPI_OC_107_A_FY_23_24_Addendum_to_Circular_for_revision_of_RRN_in_UPI_to_avoid_duplicates_SD_7287bf719d.pdf` |
| OC 141 | Safeguarding users on UPI | OCR | `OC_No_141_7e958ee3b1.pdf` |
| OC 141B / 2022-23 | Addendum to OC 141 — safeguarding users on UPI | OCR (noisy) | `NPCI_UPI_OC_141_B_Addendum_to_NPCI_OC_141_Safeguarding_users_on_UPI_Addendum_4ed7e168ed.pdf` |
| OC 145A | Final implementation timelines for ODR — complaint handling and resolution | OCR | `UPI_OC_145_A_Reminder_on_Final_implementation_timelines_for_ODR_Enhancing_Complaint_handling_and_Resolution_process_for_all_UPI_users_3d1f596303.pdf` |
| OC 148 | Processing of refunds | OCR | `UPI-OC-148-–-Processing-of-refunds.pdf` |
| OC 157 | UPI raw data version 3.0 | OCR | `UPI_OC_157_UPI_Raw_data_Version_3_0_1edb076c72.pdf` |
| OC 160 | Disabling interchange fee movement in dispute and adjustment lifecycles | OCR | `UPI_OC_160_Disabling_Interchange_fee_movement_in_dispute_and_adjustment_life_cycles_7bbec9c65b.pdf` |
| OC 165 / 2023-24 | Advanced refund API as part of UPI Help (UDIR) — updated 23 Feb 2026 | OCR | `UPI-_-OC-165-_-FY-23-24-_-Implementation-of-advanced-refund-API-as-part-of-UPI-Help-(UDIR)-(Updated-as-on-23rd-February-2026).pdf` |
| OC 172 | Evidence towards UPI dispute on small and offline merchants | OCR | `UPI_OC_172_Evidence_towords_UPI_dispute_on_Small_and_Offline_Merchants_5e0d80669a.pdf` |
| OC 181 | Compliance to merchant onboarding in UPI and usage limits | OCR | `NPCI_UPI_OC_181_Compliance_to_Merchant_Onboarding_in_UPI_and_Usage_Limits_90abfb97fa.pdf` |
| OC 184 / 2023-24 | Modification in UPI chargeback rules and procedures | OCR | `UPI_OC_No_184_FY_23_24_Modification_in_UPI_Chargeback_Rules_and_Procedures_bdfd9bba33.pdf` |
| OC 184A / 2024-25 | Addendum to OC 184 — chargeback rules and procedures (12 Mar 2025) | digital | `UPI_OC_No_184_A_FY_2024_25_Addendum_to_OC_184_Modification_in_UPI_chargeback_rules_and_procedures_e1a86aa703.pdf` |
| OC 184B / 2025-26 | Addendum to OC 184 — chargeback rules and procedures | OCR | `UPI-_-OC-No_-184-B-_-FY-2025-26-_-Addendum-to-OC-–-184-Modification-in-UPI-chargeback-rules-and-procedures (1).pdf` |
| OC 193 / 2023-24 | Non-compliance to TXNID | OCR | `UPI_OC_193_FY_23_24_Non_Compliance_to_TXNID_b76ac4cdb9.pdf` |
| OC 197 / 2024-25 | 10 settlement cycles and revised business-day cutover | OCR | `UPI_OC_No_197_FY_24_25_Implementation_of_10_settlement_cycles_and_revised_business_day_cutover_158b67f6bc.pdf` |
| OC 198 / 2024-25 | Revision of disputes TAT | OCR | `UPI_OC_No_198_FY_24_25_Revision_of_Disputes_TAT_7863285dd6.pdf` |
| OC 206 / 2024-25 | Generic good-faith debit/credit adjustments in URCS | OCR | `UPI_OC_No_206_FY_24_25_Implementation_of_Generic_Good_Faith_Debit_Credit_Adjustments_in_URCS_c1d5b8e1aa.pdf` |
| OC 206A / 2024-25 | Addendum to OC 206 — generic good-faith debit adjustments (12 Mar 2025) | digital | `UPI_OC_No_206_A_FY_2024_25_Addendum_to_OC_206_Implementation_of_Generic_Good_Faith_Debit_Adjustments_e543ac677c.pdf` |
| OC 208 / 2024-25 | NRP & PRD process & arbitration guidelines — **the core dispute circular** (§C = evidence table) | OCR | `UPI-_-OC-No_-208-_-FY-24-25-–-Implementation-of-NRP-&-PRD-process-&-arbitration-guidelines_.pdf` |
| OC 208A / 2025-26 | Addendum to OC 208 — NRP & PRD (Annexure A = invalid-reason taxonomy) | OCR | `UPI-_-OC-No_-208-A-_-FY-2025-26-_-Addendum-to-OC---208-Implementation-of-NRP-&-PRD-process-&-arbitration-guidelines (1).pdf` |
| OC 208B / 2025-26 | Addendum to OC 208A — NRP & PRD | OCR | `UPI-_-OC-No_-208-B-_-FY-2025-26-_-Addendum-to-OC-–-208A-Implementation-of-NRP-&-PRD-process-&-arbitration-guidelines (1).pdf` |
| OC 208C / 2026-27 | Addendum to OC 208/208A/208B — evidence admissible only up to pre-arbitration | OCR | `UPI-_-OC-No_-208C-_-FY-2026-27-_-(Addendum-to-OC-208,-208A-&-208B)---Implementation-of-NRP-&-PRD-process-&-arbitration-guidelines.pdf` |
| OC 213 / 2024-25 | Auto acceptance / rejection of chargeback (10 Feb 2025) | digital | `UPI_OC_No_213_FY_2024_25_Auto_Acceptance_Rejection_of_Chargeback_7e15e36905.pdf` |
| OC 222 / 2025-26 | Segregation of UPI settlement cycles for auth and dispute transactions | OCR | `UPI_OC_No_222_FY_2025_26_Segregation_of_UPI_settlement_cycles_for_Auth_and_disputes_transactions_25534237a3.pdf` |
| OC 235 / 2026-27 | Revision of TAT for responding to fraud and wrong-credit chargebacks in URCS back office | OCR | `UPI_OC_No_235_FY_2026_27_Revision_of_Turnaround_Time_TAT_for_Responding_to_Fraud_and_Wrong_Credit_Chargebacks_in_URCS_Back_Office_System_aa420e4bcc.pdf` |
| — | UPI settlement process (reference note) | digital | `UPI_Settlement_Process_256f73e1df.pdf` |

## RBI

| Reference | Title | Text | File |
|---|---|---|---|
| RBI/2017-18/15 (DBR.No.Leg.BC.78, 6 Jul 2017) | Customer Protection — Limiting Liability of Customers in Unauthorised Electronic Banking Transactions | digital | `NOTI1506072017.pdf` |
| RBI/2019-20/67 (DPSS.CO.PD No.629, 20 Sep 2019) | Harmonisation of Turn Around Time (TAT) and customer compensation for failed transactions | digital | `CIRCULAR677EC931A7A65E4D99AA957D8E85BC0A2A.pdf` |
| RBI/2020-21/21 (DPSS.CO.PD No.116, 6 Aug 2020) | Online Dispute Resolution (ODR) System for Digital Payments | digital | `21ODR602F0A579EB246AA885776A76122DB0C.pdf` |
| — (31 Jan 2019) | Ombudsman Scheme for Digital Transactions, 2019 | digital | `OSDT31012019.pdf` |
| Press release, 12 Nov 2021 | Reserve Bank — Integrated Ombudsman Scheme, 2021 | digital | `PR1184ENG121121.pdf` |
| RBI/DPSS/2025-26/141 (15 Sep 2025) | Master Direction — Payment Aggregators (PA/PG) | digital | `141MD7D7F25DEBF1F48449E20D685E4B014E5.pdf` |

## Not in this corpus (do not chase)

These are not publicly available and are reconstructed from the circulars above rather than
chased: OSG v2.1, UPI Procedural Guidelines, UPI/UDIR API & technical specification documents,
and standalone penalty / reason-code catalogues.
