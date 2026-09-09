"""Curated corpus records for the three tables OCR cannot render usefully.

OCR flattens a table into running text: the OC 208 §C evidence map, the OC 184B RGNB response
table and the OC 208A Annexure A reject taxonomy all lose their row and column structure, so a
retrieved chunk of any of them reads as a stream of codes with no reliable association between a
code and the text that answers it. Retrieval over that is worse than useless — it looks like an
answer.

OC 208A is the worst of the three, and differently so. The other two are merely flattened; that
one is *damaged*. Fifteen of its 28 reason codes are wrong or missing in the scan — `Ilagible`
for `Illegible`, `4146` for `1146`, `1184` for `1154`, `TAN` for `TXN`, and four rows whose code
or description is simply absent. The corpus is a controlled regulatory rule set verified at load
time, not a live OCR feed, so it is brought up to that standard here rather than downstream.

All three are rebuilt from the structured forms already held in `configs/rulebook/`, which is the
verified transcription of those tables and carries the source tags. Nothing here is hand-keyed at
ingestion time, and nothing here invents a rule: change the rulebook and these records change
with it. Every difference between the OC 208A scan and its curated record is itemised in
`configs/corpus/oc_208a_reconciliation.yaml` with its authority, and the scanned record stays in
the corpus — superseded, never overwritten.

**Only these three.** Every other table stays as OCR'd until someone verifies it against the
source. A curated record asserts that a human checked it, so producing them casually would
destroy the meaning of the label.
"""

from __future__ import annotations

from pathlib import Path

from src.config import project_path
from src.data.rulebook_vocab import (
    load_caps,
    load_reject_taxonomy,
    load_rulebook,
    merchant_type_rule,
    reason_code_entries,
)
from src.retrieval.corpus_ingest import (
    EXTRACTION_CURATED,
    PAGE_MARKER,
    DocumentRecord,
    PageRecord,
    sha256_of,
    source_pdfs,
)

# The scanned circulars these curated records are drawn from and cite.
OC_208_FILENAME_PREFIX = "UPI-_-OC-No_-208-_-FY-24-25"
OC_184B_FILENAME_PREFIX = "UPI-_-OC-No_-184-B"
OC_208A_FILENAME_PREFIX = "UPI-_-OC-No_-208-A"

OC_208_DOC_ID = "oc_208_sec_c_evidence_map_curated"
OC_184B_DOC_ID = "oc_184b_rgnb_response_table_curated"
OC_208A_DOC_ID = "oc_208a_annexure_a_reject_taxonomy_curated"

RECONCILIATION_FILE = "oc_208a_reconciliation.yaml"


def load_reconciliation() -> dict:
    """Read the OC 208A OCR reconciliation log."""
    import yaml

    path = project_path("configs") / "corpus" / RECONCILIATION_FILE
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _find_source(prefix: str) -> Path:
    """Locate the circular a curated record cites, so provenance points at a real file."""
    for path in source_pdfs():
        if path.name.startswith(prefix):
            return path
    raise FileNotFoundError(f"no circular in refs/circulars/ starts with {prefix!r}")


def _make_doc_id(path: Path) -> str:
    """The ingested record's doc_id for a source PDF, so a curated record can name what it beats."""
    from src.retrieval.corpus_ingest import make_doc_id

    return make_doc_id(path)


def _wrap(doc_id: str, source: Path, section: str, title: str, body: str) -> DocumentRecord:
    """Package curated text with the same provenance shape every other record carries."""
    text = f"{PAGE_MARKER.format(page=1)}\n{body.strip()}"
    return DocumentRecord(
        doc_id=doc_id,
        circular_ref=None,
        title=title,
        source_path=str(source.relative_to(source.parents[2])).replace("\\", "/"),
        source_sha256=sha256_of(source),
        source_section=section,
        supersedes_doc_id=_make_doc_id(source),
        extraction_method=EXTRACTION_CURATED,
        page_count=1,
        char_count=len(text),
        pages=[
            PageRecord(
                page=1,
                extraction_method=EXTRACTION_CURATED,
                char_count=len(text),
                text=text,
            )
        ],
        text=text,
    )


def build_oc_208_evidence_map() -> DocumentRecord:
    """The OC 208 §C evidence table, rendered from the rulebook as readable rows."""
    rulebook = load_rulebook()
    vocabulary = rulebook["evidence_types"]
    entries = reason_code_entries()
    sub_types = rulebook["meta"]["txn_sub_types"]

    lines = [
        "OC 208 Section C - Type of Evidence",
        "",
        "Mandatory evidence by chargeback reason code and transaction sub-type. Supplying any ONE",
        "of the listed evidence types satisfies the requirement for that reason code.",
        "",
        f"Transaction sub-types: {', '.join(f'{k} = {v}' for k, v in sub_types.items())}",
        "",
    ]

    for key, entry in entries.items():
        code = entry.get("code", key)
        types = entry["required_evidence_any_of"]
        lines += [
            f"Reason code {code} ({', '.join(entry['txn_sub_type'])}) - "
            f"{entry.get('source_description', entry['description'])}",
            "  Accepted evidence (any one of):",
        ]
        lines += [f"    - {vocabulary[name]} [{name}]" for name in types]
        if entry.get("notes"):
            lines.append(f"  Note: {' '.join(entry['notes'].split())}")
        lines += [f"  Source: {entry['source']}", ""]

    merchant_rule = merchant_type_rule()
    lines += ["Merchant-type rule", ""]
    for branch, rule in merchant_rule.items():
        lines.append(f"  {branch}: {' '.join(rule['rule'].split())}")
        if rule.get("substitute_evidence"):
            substitute = rule["substitute_evidence"]
            lines.append(f"    Substitute evidence: {vocabulary[substitute]} [{substitute}]")
        lines.append(f"    Source: {rule['source']}")
    lines.append("")

    auto_loss = rulebook["auto_loss_conditions"]
    lines += [
        "Auto-loss conditions - the verdict goes to the remitter regardless of the map above:",
    ]
    lines += [f"  - {' '.join(condition.split())}" for condition in auto_loss["conditions"]]
    lines.append(f"  Source: {auto_loss['source']}")

    return _wrap(
        doc_id=OC_208_DOC_ID,
        source=_find_source(OC_208_FILENAME_PREFIX),
        section="§C (Type of Evidence)",
        title="OC 208 §C - Type of Evidence (curated evidence map)",
        body="\n".join(lines),
    )


def build_oc_184b_rgnb_table() -> DocumentRecord:
    """The OC 184B RGNB response table, rendered from the rulebook as readable rows."""
    rgnb = load_caps()["rgnb_responses"]
    lines = [
        "OC 184B - RGNB reason codes, responder and response TAT",
        "",
        "The remitter good-faith negative chargeback path, taken when a genuine claim is blocked",
        "by the CD1 or CD2 chargeback cap. Front-end only, no NPCI whitelisting.",
        "",
    ]
    for row in rgnb["rows"]:
        tat = row["tat"] or "not applicable (the raising side)"
        penalty = f" (penalty {row['penalty']})" if row.get("penalty") else ""
        # Verbatim source wording wins over the rulebook gloss wherever it has been verified. A
        # row still on the gloss says so, so a reader of the record can tell which is which
        # instead of assuming every line is quoted from the circular.
        verbatim = row.get("source_text")
        meaning = verbatim or f"{row['meaning']}  [rulebook gloss - source wording not verified]"
        lines += [
            f"Flag {row['flag']} / Code {row['code']} - {row['txn']}",
            f"  Meaning  : {meaning}",
            f"  Responder: {row['responder']}",
            f"  Response TAT: {tat}{penalty}",
            "",
        ]
    lines += [f"{rgnb['note']}", f"Source: {rgnb['source']}"]

    return _wrap(
        doc_id=OC_184B_DOC_ID,
        source=_find_source(OC_184B_FILENAME_PREFIX),
        section="RGNB response table",
        title="OC 184B - RGNB reason codes, responder and response TAT (curated table)",
        body="\n".join(lines),
    )


def build_oc_208a_reject_taxonomy() -> DocumentRecord:
    """The OC 208A Annexure A reject taxonomy, rendered clean from the rulebook.

    One block per code, opening `Reason code <n>` so the structure-aware chunker splits on the
    same boundary it already uses for the OC 208 §C table — one code per chunk, which is the unit
    a question about a reject reason actually asks about.
    """
    taxonomy = load_reject_taxonomy()
    meta = taxonomy["meta"]
    reconciliation = load_reconciliation()
    damaged = {c["code"] for c in reconciliation["corrections"]}

    lines = [
        "OC 208A Annexure A - NRP verdict reason codes",
        "",
        "The reason codes an NPCI Review Panel verdict is recorded under, in two bands that must",
        "not be confused. NVB 1126-1131 record a verdict FOR the beneficiary - the representment",
        "SUCCEEDED - and are never fail conditions. NVR 1132-1157 record a verdict AGAINST: the",
        "invalid-reason catalog the verifier exists to catch before anything is filed.",
        "",
        f"Scope: {meta['covers']}. Source: {meta['primary_source']}, {meta['circular']}.",
        "",
        "This record supersedes the OCR'd rendering of the same annexure, in which 15 of these 28",
        "codes are damaged. Every correction is itemised, with its authority, in",
        f"configs/corpus/{RECONCILIATION_FILE}.",
        "",
    ]

    for entry in taxonomy["nrp_verdict_for_codes"]:
        code = entry["code"]
        lines.append(f"Reason code {code} - {' '.join(entry['description'].split())}")
        lines.append("  Verdict: FOR the beneficiary (NVB) - the representment succeeded.")
        lines += [f"  Source: {entry['source']}", ""]

    for entry in taxonomy["nrp_verdict_against_codes"]:
        code = entry["code"]
        lines.append(f"Reason code {code} - {' '.join(entry['description'].split())}")
        if entry.get("transcription_note"):
            lines.append(f"  Transcription note: {' '.join(entry['transcription_note'].split())}")
        lines.append("  Verdict: AGAINST (NVR) - an invalid or defective submission.")
        if code in damaged:
            lines.append("  OCR reconciliation: corrected against the rulebook; see the log.")
        lines += [f"  Source: {entry['source']}", ""]

    lines.append(f"Source: {meta['primary_source']}")

    return _wrap(
        doc_id=OC_208A_DOC_ID,
        source=_find_source(OC_208A_FILENAME_PREFIX),
        section="Annexure A (NRP verdict reason codes)",
        title="OC 208A Annexure A - NRP verdict reason codes (curated, OCR-reconciled)",
        body="\n".join(lines),
    )


def build_curated_records() -> list[DocumentRecord]:
    """Every curated record. Deliberately just the three verified tables."""
    return [
        build_oc_208_evidence_map(),
        build_oc_184b_rgnb_table(),
        build_oc_208a_reject_taxonomy(),
    ]
