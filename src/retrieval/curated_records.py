"""Curated corpus records for the two tables OCR cannot render usefully.

OCR flattens a table into running text: the OC 208 §C evidence map and the OC 184B RGNB response
table both lose their row and column structure, so a retrieved chunk of either reads as a stream
of codes with no reliable association between a reason code and the evidence that answers it.
Retrieval over that is worse than useless — it looks like an answer.

These two are rebuilt from the structured forms already held in `configs/rulebook/`, which is the
verified transcription of those tables and carries the source tags. Nothing here is hand-keyed at
ingestion time, and nothing here invents a rule: change the rulebook and these records change
with it.

**Only these two.** Every other table stays as OCR'd until someone verifies it against the
source. A curated record asserts that a human checked it, so producing them casually would
destroy the meaning of the label.
"""

from __future__ import annotations

from pathlib import Path

from src.data.rulebook_vocab import (
    load_caps,
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

OC_208_DOC_ID = "oc_208_sec_c_evidence_map_curated"
OC_184B_DOC_ID = "oc_184b_rgnb_response_table_curated"


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
            f"Reason code {code} ({', '.join(entry['txn_sub_type'])}) - {entry['description']}",
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
        lines += [
            f"Flag {row['flag']} / Code {row['code']} - {row['txn']}",
            f"  Meaning  : {row['meaning']}",
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


def build_curated_records() -> list[DocumentRecord]:
    """Every curated record. Deliberately just the two verified tables."""
    return [build_oc_208_evidence_map(), build_oc_184b_rgnb_table()]
