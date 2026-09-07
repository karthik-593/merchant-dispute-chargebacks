"""Structured evidence store, keyed by dispute id.

This is a **keyed lookup**, not a search problem. Given a dispute, its artifacts are found by an
exact join on `dispute_id` — there is no ranking, no scoring and no embedding anywhere in here.
Semantic retrieval belongs to the separate circulars corpus; conflating the two would be a
category error, because "which artifacts belong to this dispute" has one right answer that a
similarity score could only approximate.

Sources are the two datasets: the hand-built `seed-v1` fixtures and the generated
`dataset-v1.0` splits.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.config import project_path
from src.data.schemas import DisputeRecord, EvidenceArtifact, SeedCase
from src.data.seed_loader import load_seed_cases
from src.data.synthetic_dataset import read_jsonl, synthetic_dir
from src.logging_setup import get_logger

log = get_logger(__name__)

SEED_SOURCE = "seed-v1"
SYNTHETIC_SOURCE = "dataset-v1.0"
SYNTHETIC_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class StoredDispute:
    """One dispute with its artifacts and the dataset it came from."""

    dispute: DisputeRecord
    evidence: tuple[EvidenceArtifact, ...]
    source: str
    case_id: str
    split: str | None = None


class EvidenceStore:
    """An index over disputes and their evidence, keyed by dispute id."""

    def __init__(self, records: dict[str, StoredDispute] | None = None) -> None:
        """Hold the index. Use the `from_*` constructors to populate it."""
        self._by_dispute: dict[str, StoredDispute] = dict(records or {})

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_cases(
        cls, cases: list[SeedCase], source: str, split: str | None = None
    ) -> EvidenceStore:
        """Index a list of cases."""
        store = cls()
        store.add_cases(cases, source=source, split=split)
        return store

    @classmethod
    def load(
        cls,
        include_seed: bool = True,
        include_synthetic: bool = True,
        synthetic_directory: Path | None = None,
    ) -> EvidenceStore:
        """Load whichever datasets are present on disk.

        The synthetic splits are DVC-tracked, so on a fresh clone they may not be there yet.
        That is reported and skipped rather than raising: the seed set alone is a usable store.
        """
        store = cls()
        if include_seed:
            store.add_cases(load_seed_cases(), source=SEED_SOURCE)
        if include_synthetic:
            root = synthetic_directory or synthetic_dir()
            for split in SYNTHETIC_SPLITS:
                path = root / f"{split}.jsonl"
                if not path.is_file():
                    log.warning("synthetic split %s is not on disk; skipping (dvc pull?)", path)
                    continue
                store.add_cases(read_jsonl(path), source=SYNTHETIC_SOURCE, split=split)
        return store

    def add_cases(
        self, cases: list[SeedCase], source: str, split: str | None = None
    ) -> EvidenceStore:
        """Add cases to the index, refusing to overwrite an existing dispute id."""
        for case in cases:
            dispute_id = case.dispute.dispute_id
            if dispute_id in self._by_dispute:
                existing = self._by_dispute[dispute_id]
                raise ValueError(
                    f"duplicate dispute_id {dispute_id!r}: already indexed from "
                    f"{existing.source}/{existing.case_id}, now offered by {source}/{case.case_id}"
                )
            self._by_dispute[dispute_id] = StoredDispute(
                dispute=case.dispute,
                evidence=tuple(case.evidence),
                source=source,
                case_id=case.case_id,
                split=split,
            )
        return self

    # -- lookup --------------------------------------------------------------------------

    def get_evidence(self, dispute_id: str) -> list[EvidenceArtifact]:
        """Every artifact filed against a dispute, in the order it was filed.

        Raises:
            KeyError: if the dispute is not in the store.
        """
        return list(self._require(dispute_id).evidence)

    def get_dispute(self, dispute_id: str) -> DisputeRecord:
        """The dispute record itself.

        Raises:
            KeyError: if the dispute is not in the store.
        """
        return self._require(dispute_id).dispute

    def get(self, dispute_id: str) -> StoredDispute:
        """The dispute, its artifacts and its provenance together."""
        return self._require(dispute_id)

    def _require(self, dispute_id: str) -> StoredDispute:
        try:
            return self._by_dispute[dispute_id]
        except KeyError:
            raise KeyError(
                f"dispute {dispute_id!r} is not in the evidence store "
                f"({len(self._by_dispute)} disputes indexed)"
            ) from None

    # -- filters (still exact matching, never ranking) -------------------------------------

    def __contains__(self, dispute_id: object) -> bool:
        """Whether a dispute id is indexed."""
        return dispute_id in self._by_dispute

    def __len__(self) -> int:
        """How many disputes are indexed."""
        return len(self._by_dispute)

    def __iter__(self) -> Iterator[StoredDispute]:
        """Iterate the stored disputes."""
        return iter(self._by_dispute.values())

    def dispute_ids(self, source: str | None = None) -> list[str]:
        """Indexed dispute ids, optionally restricted to one dataset."""
        return sorted(
            dispute_id
            for dispute_id, record in self._by_dispute.items()
            if source is None or record.source == source
        )

    def evidence_by_type(self, evidence_type: str) -> list[EvidenceArtifact]:
        """Every artifact of a given type across the store. An exact filter, not a search."""
        return [
            artifact
            for record in self._by_dispute.values()
            for artifact in record.evidence
            if artifact.type == evidence_type
        ]

    def stats(self) -> dict[str, object]:
        """Counts describing what is indexed."""
        by_source: dict[str, int] = {}
        artifacts = 0
        without_evidence = 0
        for record in self._by_dispute.values():
            by_source[record.source] = by_source.get(record.source, 0) + 1
            artifacts += len(record.evidence)
            if not record.evidence:
                without_evidence += 1
        return {
            "disputes": len(self._by_dispute),
            "by_source": dict(sorted(by_source.items())),
            "artifacts": artifacts,
            "disputes_without_evidence": without_evidence,
        }


def corpus_dir() -> Path:
    """Directory holding the ingested circulars corpus. Kept separate from the evidence store."""
    return project_path("data") / "corpus"
