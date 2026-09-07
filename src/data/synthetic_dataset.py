"""Reading, writing, splitting and locking the synthetic dataset.

The split is temporal: cases are ordered by transaction timestamp and cut into train, validation
and test, so the test set is strictly the latest slice. That matches how the system would be used
— trained on the past, judged on what came after — and it makes leakage from a shuffled split
impossible by construction.

The test set is locked: its SHA-256 is written alongside it, so any later change to the file is
detectable rather than silent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import project_path
from src.data.schemas import SYNTHETIC_VERSION, SeedCase

SPLIT_NAMES = ("train", "val", "test")
DEV_SPLIT = "dev"
LOCK_FILE = "TEST_SET_LOCK.json"


@dataclass(frozen=True)
class DatasetSplits:
    """The generated cases, cut by transaction time."""

    train: list[SeedCase]
    val: list[SeedCase]
    test: list[SeedCase]
    dev: list[SeedCase]

    def as_dict(self) -> dict[str, list[SeedCase]]:
        """Splits by name, dev last."""
        return {"train": self.train, "val": self.val, "test": self.test, DEV_SPLIT: self.dev}

    @property
    def sizes(self) -> dict[str, int]:
        """Case count per split."""
        return {name: len(cases) for name, cases in self.as_dict().items()}


def synthetic_dir() -> Path:
    """Directory holding the synthetic dataset."""
    return project_path("data") / "synthetic"


def temporal_split(
    cases: list[SeedCase], fractions: dict[str, float], dev_size: int, dev_seed: int
) -> DatasetSplits:
    """Order by transaction timestamp and cut into train, validation and test.

    The dev subset is drawn from train only, so fast iteration can never touch held-out data.
    """
    ordered = sorted(cases, key=lambda case: (case.dispute.txn_timestamp, case.case_id))
    total = len(ordered)
    train_end = int(total * fractions["train"])
    val_end = train_end + int(total * fractions["val"])

    train = ordered[:train_end]
    val = ordered[train_end:val_end]
    test = ordered[val_end:]

    import numpy as np

    rng = np.random.default_rng(dev_seed)
    take = min(dev_size, len(train))
    picks = rng.choice(len(train), size=take, replace=False)
    dev = [train[int(i)] for i in sorted(picks)]

    return DatasetSplits(train=train, val=val, test=test, dev=dev)


def write_jsonl(cases: list[SeedCase], path: Path) -> Path:
    """Write cases one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(case.model_dump_json())
            handle.write("\n")
    return path


def read_jsonl(path: Path) -> list[SeedCase]:
    """Read a split back into validated models."""
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(SeedCase.model_validate_json(line))
            except Exception as error:  # noqa: BLE001 - re-raised with the offending line
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
    return cases


def sha256_of(path: Path) -> str:
    """Content hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_test_lock(test_path: Path, case_count: int, generator_seed: int) -> Path:
    """Record the test set's hash so a later change to it is detectable.

    The test set is never used to choose a model, a prompt or a threshold. Committing this hash
    makes that claim checkable instead of a promise.
    """
    lock = {
        "dataset_version": SYNTHETIC_VERSION,
        "locked_at": datetime.now(UTC).isoformat(),
        "generator_seed": generator_seed,
        "test_file": test_path.name,
        "test_case_count": case_count,
        "sha256": sha256_of(test_path),
        "note": (
            "The test split is locked. It must not be used for model, prompt or threshold "
            "selection. Regenerating the dataset with the same seed reproduces this hash; a "
            "different hash means the held-out set moved and any result measured on it is void."
        ),
    }
    path = test_path.parent / LOCK_FILE
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return path


def verify_test_lock(directory: Path | None = None) -> tuple[bool, str]:
    """Check the test file still matches its recorded hash."""
    root = directory or synthetic_dir()
    lock_path = root / LOCK_FILE
    if not lock_path.is_file():
        return False, f"no lock file at {lock_path}"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    test_path = root / lock["test_file"]
    if not test_path.is_file():
        return False, f"locked test file {test_path} is missing"
    actual = sha256_of(test_path)
    if actual != lock["sha256"]:
        return False, f"test set changed: expected {lock['sha256'][:12]}, found {actual[:12]}"
    return True, f"test set matches its lock ({lock['test_case_count']} cases)"


def load_splits(directory: Path | None = None) -> DatasetSplits:
    """Read every split back from disk."""
    root = directory or synthetic_dir()
    return DatasetSplits(
        train=read_jsonl(root / "train.jsonl"),
        val=read_jsonl(root / "val.jsonl"),
        test=read_jsonl(root / "test.jsonl"),
        dev=read_jsonl(root / "dev.jsonl"),
    )
