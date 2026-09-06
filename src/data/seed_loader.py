"""Loads the hand-built seed fixtures from data/seed/ into the typed models."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import project_path
from src.data.schemas import SeedCase


def seed_dir() -> Path:
    """Directory holding the seed case files."""
    return project_path("seed_data")


def seed_files() -> list[Path]:
    """Every seed case file, in stable order."""
    return sorted(seed_dir().glob("*.yaml"))


def load_seed_case(path: Path) -> SeedCase:
    """Parse one seed case file into a validated model."""
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return SeedCase.model_validate(payload)


def load_seed_cases() -> list[SeedCase]:
    """Parse the whole seed set, raising on the first file that fails validation."""
    cases = []
    for path in seed_files():
        try:
            cases.append(load_seed_case(path))
        except Exception as error:  # noqa: BLE001 - re-raised with the offending filename
            raise ValueError(f"{path.name}: {error}") from error
    return cases
