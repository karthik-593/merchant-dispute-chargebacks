"""Loads configs/config.yaml and resolves project paths. No domain rules live here."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@functools.cache
def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Read and cache the YAML runtime config."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(key: str) -> Path:
    """Resolve a named entry of the config `paths` block to an absolute path."""
    paths = load_config()["paths"]
    if key not in paths:
        raise KeyError(f"unknown path key {key!r}; known keys: {sorted(paths)}")
    return PROJECT_ROOT / paths[key]
