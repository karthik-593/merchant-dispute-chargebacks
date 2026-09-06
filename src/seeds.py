"""Seeds every random source in the process so runs are reproducible."""

from __future__ import annotations

import os
import random

from src.config import load_config


def set_seed(seed: int | None = None) -> int:
    """Seed stdlib/PYTHONHASHSEED and, when importable, numpy. Returns the seed used."""
    resolved = load_config()["seed"] if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(resolved)
    return resolved
