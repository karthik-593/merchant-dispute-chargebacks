"""Configures process-wide logging from configs/config.yaml."""

from __future__ import annotations

import logging

from src.config import load_config

_CONFIGURED = False

# Libraries that log every HTTP request at INFO. At our level that is one line per file a model
# download touches, which buries the experiment's own output in transport noise.
_NOISY = ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers", "transformers")


def setup_logging(level: str | None = None) -> None:
    """Install the configured root handler once per process."""
    global _CONFIGURED
    cfg = load_config()["logging"]
    logging.basicConfig(
        level=level or cfg["level"],
        format=cfg["format"],
        datefmt=cfg["datefmt"],
        force=True,
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
