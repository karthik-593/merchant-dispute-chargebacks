"""Uvicorn entry point: `uv run uvicorn app.main:app --reload`."""

from src.serving.api import app

__all__ = ["app"]
