# app/

Serving entry points for the vertical slice.

- **HTTP** — `uv run uvicorn app.main:app --reload`, then `POST /decide` with a dispute and its
  evidence. `GET /health` is a liveness probe. Interactive docs at `/docs`.
- **CLI** — `uv run python -m src.serving.cli --case seed_001` for one case,
  `--json` for the full audit record, `--list` for the available case ids.
- **Whole seed set** — `uv run python -m src.evaluation.run_seed`.

The application object lives in `src/serving/api.py`; this package holds only the entry points.
