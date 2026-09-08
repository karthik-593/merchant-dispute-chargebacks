"""Cross-encoder reranking: a second, more expensive pass over a first stage's candidates.

A bi-encoder embeds the question and every chunk *separately*, so the two never meet until a dot
product at the end. A cross-encoder puts them in one sequence and reads them together, which is
why it ranks better and why it cannot be used as a first stage: scoring a whole corpus one pair at
a time is quadratic in the wrong place. It reranks a shortlist, and it can only ever reorder what
the shortlist already contains — an answer the feeder left outside `k_retrieve` is unreachable, no
matter how good the reranker is.

**Three knobs, deliberately never collapsed into one "K":**

- `K_RETRIEVE`  — how many candidates the first stage hands over. Sets the ceiling on what the
                  reranker can recover.
- rerank        — the cross-encoder scores all of them.
- `K_CONTEXT`   — how many survive to the answer. This is what a downstream LLM actually reads and
                  pays for.

Calling all three "K" is how a pipeline ends up retrieving 25 and reporting recall@25 as though it
were the context window.

**Coverage matters as much as depth.** `k_retrieve` is only "retrieval" if it is a small slice of
the corpus. Twenty-five chunks out of 306 is a selection; twenty-five out of 36 is two-thirds of
everything, and a reranker over that is measuring the reranker with the first stage switched off.
`coverage_share` exists so that can be read off the table rather than reconstructed by eye.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.logging_setup import get_logger
from src.retrieval.retrievers import RetrievalHit

log = get_logger(__name__)

# Both fit comfortably in 8 GB. MiniLM is a 6-layer distillation trained on MS MARCO; bge-reranker
# is BERT-base sized and trained for exactly this retrieval-reranking job, so it should be the
# stronger of the two and the slower.
CROSS_ENCODERS: dict[str, dict[str, str]] = {
    "ms-marco-MiniLM-L-6-v2": {"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    "bge-reranker-base": {"model_id": "BAAI/bge-reranker-base"},
}

K_RETRIEVE = 25
K_CONTEXT = 5
RERANK_BATCH_SIZE = 32


@dataclass(frozen=True)
class RerankOutcome:
    """One question's journey through the two stages, kept separable on purpose."""

    before: list[RetrievalHit]
    after: list[RetrievalHit]

    @property
    def candidates(self) -> int:
        """How many the reranker had to work with."""
        return len(self.before)


_CACHE: dict[tuple[str, str], Any] = {}


def load_cross_encoder(name: str, device: str | None = None):  # noqa: ANN201
    """Load, cache and warm a cross-encoder.

    Warmed with a full batch of full-length pairs before it is handed out, for the same reason the
    bi-encoders are: the first call on CUDA selects and compiles kernels, and without this the
    first cell to use a model is charged seconds of start-up as though it were rerank latency.
    """
    from sentence_transformers import CrossEncoder

    from src.retrieval.retrievers import resolve_device

    resolved = resolve_device(device)
    key = (name, resolved)
    if key not in _CACHE:
        started = time.perf_counter()
        model = CrossEncoder(CROSS_ENCODERS[name]["model_id"], device=resolved)
        model.predict(
            [("warm up " * 20, "warm up " * 200)] * RERANK_BATCH_SIZE,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        _CACHE[key] = model
        log.info(
            "loaded cross-encoder %s on %s in %.1fs", name, resolved, time.perf_counter() - started
        )
    return _CACHE[key]


class CrossEncoderReranker:
    """Rescores a shortlist by reading question and passage together."""

    def __init__(self, name: str, device: str | None = None) -> None:
        """Load the named cross-encoder."""
        self.name = name
        self.model = load_cross_encoder(name, device=device)

    def rerank(self, question: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Reorder `hits` by cross-encoder score, renumbering ranks from 1.

        Ties break on the incoming order, so a reranker that cannot separate two candidates leaves
        the first stage's judgement standing rather than shuffling it.
        """
        if not hits:
            return []
        pairs = [(question, hit.chunk.text) for hit in hits]
        scores = np.asarray(
            self.model.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False),
            dtype=np.float64,
        ).reshape(-1)
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        return [
            RetrievalHit(rank=rank, score=float(scores[i]), chunk=hits[i].chunk)
            for rank, i in enumerate(order, start=1)
        ]


class RerankedRetriever:
    """A first stage plus a cross-encoder, presented as one retriever.

    `search(question, k)` returns the top `k` *after* reranking, having asked the feeder for
    `k_retrieve` candidates first. The two depths stay separate fields so a result can always say
    which one it is talking about.
    """

    def __init__(
        self,
        feeder: Any,
        reranker: CrossEncoderReranker,
        k_retrieve: int = K_RETRIEVE,
    ) -> None:
        """Wrap a feeder with a reranker at a stated retrieval depth."""
        self.feeder = feeder
        self.reranker = reranker
        self.k_retrieve = k_retrieve
        self.chunks = feeder.chunks
        self.name = f"{feeder.name}+rerank"

    @property
    def coverage_share(self) -> float:
        """`k_retrieve` as a fraction of the indexed chunks.

        The number that says whether the first stage is selecting or just handing over the corpus.
        """
        return min(1.0, self.k_retrieve / len(self.chunks))

    def candidates(self, question: str) -> list[RetrievalHit]:
        """What the feeder proposes, before reranking."""
        return self.feeder.search(question, self.k_retrieve)

    def rerank_outcome(self, question: str) -> RerankOutcome:
        """Both stages for one question, so before and after can be compared per query."""
        before = self.candidates(question)
        return RerankOutcome(before=before, after=self.reranker.rerank(question, before))

    def search(self, question: str, k: int = K_CONTEXT) -> list[RetrievalHit]:
        """Retrieve `k_retrieve`, rerank, keep `k`."""
        return self.rerank_outcome(question).after[:k]
