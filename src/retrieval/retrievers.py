"""Retrievers over the chunked circulars corpus: BM25, dense, and their fusion.

M6 held BM25 constant so that the chunker was the only thing varying. M7 varies the retriever,
and this module is where that variation lives. Three families:

- `bm25`   — lexical. Exact term overlap, no model, no vectors. The M6 constant, carried forward
             unchanged so the two experiments stay comparable.
- `dense`  — a small sentence-transformer embeds every chunk once and the query at search time;
             ranking is cosine similarity. Buys paraphrase tolerance, pays in truncation: these
             models see 512 tokens and silently drop the rest, which matters a great deal for a
             chunker whose chunks run to 6,700 tokens.
- `hybrid` — min-max normalises both score vectors and takes a weighted sum. BM25 scores are
             unbounded and cosine sits in [-1, 1], so the two cannot be added as they stand;
             normalising per query puts them on one scale and `alpha` says how much to trust the
             embedding. `alpha=0` is BM25, `alpha=1` is dense.

Every retriever returns `RetrievalHit`s that carry the `Chunk` itself, so `doc_id`,
`source_section` and the page range survive retrieval intact. A hit that cannot cite where it came
from is not usable downstream, and this is the easiest place in the system to lose that.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from src.logging_setup import get_logger
from src.retrieval.chunkers import Chunk

log = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

# Small, 8 GB-safe encoders. All three are ~22-33M parameters and emit 384-dim vectors, so the
# comparison is between training recipes rather than between model sizes.
#
# The prefixes are not decoration. e5 was trained with "query: " and "passage: " on every input
# and degrades measurably without them; bge expects an instruction on the query and a bare
# passage; MiniLM expects neither. Feeding all three the same bare text would benchmark prompt
# formatting rather than embeddings — the same class of mistake as not splitting RC_1064.
EMBEDDINGS: dict[str, dict[str, str]] = {
    "bge-small-en-v1.5": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "e5-small-v2": {
        "model_id": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    # Larger candidates, added for the M7 embedding re-selection. The three above are all
    # ~22-33M general-purpose encoders and all of them bury paraphrased regulatory rows, so the
    # question is whether that is a size limit, a family limit, or neither. BERT-large sized,
    # ~1.3 GB each - trivial against 8 GB, which the small models were never testing.
    "bge-large-en-v1.5": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "e5-large-v2": {
        "model_id": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    # A third training recipe rather than a third size. BGE and E5 are both represented above, so
    # scaling either only answers "is it too small". GTE is retrieval-tuned on a different data
    # mixture, takes no query/passage prefix, and needs no trust_remote_code - so a win here is a
    # recipe result, not a parameter-count one.
    "gte-large": {
        "model_id": "thenlper/gte-large",
        "query_prefix": "",
        "passage_prefix": "",
    },
}

DEFAULT_FUSION_ALPHA = 0.5

# Encoding batch size, shared by the warm-up and the real index build so both select the same
# CUDA kernels.
ENCODE_BATCH_SIZE = 32


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25, split on underscores as well as whitespace.

    The corpus writes evidence types as `invoice_with_proof_of_delivery` and reason codes as
    `RC_1064`, while a question says "RC 1064" and "invoice with proof of delivery". Indexing the
    identifier whole would mean the two never meet, and the resulting scores would say something
    about tokenisation rather than about retrieval.
    """
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked result, with the provenance needed to cite it."""

    rank: int
    score: float
    chunk: Chunk

    @property
    def doc_id(self) -> str:
        """Document this passage came from."""
        return self.chunk.doc_id

    @property
    def source_section(self) -> str | None:
        """Section of the circular, where the record names one."""
        return self.chunk.source_section

    def citation(self) -> str:
        """How this hit would be cited in an answer."""
        return self.chunk.citation()


class Retriever(Protocol):
    """Ranks chunks against a question."""

    name: str

    def search(self, question: str, k: int) -> list[RetrievalHit]:
        """Return the top k hits, best first."""
        ...


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1]; an all-equal vector carries no ranking signal, so it goes to 0."""
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[RetrievalHit]:
    """Take the k best-scoring chunks, ties broken by index order so runs are reproducible."""
    k = min(k, len(chunks))
    if k <= 0:
        return []
    candidates = np.argpartition(-scores, k - 1)[:k]
    ordered = sorted(candidates, key=lambda i: (-scores[i], i))
    return [
        RetrievalHit(rank=rank, score=float(scores[i]), chunk=chunks[i])
        for rank, i in enumerate(ordered, start=1)
    ]


class BM25Retriever:
    """Okapi BM25 over the chunk texts. No model, no vectors, no GPU."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        """Index the chunks lexically."""
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def scores(self, question: str) -> np.ndarray:
        """Raw BM25 score for every chunk."""
        return np.asarray(self.index.get_scores(tokenize(question)), dtype=np.float64)

    def search(self, question: str, k: int) -> list[RetrievalHit]:
        """Rank by BM25 score."""
        return _top_k(self.chunks, self.scores(question), k)


class DenseRetriever:
    """Cosine similarity over sentence-transformer embeddings.

    Embeddings are L2-normalised at encode time, so the dot product *is* the cosine and a search
    is one matrix-vector product. At a few hundred chunks there is nothing an ANN index could buy,
    and exact search removes approximation as a confound.
    """

    name = "dense"

    def __init__(self, chunks: list[Chunk], embedding: str, device: str | None = None) -> None:
        """Encode every chunk with the named embedding model."""
        self.chunks = chunks
        self.embedding = embedding
        self.spec = EMBEDDINGS[embedding]
        self.model = load_embedding_model(embedding, device=device)
        self.max_seq_length = int(self.model.max_seq_length)
        passages = [self.spec["passage_prefix"] + chunk.text for chunk in chunks]
        self.token_lengths = _subword_lengths(self.model, passages)
        self.matrix = self.model.encode(
            passages,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_query(self, question: str) -> np.ndarray:
        """Embed one question, with whatever prefix this model was trained to expect."""
        return self.model.encode(
            self.spec["query_prefix"] + question,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def scores(self, question: str) -> np.ndarray:
        """Cosine similarity between the question and every chunk."""
        return self.matrix @ self.encode_query(question)

    def search(self, question: str, k: int) -> list[RetrievalHit]:
        """Rank by cosine similarity."""
        return _top_k(self.chunks, self.scores(question), k)

    def truncated_share(self) -> float:
        """Share of chunks longer than the model's context window.

        The encoder does not complain when a chunk overruns; it truncates and returns a vector
        representing the opening of the passage as though it were the whole thing. For a chunker
        whose chunks are whole documents that is most of the text going unseen, and it is the
        single most important thing to know when reading a dense result.

        Counted in the model's own subword tokens, not in the whitespace words the chunkers count.
        A 400-word chunk is comfortably under 512 by one measure and over it by the other, and the
        one that decides whether text is dropped is the model's.
        """
        return float(np.mean([n > self.max_seq_length for n in self.token_lengths]))

    def seen_share(self) -> float:
        """Share of subword tokens the encoder actually read across the whole index."""
        total = sum(self.token_lengths)
        if not total:
            return 0.0
        return sum(min(n, self.max_seq_length) for n in self.token_lengths) / total


class HybridRetriever:
    """Weighted fusion of BM25 and dense scores, both min-max normalised per query."""

    name = "hybrid"

    def __init__(
        self,
        lexical: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = DEFAULT_FUSION_ALPHA,
    ) -> None:
        """Fuse an existing lexical and dense index, at weight `alpha` on the dense side."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"fusion alpha must be in [0, 1], got {alpha}")
        if lexical.chunks is not dense.chunks and len(lexical.chunks) != len(dense.chunks):
            raise ValueError("lexical and dense indexes must cover the same chunks")
        self.lexical = lexical
        self.dense = dense
        self.alpha = alpha
        self.chunks = lexical.chunks

    def scores(self, question: str) -> np.ndarray:
        """Normalise both score vectors and take the weighted sum."""
        lexical = _minmax(self.lexical.scores(question))
        dense = _minmax(self.dense.scores(question).astype(np.float64))
        return self.alpha * dense + (1.0 - self.alpha) * lexical

    def search(self, question: str, k: int) -> list[RetrievalHit]:
        """Rank by fused score."""
        return _top_k(self.chunks, self.scores(question), k)


def _subword_lengths(model, texts: list[str]) -> list[int]:  # noqa: ANN001
    """Length of every text in the model's own subword tokens, before any truncation."""
    encoded = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    return [len(ids) for ids in encoded]


def resolve_device(device: str | None = None) -> str:
    """Use the GPU when there is one."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_embedding_model(embedding: str, device: str | None = None):  # noqa: ANN201
    """Load, and cache, a sentence-transformer.

    Loading is cached across the cells of the grid because it is a one-off process cost, not part
    of building an index; timing it into `index_build_seconds` would charge the first cell for
    something every later cell gets free.
    """
    from sentence_transformers import SentenceTransformer

    resolved = resolve_device(device)
    key = (embedding, resolved)
    if key not in _MODEL_CACHE:
        started = time.perf_counter()
        model = SentenceTransformer(EMBEDDINGS[embedding]["model_id"], device=resolved)
        # One throwaway encode before the model is handed out. The first call on CUDA compiles and
        # caches kernels and allocates workspace, which costs seconds; without this the first
        # index built with a given model is charged for all of it and the build times come out
        # nonsense — 115 chunks appearing to take fifteen times longer to encode than 306. The
        # warm-up has to be a full batch of full-length sequences, because that is the shape the
        # kernels are selected for; a two-word string warms up the wrong thing entirely.
        model.encode(
            ["warm up " * 400] * ENCODE_BATCH_SIZE,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        _MODEL_CACHE[key] = model
        log.info("loaded %s on %s in %.1fs", embedding, resolved, time.perf_counter() - started)
    return _MODEL_CACHE[key]
