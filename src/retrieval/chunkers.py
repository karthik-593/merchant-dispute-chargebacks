"""Chunking strategies for the circulars corpus.

Four strategies, held to one rule: **every chunk must be able to cite where it came from.** A
chunk that cannot name its document and page is a retrieval result nobody can verify, which in a
system whose whole premise is provenance is not a chunk at all — so page attribution is carried
through token by token rather than reconstructed afterwards.

The strategies exist to be compared, not to be believed in:

- `whole_document`   — one chunk per record. Cheap, and these documents are short, so it may
                       simply win; that would be worth knowing before building anything cleverer.
- `fixed_size`       — 256 tokens, 25% overlap. The naive baseline any real strategy must beat.
- `sentence`         — sentence-aware packing to a token budget, so chunks end where a thought does.
- `structure_aware`  — split on the document's own boundaries: one reason-code entry, one RGNB
                       row, one numbered clause per chunk.

Tokens are whitespace-delimited words. That is an approximation of what an embedding model would
count, and it is used consistently across all four strategies, so it is fair for comparison even
though it is not exact.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.corpus_ingest import PAGE_MARKER_PATTERN, DocumentRecord

FIXED_CHUNK_TOKENS = 256
FIXED_OVERLAP_RATIO = 0.25
SENTENCE_TARGET_TOKENS = 384
SENTENCE_MAX_TOKENS = 512
STRUCTURE_MAX_TOKENS = 512
STRUCTURE_MIN_TOKENS = 24

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")

# Boundaries a document announces about itself. Order matters: the curated records use the first
# two, and everything else falls back to numbered clauses and headings.
_STRUCTURE_BOUNDARIES = (
    re.compile(r"^Reason code\s+\S+", re.I),  # curated OC 208 §C entries
    re.compile(r"^Flag\s+\S+\s*/\s*Code\s+\S+", re.I),  # curated OC 184B rows
    re.compile(r"^Merchant-type rule\b", re.I),
    re.compile(r"^Auto-loss conditions\b", re.I),
    re.compile(r"^Annexure\s*[-–—]?\s*\w*", re.I),
    re.compile(r"^\s*\d{1,2}[.)]\s+\S"),  # 1) or 1. numbered clauses
    re.compile(r"^\s*[ivxlc]{1,4}[.)]\s+\S", re.I),  # roman-numeral clauses
    re.compile(r"^\s*[A-Z][A-Z /&-]{8,}\s*:?\s*$"),  # ALL CAPS headings
)


class Chunk(BaseModel):
    """One retrievable passage, with the provenance needed to cite it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_id: str
    chunker: str
    text: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_path: str
    source_section: str | None = None
    extraction_method: str
    n_tokens: int = Field(ge=1)
    n_chars: int = Field(ge=1)

    def citation(self) -> str:
        """How this chunk would be cited in an answer."""
        pages = (
            f"p{self.page_start}"
            if self.page_start == self.page_end
            else f"pp{self.page_start}-{self.page_end}"
        )
        section = f", {self.source_section}" if self.source_section else ""
        return f"{self.doc_id}{section}, {pages}"


class Chunker(Protocol):
    """Splits a corpus record into retrievable chunks."""

    name: str

    def chunk(self, record: DocumentRecord) -> list[Chunk]:
        """Split one document."""
        ...


def _strip_page_markers(text: str) -> str:
    """Remove the [[page N]] markers; page numbers travel as metadata, not as body text."""
    return PAGE_MARKER_PATTERN.sub("", text).strip()


def _tokens(text: str) -> list[str]:
    """Whitespace tokens. The same approximation for every strategy."""
    return text.split()


def _build(
    record: DocumentRecord,
    chunker: str,
    index: int,
    pieces: list[tuple[str, int]],
) -> Chunk | None:
    """Assemble a chunk from (text, page) pieces, or None if there is nothing in it."""
    text = " ".join(piece.strip() for piece, _ in pieces if piece.strip()).strip()
    if not text:
        return None
    pages = [page for _, page in pieces]
    return Chunk(
        chunk_id=f"{record.doc_id}::{chunker}::{index:04d}",
        doc_id=record.doc_id,
        chunker=chunker,
        text=text,
        page_start=min(pages),
        page_end=max(pages),
        source_path=record.source_path,
        source_section=record.source_section,
        extraction_method=record.extraction_method,
        n_tokens=len(_tokens(text)),
        n_chars=len(text),
    )


def _page_tokens(record: DocumentRecord) -> list[tuple[str, int]]:
    """Every token in the document, each tagged with the page it came from."""
    return [
        (token, page.page)
        for page in record.pages
        for token in _tokens(_strip_page_markers(page.text))
    ]


def _page_units(record: DocumentRecord, splitter) -> list[tuple[str, int]]:
    """Split each page with `splitter`, keeping the page number on every unit."""
    units: list[tuple[str, int]] = []
    for page in record.pages:
        body = _strip_page_markers(page.text)
        if not body:
            continue
        units += [(unit, page.page) for unit in splitter(body) if unit.strip()]
    return units


def _pack(
    record: DocumentRecord,
    chunker: str,
    units: list[tuple[str, int]],
    target_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    """Greedily pack units into chunks, never exceeding the maximum."""
    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    size = 0
    for unit, page in units:
        length = len(_tokens(unit))
        if current and (size + length > max_tokens or size >= target_tokens):
            chunk = _build(record, chunker, len(chunks), current)
            if chunk:
                chunks.append(chunk)
            current, size = [], 0
        current.append((unit, page))
        size += length
    chunk = _build(record, chunker, len(chunks), current)
    if chunk:
        chunks.append(chunk)
    return chunks


class WholeDocumentChunker:
    """One chunk per record. The document is the unit."""

    name = "whole_document"

    def chunk(self, record: DocumentRecord) -> list[Chunk]:
        """Return the whole document as a single chunk."""
        pieces = [
            (_strip_page_markers(page.text), page.page)
            for page in record.pages
            if _strip_page_markers(page.text)
        ]
        chunk = _build(record, self.name, 0, pieces)
        return [chunk] if chunk else []


class FixedSizeChunker:
    """A sliding window of tokens with overlap.

    Ignores everything the document says about its own structure. The baseline to beat.
    """

    name = "fixed_size"

    def __init__(
        self, size: int = FIXED_CHUNK_TOKENS, overlap_ratio: float = FIXED_OVERLAP_RATIO
    ) -> None:
        """Set the window and its stride."""
        self.size = size
        self.overlap_ratio = overlap_ratio
        self.step = max(1, int(round(size * (1 - overlap_ratio))))

    def chunk(self, record: DocumentRecord) -> list[Chunk]:
        """Slide a fixed window over the document's tokens."""
        tokens = _page_tokens(record)
        if not tokens:
            return []
        chunks: list[Chunk] = []
        for start in range(0, len(tokens), self.step):
            window = tokens[start : start + self.size]
            if not window:
                break
            chunk = _build(record, self.name, len(chunks), window)
            if chunk:
                chunks.append(chunk)
            if start + self.size >= len(tokens):
                break
        return chunks


class SentenceChunker:
    """Packs whole sentences up to a token budget, so a chunk ends where a thought does."""

    name = "sentence"

    def __init__(
        self, target: int = SENTENCE_TARGET_TOKENS, maximum: int = SENTENCE_MAX_TOKENS
    ) -> None:
        """Set the packing budget."""
        self.target = target
        self.maximum = maximum

    def chunk(self, record: DocumentRecord) -> list[Chunk]:
        """Split into sentences and pack them."""
        units = _page_units(record, lambda body: _SENTENCE_END.split(body))
        return _pack(record, self.name, units, self.target, self.maximum)


class StructureAwareChunker:
    """Splits where the document itself splits.

    For the curated records that means one reason-code entry or one RGNB row per chunk, which is
    exactly the unit a question asks about. Elsewhere it falls back to numbered clauses and
    headings, and to blank-line paragraphs where a document announces no structure at all.
    """

    name = "structure_aware"

    def __init__(
        self, maximum: int = STRUCTURE_MAX_TOKENS, minimum: int = STRUCTURE_MIN_TOKENS
    ) -> None:
        """Set the block size bounds."""
        self.maximum = maximum
        self.minimum = minimum

    @staticmethod
    def _is_boundary(line: str) -> bool:
        return any(pattern.match(line) for pattern in _STRUCTURE_BOUNDARIES)

    def _blocks(self, body: str) -> list[str]:
        """Break a page into the blocks the document announces."""
        blocks: list[str] = []
        current: list[str] = []
        for raw in body.splitlines():
            line = raw.rstrip()
            if not line.strip():
                # A blank line only ends a block if one is already open and substantial.
                if current and len(_tokens("\n".join(current))) >= self.minimum:
                    blocks.append("\n".join(current))
                    current = []
                continue
            if self._is_boundary(line) and current:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
        return [block for block in blocks if block.strip()]

    def chunk(self, record: DocumentRecord) -> list[Chunk]:
        """Split on the document's own boundaries, merging fragments and splitting overlong ones."""
        units = _page_units(record, self._blocks)

        # A block longer than the ceiling is split on tokens rather than kept whole; without this
        # a document with no internal structure would come back as one enormous chunk and the
        # strategy would quietly become whole_document.
        bounded: list[tuple[str, int]] = []
        for text, page in units:
            tokens = _tokens(text)
            if len(tokens) <= self.maximum:
                bounded.append((text, page))
                continue
            for start in range(0, len(tokens), self.maximum):
                bounded.append((" ".join(tokens[start : start + self.maximum]), page))

        # Merge blocks that are too small to stand alone, but never across a boundary block.
        chunks: list[Chunk] = []
        current: list[tuple[str, int]] = []
        size = 0
        for text, page in bounded:
            length = len(_tokens(text))
            starts_block = self._is_boundary(text.splitlines()[0] if text else "")
            if current and (starts_block or size + length > self.maximum) and size >= self.minimum:
                chunk = _build(record, self.name, len(chunks), current)
                if chunk:
                    chunks.append(chunk)
                current, size = [], 0
            current.append((text, page))
            size += length
        chunk = _build(record, self.name, len(chunks), current)
        if chunk:
            chunks.append(chunk)
        return chunks


def all_chunkers() -> list[Chunker]:
    """The four strategies under comparison, in the order they are reported."""
    return [
        WholeDocumentChunker(),
        FixedSizeChunker(),
        SentenceChunker(),
        StructureAwareChunker(),
    ]


def chunk_corpus(records: list[DocumentRecord], chunker: Chunker) -> list[Chunk]:
    """Chunk every record with one strategy."""
    return [chunk for record in records for chunk in chunker.chunk(record)]
