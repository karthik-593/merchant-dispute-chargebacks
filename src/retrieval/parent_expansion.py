"""Small-to-big retrieval: embed at row grain, return the enclosing parent.

The index is unchanged — rows are still what gets embedded and ranked, because row grain is where
the precision is. What changes is the unit *returned*: on a hit, the row's parent is resolved and
returned in its place.

**How this can move a rank at all**, which is not obvious and is the whole mechanism. Expansion
does not rescore anything. It changes the ranking only by *de-duplicating*: several rows of one
table collapse to a single parent, and that parent inherits the best rank any of its rows achieved.
So a target row buried at 200 becomes reachable the moment any *sibling* of it ranks well, because
the parent returned for that sibling contains the target too.

**That is also the trap.** A parent containing the whole table satisfies an anchor test without the
retriever ever having isolated the row — which is exactly the B3 ruler defect that makes
`whole_document` look good on near-duplicate queries. A rule-hit on a full-table parent means "the
table was found", not "the row was found". The bounded window exists so that trade-off is
measurable rather than hidden: `±N rows` returns the target plus a few neighbours, so a hit still
means something about locality.

Three modes, and the baseline is one of them:

- `none`     — return the row. The baseline.
- `window`   — the row plus N chunks either side, within the same document.
- `document` — every chunk of the document. For the curated tables that is exactly the table.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.chunkers import Chunk
from src.retrieval.retrievers import RetrievalHit

NONE = "none"
WINDOW = "window"
DOCUMENT = "document"
MODES = (NONE, WINDOW, DOCUMENT)

DEFAULT_WINDOW = 2


@dataclass
class ParentExpander:
    """Resolves a retrieved row to the unit that will actually be returned."""

    chunks: list[Chunk]
    mode: str = NONE
    window: int = DEFAULT_WINDOW

    def __post_init__(self) -> None:
        """Index the chunks by document so a row can find its neighbours."""
        if self.mode not in MODES:
            raise ValueError(f"unknown expansion mode {self.mode!r}; expected one of {MODES}")
        self._by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk)
        self._position = {
            chunk.chunk_id: (chunk.doc_id, index)
            for chunks in self._by_doc.values()
            for index, chunk in enumerate(chunks)
        }

    def _span(self, chunk: Chunk) -> tuple[str, int, int]:
        """The parent's identity: which document, and which slice of it."""
        doc_id, index = self._position[chunk.chunk_id]
        siblings = self._by_doc[doc_id]
        if self.mode == NONE:
            return doc_id, index, index
        if self.mode == DOCUMENT:
            return doc_id, 0, len(siblings) - 1
        return doc_id, max(0, index - self.window), min(len(siblings) - 1, index + self.window)

    def _build(self, span: tuple[str, int, int], rank: int, score: float) -> RetrievalHit:
        """Assemble the parent as one chunk, carrying the provenance of the rows in it."""
        doc_id, start, end = span
        members = self._by_doc[doc_id][start : end + 1]
        text = "\n".join(member.text for member in members)
        first = members[0]
        parent = Chunk(
            chunk_id=f"{doc_id}::parent::{start:04d}-{end:04d}",
            doc_id=doc_id,
            chunker=f"{first.chunker}+{self.mode}",
            text=text,
            page_start=min(m.page_start for m in members),
            page_end=max(m.page_end for m in members),
            source_path=first.source_path,
            source_section=first.source_section,
            extraction_method=first.extraction_method,
            n_tokens=len(text.split()),
            n_chars=len(text),
        )
        return RetrievalHit(rank=rank, score=score, chunk=parent)

    def expand(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Map each hit to its parent, de-duplicate, and renumber from 1.

        De-duplication is the part that moves ranks: two rows of the same table produce one
        parent, and it keeps the better of their ranks. Order is otherwise preserved, so a parent's
        position is the position of its best-ranking member row.
        """
        seen: dict[tuple[str, int, int], RetrievalHit] = {}
        for hit in hits:
            span = self._span(hit.chunk)
            if span not in seen:
                seen[span] = self._build(span, rank=len(seen) + 1, score=hit.score)
        return [
            RetrievalHit(rank=rank, score=hit.score, chunk=hit.chunk)
            for rank, hit in enumerate(seen.values(), start=1)
        ]


class ExpandingRetriever:
    """A row-grain retriever whose results are returned as parents."""

    def __init__(self, retriever, expander: ParentExpander, probe: int = 200) -> None:
        """Wrap a retriever; `probe` is how deep the row-grain search runs before collapsing."""
        self.retriever = retriever
        self.expander = expander
        self.probe = probe
        self.chunks = retriever.chunks
        self.name = f"{retriever.name}+{expander.mode}"

    def search(self, question: str, k: int) -> list[RetrievalHit]:
        """Retrieve rows deep, collapse to parents, return the top k parents.

        The row-grain search has to run deeper than `k`, because k parents can absorb many more
        than k rows - that collapse is the point.
        """
        rows = self.retriever.search(question, max(self.probe, k))
        return self.expander.expand(rows)[:k]
