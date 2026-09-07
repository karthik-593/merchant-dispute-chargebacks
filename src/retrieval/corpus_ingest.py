"""Ingests the NPCI/RBI circulars into clean, chunk-ready text with provenance.

This is the **RAG target**, and it is deliberately separate from the evidence store: finding the
rule that governs a dispute is a search problem, while finding a dispute's artifacts is a keyed
join. The two never share an index.

Most of these circulars are image scans with no text layer, so extraction is two-stage. Each page
is tried as text first; a page that yields almost nothing is rasterised and sent to OCR. The
method is recorded per page and rolled up per document, so a downstream chunk can always say
whether its text came from a PDF text layer or from a character recogniser — which matters when
judging whether a quoted rule is trustworthy.

Nothing here chunks or embeds. The output is one clean record per circular with page boundaries
intact, so a later chunker can cite `doc_id` plus page.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Protocol

import numpy as np
import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from src.config import project_path
from src.logging_setup import get_logger

log = get_logger(__name__)

CORPUS_FILE = "circulars.jsonl"
PAGE_MARKER = "[[page {page}]]"
PAGE_MARKER_PATTERN = re.compile(r"^\[\[page (\d+)\]\]$", re.MULTILINE)

# A page yielding fewer characters than this is treated as having no usable text layer.
MIN_TEXT_CHARS_PER_PAGE = 120
# A whole document under this after extraction is flagged for a human to look at.
LOW_YIELD_DOC_CHARS = 400
# Rasterisation resolution for OCR. 300 dpi is the usual floor for reliable recognition.
OCR_DPI = 300
# Printed booklets carry intentionally blank pages. Below this share of non-white pixels a
# page is treated as genuinely blank rather than as an OCR failure - the distinction matters,
# because one is fine and the other means a rule went missing.
BLANK_INK_FRACTION = 0.0005
BLANK_CHECK_DPI = 72

EXTRACTION_TEXT = "text"
EXTRACTION_OCR = "ocr"
EXTRACTION_BLANK = "blank"
EXTRACTION_CURATED = "curated"
EXTRACTION_MIXED = "mixed"
EXTRACTION_FAILED = "failed"

# Circular references as they appear in the documents themselves.
REFERENCE_PATTERNS = (
    re.compile(r"NPCI\s*[/-]\s*UPI\s*[/-]\s*OC\s*(?:No\.?)?\s*[-–]?\s*([0-9]+\s*[A-C]?)", re.I),
    re.compile(r"UPI\s*[/-]\s*OC\s*(?:No\.?)?\s*[-–]?\s*([0-9]+\s*[A-C]?)", re.I),
    re.compile(r"\bOC\s*(?:No\.?)?\s*[-–]?\s*([0-9]{2,3}\s*[A-C]?)\b", re.I),
    re.compile(r"(RBI\s*/\s*[0-9]{4}\s*-\s*[0-9]{2,4}\s*/\s*[0-9]+)", re.I),
    re.compile(r"(RBI\s*/\s*DPSS\s*/\s*[0-9]{4}\s*-\s*[0-9]{2,4}\s*/\s*[0-9]+)", re.I),
)


class OcrUnavailableError(RuntimeError):
    """Raised when a page needs OCR and no working OCR backend is configured."""


class OcrBackend(Protocol):
    """Turns a rendered page image into text."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Whether the backend can run, and why not if it cannot."""
        ...

    def image_to_text(self, png_bytes: bytes) -> str:
        """Recognise the text in a rendered page."""
        ...


# Where the Tesseract executable usually lands on each platform when it is not on PATH.
TESSERACT_CANDIDATES = (
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    Path.home() / "AppData/Local/Tesseract-OCR/tesseract.exe",
    Path("/usr/bin/tesseract"),
    Path("/usr/local/bin/tesseract"),
    Path("/opt/homebrew/bin/tesseract"),
)


def find_tesseract() -> Path | None:
    """Locate the Tesseract executable, on PATH or in the usual install locations.

    The Windows installer does not add itself to PATH, so looking only there would report the
    engine as missing on a machine where it is installed and working.
    """
    found = shutil.which("tesseract")
    if found:
        return Path(found)
    return next((path for path in TESSERACT_CANDIDATES if path.is_file()), None)


class TesseractBackend:
    """OCR through Tesseract, driven by pytesseract.

    pytesseract is only a wrapper: it shells out to the `tesseract` executable, which has to be
    installed separately. `available()` reports that plainly rather than letting the failure
    surface later as an empty page.
    """

    name = "tesseract"

    def __init__(self, language: str = "eng", executable: Path | None = None) -> None:
        """Record the recognition language and locate the engine."""
        self.language = language
        self.executable = executable or find_tesseract()

    def _configure(self) -> None:
        """Point pytesseract at the located executable."""
        import pytesseract

        if self.executable is not None:
            pytesseract.pytesseract.tesseract_cmd = str(self.executable)

    def available(self) -> tuple[bool, str]:
        """Check that both the wrapper and the underlying binary are present."""
        try:
            import pytesseract
        except ImportError:
            return False, "pytesseract is not installed"
        if self.executable is None:
            return False, "the tesseract executable was not found on PATH or in the usual places"
        try:
            self._configure()
            version = pytesseract.get_tesseract_version()
        except Exception as error:  # noqa: BLE001 - any failure here means unusable
            return False, f"the tesseract executable is not callable: {error}"
        return True, f"tesseract {version} at {self.executable}"

    def image_to_text(self, png_bytes: bytes) -> str:
        """Recognise one rendered page."""
        import io

        import pytesseract
        from PIL import Image

        self._configure()
        with Image.open(io.BytesIO(png_bytes)) as image:
            return pytesseract.image_to_string(image, lang=self.language)


class NoOcrBackend:
    """Refuses to OCR. Used when scans should be reported rather than silently skipped."""

    name = "none"

    def available(self) -> tuple[bool, str]:
        """Never available, by design."""
        return False, "no OCR backend configured"

    def image_to_text(self, png_bytes: bytes) -> str:
        """Always raises."""
        raise OcrUnavailableError("no OCR backend configured")


class PageRecord(BaseModel):
    """One page of a circular, with how its text was obtained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: int = Field(ge=1)
    extraction_method: str
    char_count: int = Field(ge=0)
    text: str


class DocumentRecord(BaseModel):
    """One circular, normalised and ready to be chunked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    circular_ref: str | None = Field(
        default=None, description="reference parsed from the document text, not the filename"
    )
    title: str | None = Field(default=None, description="first substantive line of the document")
    source_path: str
    source_sha256: str
    source_section: str | None = Field(
        default=None,
        description="section of the circular this record covers, e.g. the evidence table",
    )
    supersedes_doc_id: str | None = Field(
        default=None,
        description=(
            "a record this one is preferred over for retrieval; set on curated tables that "
            "replace an OCR-flattened rendering, which is kept but no longer the best source"
        ),
    )
    extraction_method: str = Field(description="text, ocr, mixed, curated or failed")
    page_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    pages: list[PageRecord]
    text: str = Field(description="full document text with [[page N]] markers")

    @property
    def is_low_yield(self) -> bool:
        """Whether the document produced so little text that a human should look at it."""
        return self.char_count < LOW_YIELD_DOC_CHARS


def corpus_dir() -> Path:
    """Directory holding the ingested corpus."""
    return project_path("data") / "corpus"


def circulars_dir() -> Path:
    """Directory holding the source PDFs."""
    return project_path("refs")


def sha256_of(path: Path) -> str:
    """Content hash of the source PDF, so a record can be tied to the exact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_doc_id(path: Path) -> str:
    """A stable, readable id derived from the filename."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()
    return re.sub(r"_+", "_", stem)[:80]


def parse_circular_ref(text: str) -> str | None:
    """Pull the circular reference out of the document's own text.

    Filenames are download artefacts and sometimes disagree with the document; the reference
    printed on the page is the authority.
    """
    head = text[:4000]
    for pattern in REFERENCE_PATTERNS:
        match = pattern.search(head)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            if pattern.pattern.startswith("(RBI") or value.upper().startswith("RBI"):
                return value.upper().replace(" ", "")
            return f"OC {value.upper().replace(' ', '')}"
    return None


def parse_title(text: str) -> str | None:
    """The first line that reads like a subject or a substantive heading."""
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip() and not PAGE_MARKER_PATTERN.match(line.strip())
    ]
    for line in lines:
        match = re.match(r"^(?:sub(?:ject)?|sub)\s*[:.-]\s*(.+)$", line, re.I)
        if match and len(match.group(1)) > 10:
            return match.group(1)[:200]
    for line in lines:
        if 15 <= len(line) <= 200 and not line.lower().startswith(("to,", "madam", "dear")):
            return line[:200]
    return None


def page_ink_fraction(page: pymupdf.Page, dpi: int = BLANK_CHECK_DPI) -> float:
    """Share of non-white pixels on a rendered page.

    Used to tell an intentionally blank page apart from one whose content OCR failed to read.
    """
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    if samples.size == 0:
        return 0.0
    return float((samples < 200).mean())


def render_page_png(page: pymupdf.Page, dpi: int = OCR_DPI) -> bytes:
    """Rasterise a page for OCR.

    Done through PyMuPDF rather than pdf2image so no poppler binary is needed.
    """
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def extract_document(
    path: Path,
    ocr_backend: OcrBackend | None = None,
    min_text_chars: int = MIN_TEXT_CHARS_PER_PAGE,
) -> DocumentRecord:
    """Extract one circular, falling back to OCR page by page.

    Raises:
        OcrUnavailableError: if a page has no usable text layer and OCR cannot run. Failing here
            is deliberate: a blank record for a scanned circular would silently remove a
            governing rule from the corpus.
    """
    backend = ocr_backend or NoOcrBackend()
    pages: list[PageRecord] = []
    needs_ocr: list[int] = []

    with pymupdf.open(str(path)) as document:
        for number, page in enumerate(document, start=1):
            text = (page.get_text() or "").strip()
            if len(text) >= min_text_chars:
                pages.append(
                    PageRecord(
                        page=number,
                        extraction_method=EXTRACTION_TEXT,
                        char_count=len(text),
                        text=text,
                    )
                )
                continue

            if page_ink_fraction(page) < BLANK_INK_FRACTION:
                # A genuinely blank page. Recording it as blank keeps the page numbering honest
                # without pretending OCR read something that was never there.
                pages.append(
                    PageRecord(
                        page=number,
                        extraction_method=EXTRACTION_BLANK,
                        char_count=0,
                        text="",
                    )
                )
                continue

            usable, reason = backend.available()
            if not usable:
                needs_ocr.append(number)
                pages.append(
                    PageRecord(
                        page=number,
                        extraction_method=EXTRACTION_FAILED,
                        char_count=0,
                        text="",
                    )
                )
                continue

            recognised = backend.image_to_text(render_page_png(page)).strip()
            pages.append(
                PageRecord(
                    page=number,
                    extraction_method=EXTRACTION_OCR,
                    char_count=len(recognised),
                    text=recognised,
                )
            )

    if needs_ocr:
        _, reason = backend.available()
        raise OcrUnavailableError(
            f"{path.name}: pages {needs_ocr} have no text layer and OCR is unavailable ({reason})"
        )

    methods = {page.extraction_method for page in pages}
    if EXTRACTION_FAILED in methods:
        method = EXTRACTION_FAILED
    else:
        # Blank pages say nothing about how a document was read, so they must not make an
        # otherwise born-digital document look like a scan.
        substantive = methods - {EXTRACTION_BLANK}
        if substantive == {EXTRACTION_TEXT}:
            method = EXTRACTION_TEXT
        elif substantive == {EXTRACTION_OCR}:
            method = EXTRACTION_OCR
        elif not substantive:
            method = EXTRACTION_BLANK
        else:
            method = EXTRACTION_MIXED

    body = "\n\n".join(f"{PAGE_MARKER.format(page=page.page)}\n{page.text}" for page in pages)
    return DocumentRecord(
        doc_id=make_doc_id(path),
        circular_ref=parse_circular_ref(body),
        title=parse_title(body),
        source_path=str(path.relative_to(project_path("configs").parent)).replace("\\", "/"),
        source_sha256=sha256_of(path),
        extraction_method=method,
        page_count=len(pages),
        char_count=sum(page.char_count for page in pages),
        pages=pages,
        text=body,
    )


def source_pdfs(directory: Path | None = None) -> list[Path]:
    """Every circular in the corpus directory, in stable order."""
    return sorted((directory or circulars_dir()).glob("*.pdf"))


def ingest_corpus(
    directory: Path | None = None,
    ocr_backend: OcrBackend | None = None,
    on_error: str = "raise",
) -> tuple[list[DocumentRecord], list[str]]:
    """Ingest every circular. Returns the records and any per-document failures.

    `on_error="collect"` keeps going after a failure so the whole picture can be reported at
    once, which is more useful than stopping at the first scan when OCR is missing.
    """
    records: list[DocumentRecord] = []
    failures: list[str] = []
    for path in source_pdfs(directory):
        try:
            records.append(extract_document(path, ocr_backend))
        except Exception as error:  # noqa: BLE001 - reported per document
            if on_error == "raise":
                raise
            failures.append(f"{path.name}: {error}")
            log.warning("ingestion failed for %s: %s", path.name, error)
    return records, failures


def write_corpus(records: list[DocumentRecord], directory: Path | None = None) -> Path:
    """Write the corpus as one JSON object per document."""
    root = directory or corpus_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / CORPUS_FILE
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
    return path


def read_corpus(directory: Path | None = None) -> list[DocumentRecord]:
    """Read the corpus back into validated records."""
    path = (directory or corpus_dir()) / CORPUS_FILE
    with path.open(encoding="utf-8") as handle:
        return [DocumentRecord.model_validate_json(line) for line in handle if line.strip()]
