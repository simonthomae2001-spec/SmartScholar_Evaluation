"""
pdf_tool.py — Stateless PDF retrieval & text extraction tool.

This tool is intentionally "dumb": it takes a PDF URL plus a paper ID and
returns the extracted text, preserving page boundaries. It knows nothing
about research profiles, chunking, or ChromaDB — that logic lives in the
IngestorAgent.

Design decision A (locked):
- Input:  explicit ``url`` and ``paper_id`` (NOT the full paper dict).
- Output: a ``PdfExtractionResult`` whose ``pages`` is a list of
          ``(page_number, page_text)`` tuples with **1-based** page numbers.
          Callers that don't need pages flatten them via ``full_text()``.

Decision B: download — timeout, streamed read with a hard size cap,
            explicit User-Agent, status check.
Decision C: validation — verify the bytes really are a PDF (magic marker),
            not an HTML error / paywall page.
Decision D: caching — persist validated raw PDF bytes under data/pdf_cache,
            keyed by paper_id, so re-runs skip the download.
Decision E: parsing — extract per-page text via PyMuPDF.
Decision F: reason threading — every attempt now carries a machine-readable
            ``reason`` (paywall / timeout / not_a_pdf / scanned_no_text / …)
            on the returned ``PdfExtractionResult``, so the IngestorAgent can
            surface *why* a PDF failed in the trace and as a per-paper metric
            (feeds observability + the Termin-5 PDF-coverage measurement).

NOT in scope here: cleanup of headers/footers/bibliography (Phase 4b) and
chunking. Just: URL in, raw per-page text + a reason out.

Requires: PyMuPDF  (``pip install PyMuPDF``; imported as ``fitz``).
"""
from __future__ import annotations

import os
import re
from time import sleep
from dataclasses import dataclass, field

import requests
import fitz  # PyMuPDF


# ------------------------------------------------------------------ #
#  Tunable download parameters (single source of truth for the tool)
# ------------------------------------------------------------------ #
_USER_AGENT = (
    "Mozilla/5.0 (compatible; SmartScholar/0.1; academic research copilot)"
)
_REQUEST_TIMEOUT = (10, 30)            # (connect, read) seconds
_MAX_PDF_BYTES = 30 * 1024 * 1024      # 30 MB hard cap
_DOWNLOAD_CHUNK = 64 * 1024            # 64 KB per streamed chunk
_MAX_RETRIES = 3                       # only transient failures are retried (F)

# Decision D — local PDF cache, alongside data/chroma_db and data/api_keys.
# Path is derived from this file's location (project root), not the CWD.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "pdf_cache")


# ------------------------------------------------------------------ #
#  Decision F — reason vocabulary (single source of truth)
# ------------------------------------------------------------------ #
# Exactly one of these is always set on PdfExtractionResult.reason.
# "ok" means usable text was extracted; everything else explains the failure
# precisely enough to drive both the UI trace and the Termin-5 metrics.
REASON_OK = "ok"
REASON_NO_URL = "no_url"                 # Semantic Scholar gave no OA link
REASON_PAYWALL = "paywall"               # HTTP 403 (blocked / behind a wall)
REASON_NOT_FOUND = "not_found"           # HTTP 404 (dead link)
REASON_HTTP_ERROR = "http_error"         # any other non-200 status
REASON_TIMEOUT = "timeout"               # connect/read timeout
REASON_CONNECTION = "connection_error"   # DNS / refused / malformed URL / …
REASON_OVERSIZE = "oversize"             # exceeded the size cap
REASON_NOT_A_PDF = "not_a_pdf"           # 200 OK, but bytes are HTML/captcha
REASON_UNREADABLE = "unreadable_pdf"     # encrypted / corrupt (fitz failed)
REASON_SCANNED = "scanned_no_text"       # opened fine, but image-only (no text)

# Transient → worth a retry. Everything else is permanent → stop immediately.
_TRANSIENT_REASONS = {REASON_TIMEOUT, REASON_CONNECTION, REASON_HTTP_ERROR}


@dataclass
class PdfExtractionResult:
    """
    Result container for a single PDF extraction attempt.

    Attributes
    ----------
    pages : list[tuple[int, str]]
        One ``(page_number, text)`` tuple per page that contains text.
        Page numbers are **1-based** (page 1 = first page). An empty list
        means nothing usable was extracted.
    reason : str
        Decision F — a machine-readable explanation of how the attempt ended.
        One of the ``REASON_*`` constants above. ``REASON_OK`` iff ``pages``
        is non-empty. On every failure path it names the precise cause
        (paywall, timeout, not_a_pdf, scanned_no_text, …).
    """

    pages: list[tuple[int, str]] = field(default_factory=list)
    reason: str = REASON_NO_URL  # overwritten on every path of fetch_pdf_text

    def full_text(self, separator: str = "\n\n") -> str:
        """Flatten all pages into one string (for FAST / MEDIUM callers)."""
        return separator.join(text for _, text in self.pages)

    @property
    def has_content(self) -> bool:
        """True if at least one page produced non-empty text."""
        return any(text.strip() for _, text in self.pages)


def fetch_pdf_text(url: str | None, paper_id: str) -> PdfExtractionResult:
    """
    Download a PDF and extract its text, preserving page boundaries.

    Contract: this function NEVER raises for an expected failure (no URL,
    download error, not a PDF, parse error, empty text). It returns a
    content-less ``PdfExtractionResult`` whose ``reason`` names the cause
    instead. The caller decides what to do (silent fallback to abstract-only).

    Parameters
    ----------
    url : str | None
        The open-access PDF URL. ``None`` → immediate ``no_url`` result.
    paper_id : str
        Stable identifier; used as the cache key (decision D).

    Returns
    -------
    PdfExtractionResult
        ``pages`` populated + ``reason == "ok"`` on success; empty ``pages``
        + a specific failure ``reason`` otherwise.
    """
    result = PdfExtractionResult()

    # No open-access link (e.g. paywall) → nothing to do.
    if not url:
        result.reason = REASON_NO_URL
        return result

    # --- D wraps B + C: cache-aware retrieval of validated PDF bytes -----
    pdf_bytes, reason = _get_pdf_bytes(url, paper_id)
    if pdf_bytes is None:
        result.reason = reason
        return result

    # --- E: parse with PyMuPDF into 1-based (page_no, text) tuples -------
    pages, reason = _extract_pages(pdf_bytes)
    result.pages = pages
    result.reason = reason
    return result


# ------------------------------------------------------------------ #
#  Decision D — cache-aware retrieval (orchestrates cache + B + C)
# ------------------------------------------------------------------ #

def _get_pdf_bytes(url: str, paper_id: str) -> tuple[bytes | None, str]:
    """
    Return valid PDF bytes + a reason, using the local cache when possible.

    Order:
      1. Cache hit  → (cached bytes, "ok")   — validated when written.
      2. Cache miss → download (B, with retry), validate (C), cache (D).

    Only validated PDFs are ever written to the cache, so a cached HTML
    error page can never poison later runs. Never raises.
    """
    cached = _read_cache(paper_id)
    if cached is not None:
        return cached, REASON_OK

    raw, reason = _download_with_retries(url)
    if raw is None:
        return None, reason

    # Only cache real PDFs (C) — never persist an HTML error / paywall page.
    if not _looks_like_pdf(raw):
        return None, REASON_NOT_A_PDF

    _write_cache(paper_id, raw)
    return raw, REASON_OK


def _download_with_retries(url: str) -> tuple[bytes | None, str]:
    """
    Download raw bytes, retrying ONLY transient failures (decision F).

    A permanent failure (paywall, dead link) is hopeless to retry, so we stop
    at once instead of burning ~4 polite delays on a 403. Returns the bytes
    and ``REASON_OK`` on success, or ``None`` and the last failure reason.
    """
    reason = REASON_CONNECTION
    for _ in range(_MAX_RETRIES):
        raw, reason = _download_pdf(url)
        if raw is not None:
            return raw, REASON_OK
        if reason not in _TRANSIENT_REASONS:
            break  # permanent — retrying won't change anything
    return None, reason


# ------------------------------------------------------------------ #
#  Decision B — download
# ------------------------------------------------------------------ #

def _download_pdf(url: str) -> tuple[bytes | None, str]:
    """
    Download raw PDF bytes over HTTP, returning ``(bytes, reason)``.

    Robustness measures:
    - connect/read timeout, so a hanging server can't block the pipeline;
    - streamed download with a hard size cap, so a huge file can't exhaust
      memory (we abort as soon as the cap is exceeded);
    - an explicit User-Agent, since some servers reject the default
      ``python-requests`` UA with a 403.

    Decision F: instead of collapsing every failure to ``None``, the distinct
    causes are named — paywall (403), not_found (404), http_error (other
    non-200), oversize, timeout, connection_error. ``REASON_OK`` on HTTP 200
    with bytes read. Never raises.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf"}

    sleep(0.5)  # politeness delay before hitting the server
    try:
        with requests.get(
            url,
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True,
        ) as resp:
            if resp.status_code == 403:
                return None, REASON_PAYWALL      # blocked / behind a wall
            if resp.status_code == 404:
                return None, REASON_NOT_FOUND    # dead link
            if resp.status_code != 200:
                return None, REASON_HTTP_ERROR   # 5xx, 429, … (transient)

            buffer = bytearray()
            for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                buffer.extend(chunk)
                if len(buffer) > _MAX_PDF_BYTES:
                    # Abort early — don't pull an oversized file into memory.
                    return None, REASON_OVERSIZE

            return bytes(buffer), REASON_OK

    except requests.Timeout:
        # Must come before RequestException — Timeout is a subclass of it.
        return None, REASON_TIMEOUT
    except requests.RequestException:
        # ConnectionError, TooManyRedirects, MissingSchema, InvalidURL, ...
        return None, REASON_CONNECTION


# ------------------------------------------------------------------ #
#  Decision C — validation (magic bytes)
# ------------------------------------------------------------------ #

def _looks_like_pdf(data: bytes) -> bool:
    """
    Verify the downloaded bytes are actually a PDF.

    A server can return HTTP 200 with an HTML error / paywall / captcha page
    instead of the PDF. We do NOT trust the Content-Type header (often wrong);
    instead we look for the ``%PDF`` magic marker.

    We scan the first 1 KiB rather than requiring the marker at byte 0,
    because some otherwise-valid PDFs carry a few leading bytes (BOM /
    whitespace). This matches how lenient PDF readers (incl. PyMuPDF) locate
    the header, while still rejecting HTML pages (which have no ``%PDF`` early).
    """
    if not data:
        return False
    return b"%PDF" in data[:1024]


# ------------------------------------------------------------------ #
#  Decision D — cache I/O helpers (best-effort, never raise)
# ------------------------------------------------------------------ #

def _cache_path(paper_id: str) -> str | None:
    """
    Map a ``paper_id`` to a safe cache file path, or ``None`` if the id is
    unusable (empty after sanitising). Only ``A-Z a-z 0-9 . _ -`` survive;
    everything else becomes ``_`` so the id is always a safe filename.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", paper_id or "").strip("._")
    if not safe:
        return None
    return os.path.join(_CACHE_DIR, f"{safe}.pdf")


def _read_cache(paper_id: str) -> bytes | None:
    """Return cached PDF bytes for this paper_id, or None on miss / error."""
    path = _cache_path(paper_id)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        # Unreadable cache file → treat as a miss and re-download.
        return None


def _write_cache(paper_id: str, data: bytes) -> None:
    """
    Persist PDF bytes for this paper_id. Best-effort: never raises.

    Writes atomically (temp file + ``os.replace``) so an interrupted write
    can't leave a half-written, corrupt cache file behind.
    """
    path = _cache_path(paper_id)
    if not path:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)  # atomic on the same filesystem
    except OSError:
        # Caching is only an optimisation — a write failure must not break
        # ingestion. Just skip persisting this one.
        pass


# ------------------------------------------------------------------ #
#  Decision E — parsing (PyMuPDF / fitz)
# ------------------------------------------------------------------ #

def _extract_pages(pdf_bytes: bytes) -> tuple[list[tuple[int, str]], str]:
    """
    Extract per-page text from raw PDF bytes using PyMuPDF (fitz).

    - Opens the PDF from the in-memory byte stream (no temp file needed).
    - Returns one ``(page_number, text)`` tuple per page that contains text,
      with **1-based** page numbers (decision A).
    - Pages whose extracted text is empty / whitespace are skipped.

    Decision F: distinguishes the two empty-output cases that used to look
    identical —
        * ``unreadable_pdf``  : fitz could not open it (encrypted / corrupt);
        * ``scanned_no_text`` : opened fine, but no extractable text at all
                                (image-only / scanned — no OCR here).
    ``REASON_OK`` with the pages otherwise. Never raises.

    No cleanup (headers / footers / bibliography) and no chunking happen
    here — that is Phase 4b / the IngestorAgent's job. Raw per-page text only.
    """
    pages: list[tuple[int, str]] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_number, page in enumerate(doc, start=1):  # 1-based
                text = page.get_text().strip()
                if text:
                    pages.append((page_number, text))
    except Exception:
        # Encrypted, corrupt, or otherwise unreadable PDF.
        return [], REASON_UNREADABLE

    if not pages:
        # Opened, but every page was image-only / empty → no retrievable text.
        return [], REASON_SCANNED

    return pages, REASON_OK