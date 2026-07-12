"""
pdf_resolver.py — Multi-source open-access PDF discovery for the Ingestor.

Separation of concerns:
- pdf_tool.py   : URL → text          (download, validate, parse; with reasons)
- pdf_resolver  : paper metadata → ordered candidate PDF URLs, plus a
                  convenience that tries them in order until one yields text.

Source priority:
    1. Semantic Scholar ``openAccessPdf``  (free, but often missing in practice)
    2. arXiv   — deterministic by arXiv ID; otherwise a *guarded* title search
                 (we only accept a hit if the returned title is near-identical,
                 so a fuzzy match can't sneak in the wrong paper's PDF).
    3. Unpaywall — by DOI (the canonical OA finder).
    → only if all of these fail does the Ingestor fall back to the abstract.

Why Unpaywall and not OpenAlex as the second source: since 2026-02-13 the
OpenAlex API requires a (free, but registered) key, whereas Unpaywall needs
only a contact e-mail as a query param — lower friction for the same
DOI-keyed best-OA-location data (Unpaywall is in fact now part of OpenAlex).

The Unpaywall e-mail is read from ``data/api_keys/unpaywall_email`` (same
convention as scholar_tool's key file). If the file is absent, the Unpaywall
source is skipped gracefully (logged), never raising.
"""
from __future__ import annotations

import os
import re
from time import sleep
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import requests

from src.tools.pdf_tool import (
    fetch_pdf_text,
    PdfExtractionResult,
    REASON_NO_URL,
)
from src.core.config import get_system_config

# ------------------------------------------------------------------ #
#  Configuration / constants
# ------------------------------------------------------------------ #
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_UNPAYWALL_EMAIL_FILE = os.path.join(
    _PROJECT_ROOT, "data", "api_keys", "unpaywall_email"
)

_ARXIV_API = "http://export.arxiv.org/api/query"
_UNPAYWALL_API = "https://api.unpaywall.org/v2/"
_USER_AGENT = (
    "Mozilla/5.0 (compatible; SmartScholar/0.1; academic research copilot)"
)

def _get_timeout() -> tuple[float, float]:
    """Dynamically reads the network timeout from config.yaml as a connect/read tuple."""
    val = get_system_config().get("network", {}).get("http_timeout_seconds", 10.0)
    return (float(val), float(val) * 3.0)

# arXiv title search is fuzzy: a "close enough" title would otherwise let us
# ingest the WRONG paper's PDF. Require a very high similarity to accept it.
_TITLE_MATCH_THRESHOLD = 0.92

# Source labels — also surfaced as ``pdf_source`` for the Termin-5 metric.
SOURCE_SEMANTIC_SCHOLAR = "semantic_scholar"
SOURCE_ARXIV_ID = "arxiv_id"
SOURCE_ARXIV_TITLE = "arxiv_title"
SOURCE_UNPAYWALL = "unpaywall"

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class PdfCandidate:
    """A single candidate PDF location (where it came from + the URL)."""
    source: str
    url: str


@dataclass
class FetchOutcome:
    """
    Result of trying every candidate in priority order.

    ``result`` is the winning extraction (``has_content``) or, if all
    candidates failed, the last failed extraction (so ``result.reason`` still
    explains the final cause). ``source`` is the winning source or ``None``.
    ``attempts`` lists ``(source, reason)`` per candidate tried — for the
    trace and for debugging which sources are pulling their weight.
    """
    result: PdfExtractionResult
    source: str | None = None
    attempts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return self.result.has_content


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def resolve_pdf_candidates(
    paper: dict,
    *,
    unpaywall_email: str | None = None,
    log=None,
) -> list[PdfCandidate]:
    """
    Build the ordered list of candidate PDF URLs for a paper (no download).

    Reads ``openAccessPdf`` (SS), ``arxiv_id`` + ``title`` (arXiv), and
    ``doi`` (Unpaywall) from the paper dict. Duplicates (same URL from two
    sources) are removed while preserving priority order.

    ``log`` (optional) makes the whole discovery phase visible in the trace:
    which IDs the paper carries, whether the arXiv title search ran and what
    it found, and whether Unpaywall was queried — so a silent "nothing found"
    is no longer indistinguishable from "never tried".
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    candidates: list[PdfCandidate] = []

    arxiv_id = paper.get("arxiv_id")
    doi = paper.get("doi")
    ss_url = paper.get("openAccessPdf")

    _log(
        f"      🔎 IDs: arxiv={arxiv_id or '–'}, doi={doi or '–'}, "
        f"ss_pdf={'yes' if ss_url else 'no'}"
    )

    # 1. Semantic Scholar open-access PDF (cheap — already in the metadata).
    if ss_url:
        candidates.append(PdfCandidate(SOURCE_SEMANTIC_SCHOLAR, ss_url))

    # 2. arXiv — deterministic by ID, else a guarded title search.
    if arxiv_id:
        candidates.append(
            PdfCandidate(SOURCE_ARXIV_ID, _arxiv_url_from_id(arxiv_id))
        )
    else:
        title = paper.get("title")
        if title:
            _log("      🔎 arXiv title search…")
            title_url = _arxiv_search_by_title(title)
            if title_url:
                candidates.append(PdfCandidate(SOURCE_ARXIV_TITLE, title_url))
                _log("      ✓ arXiv: title match accepted")
            else:
                _log("      ✗ arXiv: no confident title match")

    # 3. Unpaywall — by DOI.
    if doi:
        email = unpaywall_email or _get_unpaywall_email()
        if not email:
            _log("      ⚠ Unpaywall skipped (no contact e-mail configured)")
        else:
            _log("      🔎 Unpaywall (DOI lookup)…")
            uw_url = _unpaywall_pdf_url(doi, email)
            if uw_url:
                candidates.append(PdfCandidate(SOURCE_UNPAYWALL, uw_url))
                _log("      ✓ Unpaywall: OA PDF found")
            else:
                _log("      ✗ Unpaywall: no OA PDF for this DOI")
    else:
        _log("      ⚠ Unpaywall skipped (no DOI on this paper)")

    return _dedupe_by_url(candidates)


def fetch_pdf_with_fallback(
    paper: dict,
    paper_id: str,
    *,
    log=None,
    unpaywall_email: str | None = None,
) -> FetchOutcome:
    """
    Resolve candidates and try them in priority order until one yields text.

    Stops at the first candidate whose extraction ``has_content``. If none do,
    returns the last failed result so the caller still sees a precise reason.
    Each source uses its own cache key (``{paper_id}__{source}``) so a cache
    hit always reflects the source that actually produced the PDF.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    arxiv_id = paper.get("arxiv_id")
    doi = paper.get("doi")
    ss_url = paper.get("openAccessPdf")
    title = paper.get("title")
    email = unpaywall_email or _get_unpaywall_email()

    _log(
        f"      🔎 IDs: arxiv={arxiv_id or '–'}, doi={doi or '–'}, "
        f"ss_pdf={'yes' if ss_url else 'no'}"
    )

    # Ordered chain — Semantic Scholar → arXiv → Unpaywall. Each URL is
    # resolved LAZILY (only when reached), so the trace shows every source that
    # was actually consulted AND we stop hitting the arXiv / Unpaywall APIs as
    # soon as one source delivers a usable PDF.
    def _resolve_arxiv() -> str | None:
        if arxiv_id:
            return _arxiv_url_from_id(arxiv_id)
        if title:
            _log("      🔎 arXiv title search…")
            return _arxiv_search_by_title(title)
        return None

    def _resolve_unpaywall() -> str | None:
        if not doi:
            return None
        if not email:
            _log("      ⚠ Unpaywall: no contact e-mail configured")
            return None
        _log("      🔎 Unpaywall DOI lookup…")
        return _unpaywall_pdf_url(doi, email)

    chain = [
        (SOURCE_SEMANTIC_SCHOLAR, lambda: ss_url),
        (SOURCE_ARXIV_ID if arxiv_id else SOURCE_ARXIV_TITLE, _resolve_arxiv),
        (SOURCE_UNPAYWALL, _resolve_unpaywall),
    ]

    attempts: list[tuple[str, str]] = []
    last_result = PdfExtractionResult(reason=REASON_NO_URL)

    for label, resolve in chain:
        url = resolve()
        if not url:
            _log(f"      ✗ {label}: no URL")
            attempts.append((label, REASON_NO_URL))
            continue

        _log(f"      → {label}: downloading…")
        res = fetch_pdf_text(url, f"{paper_id}__{label}")
        attempts.append((label, res.reason))

        if res.has_content:
            _log(f"      ✓ {label}: PDF ok")
            return FetchOutcome(result=res, source=label, attempts=attempts)

        _log(f"      ✗ {label}: {res.reason}")
        last_result = res

    return FetchOutcome(result=last_result, source=None, attempts=attempts)


# ------------------------------------------------------------------ #
#  Source: arXiv
# ------------------------------------------------------------------ #

def _arxiv_url_from_id(arxiv_id: str) -> str:
    """Build the canonical arXiv PDF URL from an arXiv ID (e.g. '1706.03762')."""
    aid = re.sub(r"(?i)^arxiv:", "", (arxiv_id or "").strip()).strip()
    return f"https://arxiv.org/pdf/{aid}"


def _arxiv_search_by_title(title: str | None) -> str | None:
    """
    Find an arXiv PDF by title via the arXiv Atom API.

    Guarded: only returns a URL if the top hit's title is near-identical to
    the query title (>= _TITLE_MATCH_THRESHOLD). Never raises — returns None
    on any network / parse / no-match condition.
    """
    if not title or not title.strip():
        return None

    params = {"search_query": f'ti:"{title}"', "start": 0, "max_results": 1}
    try:
        sleep(0.4)  # be polite to the arXiv API
        resp = requests.get(
            _ARXIV_API,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_get_timeout(),
        )
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return None

    entry = root.find("a:entry", _ATOM_NS)
    if entry is None:
        return None

    found_title = (
        entry.findtext("a:title", default="", namespaces=_ATOM_NS) or ""
    ).strip()
    if _title_similarity(title, found_title) < _TITLE_MATCH_THRESHOLD:
        return None  # too risky — likely a different paper

    # Preferred: the explicit PDF <link title="pdf">.
    for link in entry.findall("a:link", _ATOM_NS):
        if link.get("title") == "pdf" and link.get("href"):
            return link.get("href")

    # Fallback: derive the PDF URL from the entry id (…/abs/<id> → …/pdf/<id>).
    id_url = entry.findtext("a:id", default="", namespaces=_ATOM_NS) or ""
    m = re.search(r"arxiv\.org/abs/(.+)$", id_url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return None


# ------------------------------------------------------------------ #
#  Source: Unpaywall
# ------------------------------------------------------------------ #

def _get_unpaywall_email() -> str | None:
    """Read the Unpaywall contact e-mail from data/api_keys/unpaywall_email."""
    if not os.path.exists(_UNPAYWALL_EMAIL_FILE):
        return None
    try:
        with open(_UNPAYWALL_EMAIL_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _unpaywall_pdf_url(doi: str, email: str) -> str | None:
    """
    Ask Unpaywall for the best OA PDF URL for a DOI. Never raises.

    Returns ``best_oa_location.url_for_pdf`` (or ``.url``), else scans
    ``oa_locations`` for any direct PDF link, else None.
    """
    if not doi or not email:
        return None

    clean_doi = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", doi.strip())
    try:
        resp = requests.get(
            f"{_UNPAYWALL_API}{clean_doi}",
            params={"email": email},
            headers={"User-Agent": _USER_AGENT},
            timeout=_get_timeout(),
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if not data.get("is_oa"):
        return None

    best = data.get("best_oa_location") or {}
    url = best.get("url_for_pdf") or best.get("url")
    if url:
        return url

    for loc in data.get("oa_locations") or []:
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _norm_title(t: str) -> str:
    """Lowercase + collapse non-alphanumerics, for robust title comparison."""
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def _dedupe_by_url(candidates: list[PdfCandidate]) -> list[PdfCandidate]:
    """Drop later candidates whose URL already appeared (keep priority order)."""
    seen: set[str] = set()
    out: list[PdfCandidate] = []
    for c in candidates:
        if c.url not in seen:
            seen.add(c.url)
            out.append(c)
    return out