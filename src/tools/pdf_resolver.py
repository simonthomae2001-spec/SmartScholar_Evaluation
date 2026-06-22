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
_HTTP_TIMEOUT = (10, 30)
_USER_AGENT = (
    "Mozilla/5.0 (compatible; SmartScholar/0.1; academic research copilot)"
)

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
) -> list[PdfCandidate]:
    """
    Build the ordered list of candidate PDF URLs for a paper (no download).

    Reads ``openAccessPdf`` (SS), ``arxiv_id`` + ``title`` (arXiv), and
    ``doi`` (Unpaywall) from the paper dict. Duplicates (same URL from two
    sources) are removed while preserving priority order.
    """
    candidates: list[PdfCandidate] = []

    # 1. Semantic Scholar open-access PDF (cheap — already in the metadata).
    ss_url = paper.get("openAccessPdf")
    if ss_url:
        candidates.append(PdfCandidate(SOURCE_SEMANTIC_SCHOLAR, ss_url))

    # 2. arXiv — deterministic by ID, else a guarded title search.
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        candidates.append(
            PdfCandidate(SOURCE_ARXIV_ID, _arxiv_url_from_id(arxiv_id))
        )
    else:
        title_url = _arxiv_search_by_title(paper.get("title"))
        if title_url:
            candidates.append(PdfCandidate(SOURCE_ARXIV_TITLE, title_url))

    # 3. Unpaywall — by DOI.
    doi = paper.get("doi")
    if doi:
        email = unpaywall_email or _get_unpaywall_email()
        if email:
            uw_url = _unpaywall_pdf_url(doi, email)
            if uw_url:
                candidates.append(PdfCandidate(SOURCE_UNPAYWALL, uw_url))

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

    candidates = resolve_pdf_candidates(paper, unpaywall_email=unpaywall_email)

    if not candidates:
        # Nothing to try at all (no SS link, no arXiv, no DOI/OA copy).
        return FetchOutcome(
            result=PdfExtractionResult(reason=REASON_NO_URL),
            source=None,
            attempts=[],
        )

    attempts: list[tuple[str, str]] = []
    last_result = PdfExtractionResult(reason=REASON_NO_URL)

    for cand in candidates:
        _log(f"      → trying {cand.source}…")
        cache_key = f"{paper_id}__{cand.source}"
        res = fetch_pdf_text(cand.url, cache_key)
        attempts.append((cand.source, res.reason))

        if res.has_content:
            return FetchOutcome(result=res, source=cand.source, attempts=attempts)
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
            timeout=_HTTP_TIMEOUT,
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
            timeout=_HTTP_TIMEOUT,
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