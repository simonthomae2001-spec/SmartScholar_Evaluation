"""
ingestor_agent.py — Knowledge ingestion.

The Ingestor is the "librarian": for each finalised paper it ingests text
into ChromaDB at the profile's read_depth, and attaches a LEAN reference
(chunk_ids + ingestion_status + ingested_depth + pdf_reason + pdf_source) to
the paper dict. Large content lives ONLY in ChromaDB, never in graph state.

PDF acquisition (Thema A): instead of trusting Semantic Scholar's (often
missing) openAccessPdf alone, the Ingestor asks pdf_resolver to try a chain
of sources — Semantic Scholar → arXiv (by id / guarded title) → Unpaywall
(by DOI) — and only falls back to the abstract once that chain has failed.

Mode strategy (Thema B):
- FAST   ("abstract") — every paper: abstract only.
- MEDIUM ("hybrid")   — papers are already ranked by relevance; the TOP
                        ``full_text_top_n`` papers get the full-text PDF
                        (coarse, no page metadata), the REST get the abstract
                        only. A paper is NEVER stored as abstract + PDF in
                        parallel. PDF-failure on a top paper → abstract fallback.
- PRO    ("full_pdf") — every paper: full-text PDF, page-aware (each chunk
                        carries its page number). Fallback to abstract.

Per-paper ``ingested_depth`` reflects what actually happened to THAT paper:
    "abstract"  — abstract only (FAST, or the MEDIUM "rest", or a fallback)
    "pdf"       — full-text PDF, coarse  (MEDIUM top papers)
    "full_pdf"  — full-text PDF, paged   (PRO)

Status vocabulary (decision F):
- "success_pdf"        : a PDF was downloaded, parsed, and indexed.
- "success_abstract"   : the abstract was indexed as the *intended* depth
                         (FAST, or a MEDIUM "rest" paper — by policy, not a
                         failure).
- "fallback_abstract"  : a full-text attempt failed, silently fell back.
- "no_content"         : neither PDF nor abstract usable.

Observability (decision F + Thema A):
- pdf_reason : why the PDF attempt ended ("ok" / "paywall" / "no_url" / …).
- pdf_source : which source produced the PDF ("arxiv_id" / "unpaywall" / …)
               or None when the abstract was used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Tuple

from src.core.vector_store import VectorEngine
from src.tools.pdf_tool import REASON_OK
from src.tools.pdf_resolver import fetch_pdf_with_fallback

# Marker for papers whose depth never triggers a PDF attempt
# (FAST, or a MEDIUM paper below the full-text cutoff).
_REASON_NOT_ATTEMPTED = "not_attempted"


@dataclass
class _DepthResult:
    """Outcome of ingesting one paper."""
    chunk_ids: list[str]
    status: str
    depth: str
    pdf_reason: str = _REASON_NOT_ATTEMPTED
    pdf_source: str | None = None


class IngestorAgent:
    """Ingests paper content into ChromaDB at a configurable depth."""

    def __init__(self, collection_name: str = "scholar_papers"):
        # The Ingestor owns its VectorEngine (the "shelf").
        self.vector_engine = VectorEngine(collection_name=collection_name)

    # ------------------------------------------------------------------ #
    #  Entry point (called by the orchestrator's ingestor_node)
    # ------------------------------------------------------------------ #
    def ingest_knowledge(
            self,
            papers: list[dict],
            config: dict,
            status_callback: Callable[[str], None] | None = None,
    ) -> Tuple[list[dict], VectorEngine]:
        """
        Ingest all papers into a fresh ChromaDB collection.

        ``papers`` is assumed to be ordered by relevance (the Researcher ranks
        them and the order is preserved through the UI). In MEDIUM mode the
        first ``full_text_top_n`` papers get full text; the rest get abstract.
        """
        read_depth = config["read_depth"]
        chunk_size = config["chunk_size"]
        quota = self._full_text_quota(read_depth, config, len(papers))

        def _log(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        # The knowledge base is cleared by the UI (session start / "Start
        # Over"); here we only re-bind to the current collection — cheap, no
        # model reload — and ingest into it.
        _log(
            f"[Ingestor] Ingesting — "
            f"full text for top {quota} of {len(papers)} papers"
        )
        self.vector_engine.rebind()

        for idx, paper in enumerate(papers, start=1):
            citation_id = paper.get("citation_id") or f"[{idx}]"
            paper["citation_id"] = citation_id

            title = paper.get("title") or "Untitled"
            _log(f"[Ingestor] Ingesting {citation_id} — {title[:60]}")

            wants_full_text = idx <= quota

            if wants_full_text and read_depth == "full_pdf":
                res = self._ingest_full_text_paged(paper, citation_id, chunk_size, _log)
            elif wants_full_text:
                res = self._ingest_full_text_coarse(paper, citation_id, chunk_size, _log)
            else:
                # Abstract by intent: FAST, or a MEDIUM paper below the cutoff.
                if read_depth != "abstract":
                    _log(
                        f"  ↳ {citation_id} rank {idx} > top-{quota} "
                        f"→ abstract only (policy)"
                    )
                res = self._ingest_abstract(paper, citation_id, chunk_size, intended=True)

            # Attach the lean reference — NOT the text itself.
            paper["chunk_ids"] = res.chunk_ids
            paper["ingestion_status"] = res.status
            paper["ingested_depth"] = res.depth
            paper["pdf_reason"] = res.pdf_reason
            paper["pdf_source"] = res.pdf_source

            _log(
                f"   ↳ {citation_id}: {res.status} "
                f"({len(res.chunk_ids)} chunks, depth={res.depth})"
            )

        # Decision F + Thema A — distributions for this run (baseline metric).
        reason_dist = Counter(p.get("pdf_reason", _REASON_NOT_ATTEMPTED) for p in papers)
        source_dist = Counter(p.get("pdf_source") for p in papers if p.get("pdf_source"))

        _log(f"[Ingestor] PDF-Reasons: {self._fmt_counter(reason_dist)}")
        if source_dist:
            _log(f"[Ingestor] PDF-Sources: {self._fmt_counter(source_dist)}")

        _log(f"✓ [Ingestor] Done — {len(papers)} papers ingested")
        return papers, self.vector_engine

    # ------------------------------------------------------------------ #
    #  Full-text quota per mode (Thema B)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _full_text_quota(read_depth: str, config: dict, n_papers: int) -> int:
        """
        How many of the (relevance-ranked) papers should attempt full text.

        FAST → 0 (abstract only). PRO → all papers. MEDIUM → ``full_text_top_n``
        from the config (capped at the number of papers). If the config key is
        missing, defaults to "all" so we still prefer full text rather than
        silently degrading — but the real profiles SHOULD set it.
        """
        if read_depth == "abstract":
            return 0
        if read_depth == "full_pdf":
            return n_papers
        return min(config.get("full_text_top_n", n_papers), n_papers)

    # ------------------------------------------------------------------ #
    #  Depth: abstract-only
    # ------------------------------------------------------------------ #
    def _ingest_abstract(
            self,
            paper: dict,
            citation_id: str,
            chunk_size: int,
            intended: bool,
    ) -> _DepthResult:
        """
        Index just the abstract.

        ``intended`` distinguishes:
        - True  → abstract is the chosen depth (FAST, or MEDIUM "rest")
                  → "success_abstract"
        - False → a full-text attempt failed, this is the fallback
                  → "fallback_abstract" (caller overwrites pdf_reason).
        """
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            return _DepthResult([], "no_content", "abstract")

        chunk_ids = self.vector_engine.index_paper(
            text=abstract,
            citation_id=citation_id,
            chunk_size=chunk_size,
            metadata=self._meta(paper),
        )
        if not chunk_ids:
            return _DepthResult([], "no_content", "abstract")

        status = "success_abstract" if intended else "fallback_abstract"
        return _DepthResult(chunk_ids, status, "abstract")

    # ------------------------------------------------------------------ #
    #  Depth: full-text PDF, coarse (MEDIUM top papers) — PDF body ONLY
    # ------------------------------------------------------------------ #
    def _ingest_full_text_coarse(
            self,
            paper: dict,
            citation_id: str,
            chunk_size: int,
            log: Callable[[str], None],
    ) -> _DepthResult:
        """
        MEDIUM full-text path: acquire the PDF via the resolver and index the
        PDF body ONLY (Thema B — no abstract concatenated). Coarse chunks, no
        page metadata. Fallback to abstract-only if no usable PDF.
        """
        paper_id = paper.get("paperId") or citation_id
        log(f"  ↳ {citation_id} Downloading full-text PDF...")

        outcome = fetch_pdf_with_fallback(paper, paper_id, log=log)

        if outcome.has_content:
            pdf_text = outcome.result.full_text()  # PDF body only — no abstract
            chunk_ids = self.vector_engine.index_paper(
                text=pdf_text,
                citation_id=citation_id,
                chunk_size=chunk_size,
                metadata=self._meta(paper),
            )
            if chunk_ids:
                log(f"  ↳ {citation_id} PDF acquired via {outcome.source}")
                return _DepthResult(
                    chunk_ids, "success_pdf", "pdf", REASON_OK, outcome.source
                )

        log(
            f"  ↳ ⚠ {citation_id} No PDF ({outcome.result.reason}) "
            f"→ abstract fallback"
        )
        res = self._ingest_abstract(paper, citation_id, chunk_size, intended=False)
        res.pdf_reason = outcome.result.reason
        return res

    # ------------------------------------------------------------------ #
    #  Depth: full-text PDF, page-aware (PRO) — PDF pages ONLY
    # ------------------------------------------------------------------ #
    def _ingest_full_text_paged(
            self,
            paper: dict,
            citation_id: str,
            chunk_size: int,
            log: Callable[[str], None],
    ) -> _DepthResult:
        """
        PRO full-text path: acquire the PDF and index it page-by-page, so every
        chunk carries its 1-based page number (fine-grained reference for the
        Critic). PDF only — no abstract. Fallback to abstract-only on failure.
        """
        paper_id = paper.get("paperId") or citation_id
        log(f"  ↳ {citation_id} Downloading full-text PDF (page-aware)...")

        outcome = fetch_pdf_with_fallback(paper, paper_id, log=log)

        if outcome.result.pages:
            chunk_ids = self.vector_engine.index_paper_by_pages(
                pages=outcome.result.pages,
                citation_id=citation_id,
                chunk_size=chunk_size,
                metadata=self._meta(paper),
            )
            if chunk_ids:
                log(f"  ↳ {citation_id} PDF acquired via {outcome.source}")
                return _DepthResult(
                    chunk_ids, "success_pdf", "full_pdf", REASON_OK, outcome.source
                )

        log(
            f"  ↳ ⚠ {citation_id} No PDF ({outcome.result.reason}) "
            f"→ abstract fallback"
        )
        res = self._ingest_abstract(paper, citation_id, chunk_size, intended=False)
        res.pdf_reason = outcome.result.reason
        return res

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fmt_counter(counter: Counter) -> str:
        """Render a Counter as 'key=n, key=n' ordered by frequency."""
        return ", ".join(f"{key}={n}" for key, n in counter.most_common())

    @staticmethod
    def _meta(paper: dict) -> dict:
        """Flat metadata for ChromaDB (simple types only)."""
        return {
            "title": str(paper.get("title") or "Untitled"),
            "year": str(paper.get("year") or "n.d."),
        }
