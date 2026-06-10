"""
ingestor_agent.py — Knowledge ingestion.

The Ingestor is the "librarian": for each finalised paper it ingests text
into ChromaDB at the profile's read_depth, and attaches a LEAN reference
(chunk_ids + ingestion_status + ingested_depth) to the paper dict. Large
content lives ONLY in ChromaDB, never in the graph state.

Implemented depths:
- "abstract" (FAST)   — index the abstract only.
- "hybrid"   (MEDIUM) — try the open-access PDF; index abstract + PDF body
                        together (coarse, no page metadata). Silent per-paper
                        fallback to abstract-only if no usable PDF.
- "full_pdf" (PRO)    — Phase 4. Until then routed through the hybrid path
                        (uses the pro chunk_size, but no page metadata /
                        cleanup yet).

Status vocabulary (decision F — feeds observability + Termin-5 metrics):
- "success_pdf"        : a PDF was downloaded, parsed, and indexed.
- "success_abstract"   : the abstract was indexed as the chosen depth (FAST).
- "fallback_abstract"  : no usable PDF, silently fell back to the abstract.
- "no_content"         : neither PDF nor abstract usable.
"""
from __future__ import annotations

from typing import Callable

from src.core.vector_store import VectorEngine
from src.tools.pdf_tool import fetch_pdf_text


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
    ) -> list[dict]:
        """
        Ingest all papers into a fresh ChromaDB collection.

        Returns the same papers, each augmented with ``citation_id``,
        ``chunk_ids``, ``ingestion_status``, and ``ingested_depth``.
        """
        read_depth = config["read_depth"]
        chunk_size = config["chunk_size"]

        def _log(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        # Fresh collection for exactly this run (decision 0.C).
        _log("🗄️ [Ingestor] Resetting knowledge base for this run…")
        self.vector_engine.reset_collection()

        for idx, paper in enumerate(papers, start=1):
            # Guarantee a citation_id — the chunk-id prefix AND the backward
            # reference the Critic relies on.
            citation_id = paper.get("citation_id") or f"[{idx}]"
            paper["citation_id"] = citation_id

            title = paper.get("title") or "Untitled"
            _log(f"📄 [Ingestor] Ingesting {citation_id} — {title[:60]}…")

            if read_depth == "abstract":
                chunk_ids, status, depth = self._ingest_abstract(
                    paper, citation_id, chunk_size, intended=True
                )
            elif read_depth == "hybrid":
                chunk_ids, status, depth = self._ingest_hybrid(
                    paper, citation_id, chunk_size, _log
                )
            elif read_depth == "full_pdf":
                chunk_ids, status, depth = self._ingest_full_pdf(
                    paper, citation_id, chunk_size, _log
                )
            else:
                # Unknown depth → safe default: abstract.
                chunk_ids, status, depth = self._ingest_abstract(
                    paper, citation_id, chunk_size, intended=True
                )

            # Attach the lean reference — NOT the text itself.
            paper["chunk_ids"] = chunk_ids
            paper["ingestion_status"] = status
            paper["ingested_depth"] = depth

            _log(
                f"   ↳ {citation_id}: {status} "
                f"({len(chunk_ids)} chunks, depth={depth})"
            )

        _log(f"✅ [Ingestor] Done — {len(papers)} papers ingested.")
        return papers

    # ------------------------------------------------------------------ #
    #  Depth: abstract-only (FAST + the silent fallback for MEDIUM/PRO)
    # ------------------------------------------------------------------ #
    def _ingest_abstract(
            self,
            paper: dict,
            citation_id: str,
            chunk_size: int,
            intended: bool,
    ) -> tuple[list[str], str, str]:
        """
        Index just the abstract.

        ``intended`` distinguishes the two reasons we end up here:
        - True  → abstract is the chosen depth (FAST)        → "success_abstract"
        - False → PDF was unavailable, this is the fallback  → "fallback_abstract"

        Returns ``(chunk_ids, ingestion_status, depth)``.
        """
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            return ([], "no_content", "abstract")

        chunk_ids = self.vector_engine.index_paper(
            text=abstract,
            citation_id=citation_id,
            chunk_size=chunk_size,
            metadata=self._meta(paper),
        )
        if not chunk_ids:
            return ([], "no_content", "abstract")

        status = "success_abstract" if intended else "fallback_abstract"
        return (chunk_ids, status, "abstract")

    # ------------------------------------------------------------------ #
    #  Depth: hybrid (MEDIUM) — abstract + PDF body, coarse, with fallback
    # ------------------------------------------------------------------ #
    def _ingest_hybrid(
            self,
            paper: dict,
            citation_id: str,
            chunk_size: int,
            log: Callable[[str], None],
    ) -> tuple[list[str], str, str]:
        """
        Try the open-access PDF. On success, index abstract + PDF body text
        together (coarse, no page metadata). On any failure (no link,
        paywall, parse error), silently fall back to abstract-only.

        Returns ``(chunk_ids, ingestion_status, depth)``.
        """
        url = paper.get("openAccessPdf")
        paper_id = paper.get("paperId") or citation_id

        if url:
            log(f"   ⬇️ {citation_id} Downloading & parsing PDF…")

        pdf_result = fetch_pdf_text(url, paper_id)

        if pdf_result.has_content:
            abstract = (paper.get("abstract") or "").strip()
            pdf_text = pdf_result.full_text()
            # "hybrid": the reliable abstract + the full PDF body.
            combined = f"{abstract}\n\n{pdf_text}" if abstract else pdf_text

            chunk_ids = self.vector_engine.index_paper(
                text=combined,
                citation_id=citation_id,
                chunk_size=chunk_size,
                metadata=self._meta(paper),
            )
            if chunk_ids:
                return (chunk_ids, "success_pdf", "hybrid")

        # Silent per-paper fallback to abstract-only.
        return self._ingest_abstract(
            paper, citation_id, chunk_size, intended=False
        )

    # ------------------------------------------------------------------ #
    #  Depth: full_pdf (PRO) — page-aware, fine chunks, page metadata
    # ------------------------------------------------------------------ #
    def _ingest_full_pdf(
            self,
            paper: dict,
            citation_id: str,
            chunk_size: int,
            log: Callable[[str], None],
    ) -> tuple[list[str], str, str]:
        """
        PRO: download the full PDF and index it page-by-page, so every chunk
        carries its page number (fine-grained reference for the Critic).
        Silent per-paper fallback to abstract-only if no usable PDF.

        Returns ``(chunk_ids, ingestion_status, depth)``.
        """
        url = paper.get("openAccessPdf")
        paper_id = paper.get("paperId") or citation_id

        if url:
            log(f"   ⬇️ {citation_id} Downloading & parsing full PDF…")

        pdf_result = fetch_pdf_text(url, paper_id)

        if pdf_result.pages:
            chunk_ids = self.vector_engine.index_paper_by_pages(
                pages=pdf_result.pages,
                citation_id=citation_id,
                chunk_size=chunk_size,
                metadata=self._meta(paper),
            )
            if chunk_ids:
                return (chunk_ids, "success_pdf", "full_pdf")

        # Silent per-paper fallback to abstract-only.
        return self._ingest_abstract(
            paper, citation_id, chunk_size, intended=False
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _meta(paper: dict) -> dict:
        """Flat metadata for ChromaDB (simple types only)."""
        return {
            "title": str(paper.get("title") or "Untitled"),
            "year": str(paper.get("year") or "n.d."),
        }
