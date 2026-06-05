"""
ingestor_agent.py — Knowledge ingestion.

The Ingestor is the "librarian": for each finalised paper it ingests text
into ChromaDB at the profile's read_depth, and attaches a LEAN reference
(chunk_ids + ingestion_status + ingested_depth) to the paper dict. Large
content (chunks + embeddings) lives ONLY in ChromaDB, never in the graph
state.

Phase 1 status: read_depth == "abstract" (FAST) is fully implemented.
hybrid / full_pdf fall back to abstract until the PDF logic is added in
Phase 3/4 (the pdf_tool already exists and is tested separately).
"""
from __future__ import annotations

from typing import Callable

from src.core.vector_store import VectorEngine


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

        Parameters
        ----------
        papers : list[dict]
            The finalised active_papers (from the paper-review step).
        config : dict
            Profile config; must contain ``read_depth`` and ``chunk_size``.
        status_callback : callable | None
            Optional progress sink (wired to ``_ui_log`` by the orchestrator)
            so the Streamlit UI doesn't freeze during long ingestion.

        Returns
        -------
        list[dict]
            The same papers, each augmented with ``citation_id``,
            ``chunk_ids``, ``ingestion_status``, and ``ingested_depth``.
            Large content lives only in ChromaDB.
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
            # Guarantee a citation_id — it's the chunk-id prefix AND the
            # backward reference the Critic relies on.
            citation_id = paper.get("citation_id") or f"[{idx}]"
            paper["citation_id"] = citation_id

            title = paper.get("title") or "Untitled"
            _log(f"📄 [Ingestor] Ingesting {citation_id} — {title[:60]}…")

            if read_depth == "abstract":
                chunk_ids, status = self._ingest_abstract(
                    paper, citation_id, chunk_size
                )
                depth_done = "abstract"
            else:
                # hybrid / full_pdf: PDF logic arrives in Phase 3/4.
                # Until then, fall back to abstract so the pipeline runs.
                chunk_ids, status = self._ingest_abstract(
                    paper, citation_id, chunk_size
                )
                depth_done = "abstract"

            # Attach the lean reference — NOT the text itself.
            paper["chunk_ids"] = chunk_ids
            paper["ingestion_status"] = status
            paper["ingested_depth"] = depth_done

            _log(
                f"   ↳ {citation_id}: {status} "
                f"({len(chunk_ids)} chunks, depth={depth_done})"
            )

        _log(f"✅ [Ingestor] Done — {len(papers)} papers ingested.")
        return papers

    # ------------------------------------------------------------------ #
    #  Depth: abstract-only (FAST, and current fallback for all profiles)
    # ------------------------------------------------------------------ #
    def _ingest_abstract(
        self, paper: dict, citation_id: str, chunk_size: int
    ) -> tuple[list[str], str]:
        """
        Index just the abstract.

        Returns
        -------
        tuple[list[str], str]
            ``(chunk_ids, ingestion_status)``. Status is ``"success"`` when
            chunks were written, or ``"no_content"`` when the paper has no
            usable abstract (a graceful, non-crashing edge case).
        """
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            return ([], "no_content")

        metadata = {
            "title": str(paper.get("title") or "Untitled"),
            "year": str(paper.get("year") or "n.d."),
        }
        chunk_ids = self.vector_engine.index_paper(
            text=abstract,
            citation_id=citation_id,
            chunk_size=chunk_size,
            metadata=metadata,
        )
        status = "success" if chunk_ids else "no_content"
        return (chunk_ids, status)