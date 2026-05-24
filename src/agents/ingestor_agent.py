"""
ingestor_agent.py — Knowledge ingestion (Steps 10A, 10B, 11B).

The Ingestor processes accepted papers at the depth defined by the
active research profile's ``read_depth``:

- ``"abstract"``  → index only the abstract text.
- ``"hybrid"``    → abstract + first N chunks of the full PDF.
- ``"full_pdf"``  → download and chunk the complete PDF.

Current implementation: **stub** that simulates ingestion by echoing
the depth without performing real PDF processing.
"""

from __future__ import annotations


class IngestorAgent:
    """Ingests paper content at a configurable depth."""

    def ingest_knowledge(
        self,
        papers: list[dict],
        config: dict,
    ) -> list[dict]:
        """
        Simulate (or perform) knowledge ingestion for a set of papers.

        Parameters
        ----------
        papers : list[dict]
            Paper dicts from the active_papers list.
        config : dict
            The profile config dict (from ``get_config()``).
            Must contain ``read_depth`` and ``chunk_size``.

        Returns
        -------
        list[dict]
            The same paper dicts, each augmented with an
            ``ingested_content`` key summarising what was processed.
        """
        read_depth = config["read_depth"]
        chunk_size = config["chunk_size"]

        for paper in papers:
            if read_depth == "abstract":
                paper["ingested_content"] = (
                    f"[abstract-only] {(paper.get('abstract') or 'N/A')[:chunk_size]}"
                )
            elif read_depth == "hybrid":
                abstract_text = (paper.get("abstract") or "N/A")[:chunk_size]
                paper["ingested_content"] = (
                    f"[hybrid] Abstract: {abstract_text} "
                    f"| PDF first {chunk_size} chars: <stub>"
                )
            elif read_depth == "full_pdf":
                paper["ingested_content"] = (
                    f"[full_pdf] Full PDF processed in "
                    f"{chunk_size}-char chunks: <stub>"
                )
            else:
                paper["ingested_content"] = (
                    f"[unknown depth '{read_depth}'] Fallback to abstract."
                )

        return papers
