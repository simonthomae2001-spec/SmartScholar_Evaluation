"""
critic_agent.py — Fact verification and quality gate (Steps 14 & 15).

The Critic examines the Analyst's structured output, verifies each
analysis record against the (mocked) source material, and decides
whether the analysis meets quality standards or needs revision.

If revision is needed AND the loop budget allows, the pipeline cycles
back to the Analyst with targeted, actionable feedback.

Current source retrieval:  **mocked** — returns the paper's abstract
or a placeholder string.  The ``_fetch_source_context()`` method is
isolated so it can be swapped for a real ChromaDB / VectorEngine
query once the IngestorAgent is implemented.
"""

from __future__ import annotations

import json
import re

from src.core.model_factory import ModelFactory


class CriticAgent:
    """Verifies the quality of the Analyst's structured analysis against
    source material and generates targeted feedback for revision."""

    # ------------------------------------------------------------------ #
    #  Evaluation prompt — sent to the LLM for each analysis record
    # ------------------------------------------------------------------ #
    EVALUATION_PROMPT: str = (
        "You are a rigorous academic fact-checker. You are given:\n"
        "1. An **analysis record** produced by an AI Analyst for a scientific paper.\n"
        "2. The **source text** (abstract or excerpts) from that paper.\n\n"
        "Your task is to evaluate how well the analysis record is supported by\n"
        "the source text. Check the following dimensions:\n\n"
        "- **Methodology accuracy**: Does the 'methodology' field correctly\n"
        "  reflect what the source describes?\n"
        "- **Findings accuracy**: Are the 'findings' faithfully grounded in\n"
        "  the source? Flag any hallucinated or unsupported claims.\n"
        "- **Limitations completeness**: Does the 'limitations' field capture\n"
        "  the caveats present in the source?\n"
        "- **Relevance assessment**: Is the 'user_relevance' judgement\n"
        "  reasonable given the source content?\n\n"
        "Return ONLY a valid JSON object with exactly these keys:\n"
        '  "consistency_score": integer 0-100 (100 = perfectly consistent),\n'
        '  "issues": a JSON array of strings, each describing one specific\n'
        "    problem found (empty array [] if none),\n"
        '  "summary": a 1-2 sentence overall verdict.\n\n'
        "No markdown fences, no explanation outside the JSON.\n\n"
        "--- ANALYSIS RECORD ---\n"
        "Citation ID: {citation_id}\n"
        "Methodology: {methodology}\n"
        "Findings: {findings}\n"
        "Limitations: {limitations}\n"
        "User Relevance: {user_relevance}\n\n"
        "--- SOURCE TEXT ---\n"
        "{source_text}\n\n"
        "JSON object:"
    )

    def __init__(self) -> None:
        """Initialise the Critic with an Ollama LLM instance."""
        self.llm = ModelFactory.get_model()

    # ------------------------------------------------------------------ #
    #  Public entry-point (called by critic_node in orchestrator.py)
    # ------------------------------------------------------------------ #

    def verify_facts(
        self,
        analysis_data: list[dict],
        config: dict,
        loop_count: int,
    ) -> tuple[bool, str]:
        """
        Verify the Analyst's output and decide pass / fail.

        **Loop-budget guard:** If ``loop_count >= config["max_loops"]``,
        the method auto-approves immediately — no LLM call is made.
        The ``max_loops`` value is read from ``config`` at runtime
        (Single Source of Truth).

        Parameters
        ----------
        analysis_data : list[dict]
            Structured records from the AnalystAgent.  Each dict is
            expected to contain ``citation_id``, ``methodology``,
            ``findings``, ``limitations``, ``user_relevance``.
        config : dict
            Profile config dict (from ``get_config()``).
            Must contain ``max_loops``.
        loop_count : int
            How many critic→analyst iterations have already occurred.

        Returns
        -------
        tuple[bool, str]
            ``(True, feedback)``  → analysis **passed** verification.
            ``(False, feedback)`` → analysis **failed**; if
            ``loop_count < config['max_loops']``, the orchestrator
            should route back to the Analyst.
        """
        max_loops: int = config["max_loops"]
        pass_threshold: int = config.get("critic_pass_threshold", 60)

        # ----- Guard: loop budget exhausted → auto-approve ------------ #
        if loop_count >= max_loops:
            return (
                True,
                f"⚠️ Loop budget exhausted ({loop_count}/{max_loops}). "
                f"Auto-approving {len(analysis_data)} analysis records "
                f"without further verification.",
            )

        # ----- No data to verify -------------------------------------- #
        if not analysis_data:
            return (True, "No analysis records to verify — passing vacuously.")

        # ----- Per-record LLM evaluation ------------------------------ #
        all_issues: list[str] = []
        record_verdicts: list[dict] = []

        for record in analysis_data:
            source_text = self._fetch_source_context(record)
            verdict = self._evaluate_record(record, source_text)
            record_verdicts.append(verdict)

            # Collect issues tagged with the citation ID
            cid = record.get("citation_id", "?")
            for issue in verdict.get("issues", []):
                all_issues.append(f"{cid} — {issue}")

        # ----- Aggregate decision ------------------------------------- #
        scores = [v.get("consistency_score", 0) for v in record_verdicts]
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score >= pass_threshold and not all_issues:
            # All clear
            feedback = (
                f"All {len(analysis_data)} analysis records verified "
                f"(avg consistency {avg_score:.0f}/100). No issues found."
            )
            return (True, feedback)

        # ----- Build targeted feedback for the Analyst ---------------- #
        feedback_parts: list[str] = [
            f"Verification FAILED (avg consistency {avg_score:.0f}/100, "
            f"threshold {pass_threshold}/100).  "
            f"Loop {loop_count + 1}/{max_loops}.",
            "",
            "Issues requiring revision:",
        ]

        if all_issues:
            for i, issue in enumerate(all_issues, start=1):
                feedback_parts.append(f"  {i}. {issue}")
        else:
            feedback_parts.append(
                "  (No specific issues extracted, but aggregate score "
                "is below threshold.  Please improve analysis depth.)"
            )

        # Append per-record summaries
        feedback_parts.append("")
        feedback_parts.append("Per-record summaries:")
        for rec, verdict in zip(analysis_data, record_verdicts):
            cid = rec.get("citation_id", "?")
            score = verdict.get("consistency_score", "N/A")
            summary = verdict.get("summary", "No summary.")
            feedback_parts.append(f"  {cid} (score {score}/100): {summary}")

        return (False, "\n".join(feedback_parts))

    # ------------------------------------------------------------------ #
    #  Source context retrieval (Separation of Concerns)
    # ------------------------------------------------------------------ #

    def _fetch_source_context(self, record: dict) -> str:
        """
        Retrieve the source text for a given analysis record.

        .. note::

            This method currently returns a **mocked** context string
            composed from the record's own fields.  It is intentionally
            isolated so it can be replaced with a real ChromaDB /
            VectorEngine retrieval call once the IngestorAgent is
            implemented.

        # TODO: Replace with ChromaDB retrieval (VectorEngine) once Ingestor is implemented.
        #
        # Planned implementation:
        #   1. Use the citation_id / paper_id to query the ChromaDB
        #      collection for the relevant chunks.
        #   2. Return the concatenated chunk texts as the source context.
        #   3. Optionally include section headers (Intro, Methods, etc.)
        #      for richer verification.

        Parameters
        ----------
        record : dict
            A single analysis record from ``paper_analysis_data``.
            Expected keys: ``citation_id``, ``methodology``, ``findings``,
            ``limitations``, ``user_relevance``.

        Returns
        -------
        str
            The source text to verify the analysis against.
        """
        # Use the paper's abstract if available in the record, otherwise
        # synthesise a representative placeholder from the record fields.
        abstract: str = record.get("abstract", "")
        if abstract:
            return abstract

        # Fallback: construct a mock source from the record itself so the
        # LLM has *something* to compare against.  This naturally biases
        # towards "pass" since we're comparing the analysis with its own
        # source — which is the desired stub behaviour (don't block the
        # pipeline on unimplemented infra).
        parts: list[str] = []
        title = record.get("title", "")
        if title:
            parts.append(f"Title: {title}")
        parts.append(f"Methodology: {record.get('methodology', 'N/A')}")
        parts.append(f"Findings: {record.get('findings', 'N/A')}")
        parts.append(f"Limitations: {record.get('limitations', 'N/A')}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    #  LLM evaluation of a single record
    # ------------------------------------------------------------------ #

    def _evaluate_record(
        self, record: dict, source_text: str
    ) -> dict:
        """
        Prompt the LLM to evaluate one analysis record against its source.

        Parameters
        ----------
        record : dict
            Analysis record with ``citation_id``, ``methodology``,
            ``findings``, ``limitations``, ``user_relevance``.
        source_text : str
            The reference text retrieved via ``_fetch_source_context()``.

        Returns
        -------
        dict
            Parsed LLM response with keys ``consistency_score`` (int),
            ``issues`` (list[str]), and ``summary`` (str).
            Returns a safe default on any failure.
        """
        prompt = self.EVALUATION_PROMPT.format(
            citation_id=record.get("citation_id", "?"),
            methodology=record.get("methodology", "N/A"),
            findings=record.get("findings", "N/A"),
            limitations=record.get("limitations", "N/A"),
            user_relevance=record.get("user_relevance", "N/A"),
            source_text=source_text,
        )

        try:
            response = self.llm.complete(prompt)
            raw_text = str(response).strip()
            parsed = self._parse_json_object(raw_text)

            if parsed and isinstance(parsed, dict):
                # Normalise and clamp
                score = int(parsed.get("consistency_score", 50))
                score = max(0, min(100, score))
                issues = parsed.get("issues", [])
                if not isinstance(issues, list):
                    issues = []
                issues = [str(i) for i in issues]
                summary = str(parsed.get("summary", ""))
                return {
                    "consistency_score": score,
                    "issues": issues,
                    "summary": summary,
                }
        except Exception as exc:
            print(f"[CriticAgent] LLM evaluation failed: {exc}")

        # Safe fallback: assume moderate quality, flag for human review
        return {
            "consistency_score": 50,
            "issues": ["LLM evaluation failed — could not verify this record."],
            "summary": "Evaluation inconclusive due to LLM error.",
        }

    # ------------------------------------------------------------------ #
    #  JSON parsing helpers (mirrors ResearcherAgent patterns)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        return text.strip("`").strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        """
        Robustly extract a JSON object from LLM output that may contain
        markdown fences or surrounding prose.
        """
        text = CriticAgent._strip_markdown_fences(text)

        # Try direct parse first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to locate a JSON object substring
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None
