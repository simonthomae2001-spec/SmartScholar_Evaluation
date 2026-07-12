"""
critic_agent.py — Fact verification and quality gate (Steps 14 & 15).

The Critic examines the Analyst's structured output, verifies each
analysis record against the ingested source material stored in ChromaDB,
and decides whether the analysis meets quality standards or needs revision.

If revision is needed AND the loop budget allows, the pipeline cycles
back to the Analyst with targeted, actionable feedback.

Source retrieval:  **RAG-based** — uses the VectorEngine to perform
similarity searches on ChromaDB with strict ``citation_id`` metadata
filtering, ensuring that each claim is verified exclusively against
its own paper's chunks (no cross-contamination between papers).
"""

from __future__ import annotations

import json
from typing import List, Dict
import re
from typing import Callable

from src.core.model_factory import ModelFactory
from src.core.vector_store import VectorEngine

class CriticAgent:
    """Verifies the quality of the Analyst's structured analysis against
    source material retrieved from ChromaDB via filtered similarity search."""

    # ------------------------------------------------------------------ #
    #  Evaluation prompt — sent to the LLM for each analysis record
    # ------------------------------------------------------------------ #
    EVALUATION_PROMPT: str = (
        "You are a rigorous academic fact-checker. You are given:\n"
        "1. An **analysis record** produced by an AI Analyst for a scientific paper.\n"
        "2. The **source text** (retrieved excerpts) from that paper.\n\n"
        "THINK FIRST: Before scoring, explain your step-by-step reasoning in the\n"
        "\"rationale\" field. Compare each claim against the source text and note\n"
        "matches and contradictions. Then assign a score and list specific issues.\n\n"
        "Evaluation dimensions:\n"
        "- **Methodology accuracy**: Does the 'methodology' field correctly\n"
        "  reflect what the source describes?\n"
        "- **Findings accuracy**: Are the 'findings' faithfully grounded in\n"
        "  the source? Flag any hallucinated or unsupported claims.\n"
        "- **Limitations completeness**: Does the 'limitations' field capture\n"
        "  the caveats present in the source?\n"
        "- **Relevance assessment**: Is the 'user_relevance' judgement\n"
        "  reasonable given the source content?\n\n"
        "--- ANALYSIS RECORD ---\n"
        "Citation ID: {citation_id}\n"
        "Methodology: {methodology}\n"
        "Findings: {findings}\n"
        "Limitations: {limitations}\n"
        "User Relevance: {user_relevance}\n\n"
        "--- SOURCE TEXT ---\n"
        "{source_text}\n\n"
        "YOU MUST RESPOND ONLY WITH A VALID JSON OBJECT ENCLOSED IN A MARKDOWN BLOCK (```json ... ```). "
        "DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.\n"
        "EXACT FORMAT REQUIRED (write the rationale FIRST to enforce reasoning):\n"
        "```json\n"
        "{{\n"
        '  "rationale": "Explain why it matches or contradicts. STRICT RULE: KEEP THIS EXTREMELY CONCISE, MAXIMUM 2 SENTENCES.",\n'
        '  "consistency_score": 75,\n'
        '  "issues": ["Short issue 1", "Short issue 2"],\n'
        '  "summary": "1-2 sentence overall verdict."\n'
        "}}\n"
        "```\n"
    )

    def __init__(self) -> None:
        """Initialise the Critic with an Ollama LLM and a VectorEngine.

        The VectorEngine connects to the same persistent ChromaDB instance
        used by the IngestorAgent, giving the Critic read access to all
        ingested paper chunks for RAG-based verification.
        """
        self.llm = ModelFactory.get_model()
        self.vector_engine = VectorEngine()

    # ------------------------------------------------------------------ #
    #  Public entry-point (called by critic_node in orchestrator.py)
    # ------------------------------------------------------------------ #

    def verify_facts(
        self,
        analysis_data: list[dict],
        config: dict,
        loop_count: int,
        status_callback: Callable[[str], None] | None = None,
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
        status_callback : Callable[[str], None] | None
            Optional callback to stream trace-of-thought messages to
            the Streamlit UI.  If ``None``, logging is silently skipped.

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

        def _log(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        # Re-hydrate the vector engine to ensure binding to the active collection after UI resets
        if hasattr(self, 'vector_engine') and self.vector_engine is not None:
            self.vector_engine.rebind()
        else:
            self.vector_engine = VectorEngine()

        _log(
            f"🧠 [Critic] Verification started "
            f"(Round {loop_count + 1}/{max_loops}, "
            f"Threshold: {pass_threshold}/100)"
        )

        # ----- Guard: loop budget exhausted → auto-approve ------------ #
        if loop_count >= max_loops:
            _log(
                f"⚠ [Critic] Loop budget exhausted ({loop_count}/{max_loops}). "
                f"Auto-approving {len(analysis_data)} records"
            )
            return (
                True,
                f"Loop budget exhausted ({loop_count}/{max_loops}). "
                f"Auto-approved {len(analysis_data)} analysis records "
                f"without further verification.",
            )

        # ----- No data to verify -------------------------------------- #
        if not analysis_data:
            _log("[Critic] No analysis records to verify — passing vacuously")
            return (True, "No analysis records to verify — passing vacuously.")

        # ----- Per-record LLM evaluation ------------------------------ #
        all_issues: list[str] = []
        feedback_per_record: Dict[int, dict] = {}
        record_verdicts: list[dict] = []

        top_k: int = config.get("critic_top_k", 3)

        for record in analysis_data:
            cid = record.get("citation_id", "?")
            _log(f"🧠 [Critic] Verifying paper {cid}")

            source_text = self._build_verification_context(record, top_k, _log)
            verdict = self._evaluate_record(record, source_text, _log)
            record_verdicts.append(verdict)
            feedback_per_record[cid] = verdict

            # Log the LLM's verdict for this record
            score = verdict.get("consistency_score", 0)
            issues = verdict.get("issues", [])
            summary = verdict.get("summary", "")

            if issues:
                for issue in issues:
                    _log(f"  ↳ ✕ Issue: {issue}")
                    all_issues.append(f"{cid} — {issue}")
            else:
                _log(f"  ↳ ✓ No issues found (score {score}/100)")

            if summary:
                _log(f"  ↳ Verdict: {summary}")

        # ----- Aggregate decision ------------------------------------- #
        valid_scores = [v.get("consistency_score", 0) for v in record_verdicts if isinstance(v.get("consistency_score"), (int, float))]
        aggregate_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        # A batch passes if the average meets the threshold AND no single paper is completely broken (< 50)
        min_floor_score = 50
        lowest_score = min(valid_scores) if valid_scores else 0

        _critic_passed = (aggregate_score >= pass_threshold) and (lowest_score >= min_floor_score)

        if _critic_passed:
            # All clear (treat remaining issues as minor constructive feedback)
            feedback = (
                f"All {len(analysis_data)} analysis records verified "
                f"(avg consistency {aggregate_score:.0f}/100, lowest {lowest_score:.0f}/100)."
            )
            _log(f"✓ [Critic] Aggregate score: {aggregate_score:.0f}/100 meets threshold ({pass_threshold}/100) — releasing to Synthesizer")
            return (True, feedback)

        current_loop = loop_count + 1
        if current_loop < max_loops:
            _log(f"✕ [Critic] Verification failed (Score: {aggregate_score:.0f}/100, Lowest: {lowest_score}/100 vs Threshold: {pass_threshold}/100) — sending feedback to Analyst")
        else:
            _log(
                f"⚠ [Critic] Loop budget exhausted ({current_loop}/{max_loops}) "
                "— force-releasing unverified records to Synthesizer"
            )
            
        return (False, feedback_per_record)

    # ------------------------------------------------------------------ #
    #  RAG-based source context retrieval
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_id(cid: str | int) -> str:
        s_id = str(cid).strip()
        if not s_id.startswith("["):
            s_id = f"[{s_id}]"
        if not s_id.endswith("]"):
            s_id = f"{s_id}]"
        return s_id

    def _build_verification_context(
        self,
        record: dict,
        top_k: int,
        _log: Callable[[str], None] | None = None,
    ) -> str:
        """
        Build a combined source-text context for a single analysis record
        by retrieving relevant chunks for each verifiable claim.
        """
        if _log is None:
            def _log(msg: str) -> None:
                pass

        raw_id = record.get("citation_id", "?")
        norm_id = self._normalize_id(raw_id)
        methodology = record.get("methodology", "")
        findings = record.get("findings", "")

        context_parts: list[str] = []

        # Retrieve context for methodology claim
        if methodology:
            _log(f"  ↳ Searching ChromaDB for methodology evidence ({norm_id})")
            meth_ctx = self._fetch_source_context(methodology, norm_id, top_k)
            if meth_ctx:
                n_chunks = meth_ctx.count("\n---\n") + 1
                _log(f"  ↳ Found {n_chunks} chunk(s) for methodology")
                context_parts.append(
                    f"[Source context for METHODOLOGY]\n{meth_ctx}"
                )
            else:
                _log(f"  ↳ ⚠ No chunks found for methodology claim")

        # Retrieve context for findings claim
        if findings:
            _log(f"  ↳ Searching ChromaDB for findings evidence ({norm_id})")
            find_ctx = self._fetch_source_context(findings, norm_id, top_k)
            if find_ctx:
                n_chunks = find_ctx.count("\n---\n") + 1
                _log(f"  ↳ Found {n_chunks} chunk(s) for findings")
                context_parts.append(
                    f"[Source context for FINDINGS]\n{find_ctx}"
                )
            else:
                _log(f"  ↳ ⚠ No chunks found for findings claim")

        if not context_parts:
            _log(f"  ↳ ⚠ No source context available for {norm_id}")
            return "No source context found in database for verification."

        return "\n\n".join(context_parts)

    def _fetch_source_context(
        self, claim_text: str, citation_id: str, top_k: int,
    ) -> str:
        """
        Retrieve the most relevant source chunks for a specific claim from
        ChromaDB, filtered to a single paper.

        Uses the VectorEngine's underlying ChromaDB collection to perform
        a similarity search for ``claim_text``, applying a strict metadata
        filter: ``{"citation_id": citation_id}``.  This guarantees that
        only chunks belonging to the specified paper are considered —
        preventing cross-contamination of facts between different papers.

        The number of chunks retrieved is controlled by ``top_k``, which
        is read from ``config["critic_top_k"]`` (Single Source of Truth
        in ``config.yaml``).

        Parameters
        ----------
        claim_text : str
            The specific claim to verify (e.g., the methodology or
            findings text from the Analyst's output).
        citation_id : str
            The paper identifier used during ingestion (e.g., ``"[1]"``).
            Must match the ``citation_id`` stored in chunk metadata
            by the IngestorAgent.
        top_k : int
            Number of most-similar chunks to retrieve.

        Returns
        -------
        str
            Concatenated text of the top-K matching chunks, separated by
            ``---`` dividers. Returns an empty string if no matching
            chunks are found (caller handles the fallback).
        """
        try:
            # Use the embedding model to encode the claim for similarity search
            claim_embedding = self.vector_engine.embed_model.get_query_embedding(
                claim_text
            )

            # Query ChromaDB directly with metadata filter to scope results
            # to ONLY this paper's chunks. The `where` filter enforces
            # citation_id == citation_id, preventing cross-paper leakage.
            results = self.vector_engine.chroma_collection.query(
                query_embeddings=[claim_embedding],
                n_results=top_k,
                where={"citation_id": citation_id},
            )

            # ChromaDB returns {"documents": [[str, ...]], ...}
            documents = results.get("documents", [[]])[0]

            if not documents:
                return ""

            # Concatenate chunks with clear separators
            return "\n---\n".join(documents)

        except Exception as exc:
            print(
                f"[CriticAgent] ChromaDB retrieval failed for "
                f"{citation_id}: {exc}"
            )
            return ""

    # ------------------------------------------------------------------ #
    #  LLM evaluation of a single record
    # ------------------------------------------------------------------ #

    def _evaluate_record(
        self,
        record: dict,
        source_text: str,
        _log: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Prompt the LLM to evaluate one analysis record against its source.

        The parsing logic handles the LLM response with robust fallbacks:

        - **Success:** The LLM returns a valid JSON object.  The
          ``consistency_score``, ``issues``, and ``summary`` keys are
          extracted and normalised.  The optional ``rationale`` key
          (from the CoT prompt) is folded into the summary for
          traceability.
        - **Parse failure:** If JSON extraction fails entirely, the
          record is scored at **0** with a ``Critical System Error``
          issue so the Critic loop correctly flags it for revision
          instead of silently passing.

        Parameters
        ----------
        record : dict
            Analysis record with ``citation_id``, ``methodology``,
            ``findings``, ``limitations``, ``user_relevance``.
        source_text : str
            The reference text retrieved via ``_fetch_source_context()``.
        _log : Callable[[str], None] | None
            Optional logging callback for trace-of-thought messages.

        Returns
        -------
        dict
            Parsed LLM response with keys ``consistency_score`` (int),
            ``issues`` (list[str]), and ``summary`` (str).
            Returns a score-0 fallback on total parsing failure.
        """
        if _log is None:
            def _log(msg: str) -> None:
                pass

        cid = record.get("citation_id", "?")
        _log(f"  ↳ Sending {cid} to LLM for fact-check evaluation")

        prompt = self.EVALUATION_PROMPT.format(
            citation_id=cid,
            methodology=record.get("methodology", "N/A"),
            findings=record.get("findings", "N/A"),
            limitations=record.get("limitations", "N/A"),
            user_relevance=record.get("user_relevance", "N/A"),
            source_text=source_text,
        )

        try:
            response = self.llm.complete(
                prompt,
                format="json",
                temperature=0.1,
                num_predict=1500
            )
            raw_text = str(response).strip()
            parsed = self._extract_and_parse_json(raw_text)

            if parsed and isinstance(parsed, dict):
                # Normalise and clamp consistency_score
                score = int(parsed.get("consistency_score", 50))
                score = max(0, min(100, score))

                issues = parsed.get("issues", [])
                if not isinstance(issues, list):
                    issues = []
                issues = [str(i) for i in issues]

                # Build summary — include CoT rationale if present
                rationale = str(parsed.get("rationale", "")).strip()
                summary = str(parsed.get("summary", "")).strip()
                if rationale and not summary:
                    summary = rationale
                elif rationale and summary:
                    summary = f"{summary} (Rationale: {rationale})"

                _log(f"  ↳ LLM score for {cid}: {score}/100")
                return {
                    "consistency_score": score,
                    "issues": issues,
                    "summary": summary,
                }

            # Parsing returned None — LLM produced unparseable output
            _log(f"  ↳ ⚠ JSON parsing failed for {cid}")

        except Exception as exc:
            print(f"[CriticAgent] LLM evaluation failed: {exc}")
            _log(f"  ↳ ⚠ [Critic] JSON parsing failed for {cid}: {exc}")

        # Hard fallback: score 0 so the loop correctly rejects this record.
        return {
            "consistency_score": 0,
            "issues": [
                "System: LLM output was severely truncated or invalid and could not be auto-repaired."
            ],
            "summary": "Evaluation could not be completed — LLM output was not valid JSON.",
        }

    # ------------------------------------------------------------------ #
    #  JSON parsing helpers (mirrors ResearcherAgent patterns)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        return text.strip("`").strip()

    def _extract_and_parse_json(self, text: str) -> dict:
        """
        Robustly extract and auto-repair a JSON object from LLM output.
        """
        if not text:
            raise ValueError("Empty response from LLM")

        # 1. Strip markdown code blocks if present
        cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)

        # 2. Locate the outermost curly braces
        match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # Fallback: take everything from the first '{' to the last '}'
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = cleaned[start:end+1]
            else:
                json_str = cleaned

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Simple auto-repair for truncated JSON strings
            try:
                repaired_str = json_str.strip()
                if not repaired_str.endswith("}"):
                    if repaired_str.count('"') % 2 != 0:
                        repaired_str += '"'
                    repaired_str += "}"
                return json.loads(repaired_str)
            except Exception:
                # If repair fails, re-raise original exception or log cleanly
                raise e
