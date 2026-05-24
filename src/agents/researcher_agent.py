"""
ResearcherAgent: Agentic query expansion and multi-query research orchestration.

This agent takes a user's natural-language research topic, uses an LLM to expand it
into refined scientific search queries, executes each against the Semantic Scholar
API via ScholarTool, and deduplicates the aggregated results by paperId.

Key capabilities:
    enhance_prompt()      – profile-aware query expansion with strategy trace.
    evaluate_papers()     – LLM-based relevance scoring (0-100) with per-paper
                            rationale and heuristic fallback.
"""

import json
import re
from datetime import datetime
from typing import Any

from src.core.model_factory import ModelFactory
from src.tools.scholar_tool import ScholarTool


class ResearcherAgent:
    """
    An agentic researcher that expands a user query into multiple refined
    scientific search terms, retrieves papers for each, and deduplicates.
    """

    # ------------------------------------------------------------------ #
    #  System prompt for query expansion
    # ------------------------------------------------------------------ #
    QUERY_EXPANSION_PROMPT = (
        "You are a research assistant specializing in academic literature search. "
        "Given a user's research topic, generate EXACTLY {num_queries} highly "
        "specific, scientific search queries that would surface the most relevant "
        "papers on the Semantic Scholar API.\n\n"
        "Rules:\n"
        "- Each query should target a different facet, sub-topic, or methodology "
        "related to the user's topic.\n"
        "- Use concise academic concepts, NOT full sentences.\n"
        "- Use precise scientific terminology.\n"
        "- Return ONLY a valid JSON object — no markdown, no explanation.\n\n"
        "The JSON object MUST have exactly two keys:\n"
        '  "strategy": A 1-2 sentence internal monologue explaining why you '
        "chose these specific search angles.\n"
        '  "queries": A JSON array of EXACTLY {num_queries} query strings.\n\n'
        "Example output (Note: you MUST generate EXACTLY {num_queries} queries, not 3!):\n"
        '{{\n'
        '  "strategy": "I focus on the core algorithm, its real-world deployment '
        'challenges, and data-efficiency improvements to cover theory, practice, '
        'and scalability.",\n'
        '  "queries": [\n'
        '    "deep reinforcement learning robotic manipulation",\n'
        '    "sim-to-real transfer policy learning",\n'
        '    "sample efficient model-based RL robotics"\n'
        '  ]\n'
        '}}\n\n'
        "User topic: {user_input}\n\n"
        "JSON object:"
    )

    # ------------------------------------------------------------------ #
    #  System prompt for paper relevance scoring
    # ------------------------------------------------------------------ #
    PAPER_SCORING_PROMPT = (
        "You are an expert academic reviewer. Given a user's research topic and a "
        "list of papers, score each paper's relevance from 0 to 100.\n\n"
        "Scoring criteria:\n"
        "- Topical relevance to the user's query (50 points)\n"
        "- Recency / year of publication (15 points)\n"
        "- Abstract quality and depth (20 points)\n"
        "- Citation impact — higher citation count is better (15 points)\n\n"
        "User topic: {user_query}\n\n"
        "Papers (JSON array of objects with keys id, title, abstract, year, citationCount):\n"
        "{papers_json}\n\n"
        "Return ONLY a valid JSON array of objects with keys:\n"
        '  "id" (the paper index, integer),\n'
        '  "score" (integer 0-100),\n'
        '  "rationale" (1 sentence explaining the score).\n\n'
        "No markdown, no explanation outside the JSON.\n\n"
        "JSON array:"
    )

    def __init__(self):
        self.llm = ModelFactory.get_model()

    # ------------------------------------------------------------------ #
    #  Query expansion via LLM
    # ------------------------------------------------------------------ #
    def generate_search_queries(
        self, user_input: str, num_queries: int
    ) -> tuple[list[str], str]:
        """
        Prompt the LLM to produce *num_queries* refined search queries for
        the given topic, together with a strategy explanation.

        Returns
        -------
        tuple[list[str], str]
            ``(queries, strategy)`` — the list of search strings and
            a 1-2 sentence explanation of the chosen search angles.
            Falls back to ``([user_input], fallback_message)`` on
            any parsing error.
        """
        prompt = self.QUERY_EXPANSION_PROMPT.format(
            user_input=user_input,
            num_queries=num_queries,
        )

        fallback_strategy = "Fallback: using the original query directly."

        try:
            response = self.llm.complete(prompt)
            raw_text = str(response).strip()

            # Try to extract JSON object with strategy + queries
            parsed = self._parse_json_object(raw_text)

            if parsed and isinstance(parsed, dict):
                strategy = str(parsed.get("strategy", "")).strip()
                queries_raw = parsed.get("queries", [])

                if isinstance(queries_raw, list) and len(queries_raw) >= 1:
                    queries = [str(q) for q in queries_raw[:num_queries]]
                    return (queries, strategy or fallback_strategy)

            # Legacy fallback: maybe the LLM returned a plain array
            queries = self._parse_json_array(raw_text)
            if queries and isinstance(queries, list) and len(queries) >= 1:
                return (
                    [str(q) for q in queries[:num_queries]],
                    fallback_strategy,
                )

        except Exception as e:
            print(f"[ResearcherAgent] Query expansion failed: {e}")

        # Fallback: use the original query as-is
        return ([user_input], fallback_strategy)

    # ------------------------------------------------------------------ #
    #  Profile-aware prompt enhancement
    # ------------------------------------------------------------------ #
    def enhance_prompt(
        self, user_query: str, config: dict
    ) -> tuple[list[str], str]:
        """
        Expand a user query into refined academic search terms, respecting
        the profile's ``max_queries`` cap.

        This is the primary entry-point used by the LangGraph
        ``researcher_enhance_node``.

        Parameters
        ----------
        user_query : str
            The raw topic the user typed.
        config : dict
            The profile config dict from ``get_config()``.
            Must contain ``max_queries``.

        Returns
        -------
        tuple[list[str], str]
            ``(queries, strategy)`` — refined search queries
            (length ≤ ``config["max_queries"]``) and the agent's
            internal strategy monologue.
        """
        max_q = config["max_queries"]
        queries, strategy = self.generate_search_queries(
            user_query, num_queries=max_q
        )
        return (queries[:max_q], strategy)

    # ------------------------------------------------------------------ #
    #  Full research pipeline
    # ------------------------------------------------------------------ #
    def execute_research(
        self,
        user_input: str,
        queries: list[str] | None,
        limit_per_query: int,
        status_callback: Any | None = None,
    ) -> list[dict]:
        """
        Run the full search pipeline:
        1. (Optionally) generate refined queries (or accept pre-generated ones).
        2. Search Semantic Scholar for each query.
        3. Deduplicate by paperId.

        Parameters
        ----------
        user_input : str
            The original user question / topic.
        queries : list[str] | None
            Pre-generated queries. If None, generate_search_queries is called.
        limit_per_query : int
            Number of results to request per query from the API.
        status_callback : callable | None
            An optional function(msg: str) that will be called with status
            messages (useful for Streamlit st.write).

        Returns
        -------
        list[dict]  — deduplicated papers.
        """

        if queries is None:
            queries = self.generate_search_queries(user_input)

        all_papers: list[dict] = []
        seen_ids: set[str] = set()

        for i, q in enumerate(queries, start=1):
            if status_callback:
                status_callback(f"🔎 **Query {i}/{len(queries)}:** `{q}`")

            results = ScholarTool.search_papers(q, limit=limit_per_query)

            for paper in results:
                pid = paper.get("paperId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(paper)

            if status_callback:
                new_count = len(all_papers)
                status_callback(
                    f"   ↳ Found {len(results)} results, "
                    f"**{new_count} unique papers** so far."
                )

        if status_callback:
            status_callback(
                f"✅ **Deduplication complete — {len(all_papers)} unique papers collected.**"
            )

        return all_papers

    # ------------------------------------------------------------------ #
    #  Paper evaluation & relevance scoring  (NEW)
    # ------------------------------------------------------------------ #
    def evaluate_papers(
        self,
        papers: list[dict],
        user_query: str,
        config: dict,
    ) -> tuple[list[dict], list[dict]]:
        """
        Score and rank papers by relevance, then split into active vs.
        discarded sets.

        Attempts an LLM-based batch evaluation first.  If that fails
        (timeout, JSON parse error, etc.), falls back to a deterministic
        heuristic that combines keyword overlap, recency, and citation
        count.

        Each paper dict gets a new key ``relevance_score`` (int, 0-100).
        The returned lists are sorted **descending** by score.

        Parameters
        ----------
        papers : list[dict]
            Raw paper dicts from the Semantic Scholar API.
        user_query : str
            The original user topic (used for relevance comparison).
        config : dict
            Profile config dict (from ``get_config()``).
            Must contain ``top_n_papers``.

        Returns
        -------
        tuple[list[dict], list[dict]]
            ``(active_papers, discarded_papers)`` where
            ``active_papers`` is the top ``config["top_n_papers"]``.
        """
        if not papers:
            return ([], [])

        # Try LLM-based scoring first
        try:
            scored = self._llm_score(papers, user_query)
            if scored:
                top_n = config["top_n_papers"]
                return (scored[:top_n], scored[top_n:])
        except Exception as e:
            print(f"[ResearcherAgent] LLM scoring failed, using heuristic: {e}")

        # Fallback: heuristic scoring
        scored = self._heuristic_score(papers, user_query)
        top_n = config["top_n_papers"]
        return (scored[:top_n], scored[top_n:])

    # ---- LLM-based scorer ----------------------------------------- #

    def _llm_score(
        self, papers: list[dict], user_query: str
    ) -> list[dict] | None:
        """
        Send a batch prompt to the LLM asking it to score every paper.
        Returns the scored + sorted list, or ``None`` on failure.
        """
        # Build a compact JSON representation for the prompt
        compact = []
        for idx, p in enumerate(papers):
            compact.append({
                "id": idx,
                "title": p.get("title", ""),
                "abstract": (p.get("abstract") or "")[:300],
                "year": p.get("year"),
                "citationCount": p.get("citationCount", 0),
            })

        prompt = self.PAPER_SCORING_PROMPT.format(
            user_query=user_query,
            papers_json=json.dumps(compact, ensure_ascii=False),
        )

        response = self.llm.complete(prompt)
        raw = str(response).strip()
        scores_list = self._parse_json_array(raw)

        if not scores_list or not isinstance(scores_list, list):
            return None

        # Map scores and rationales back onto the original paper dicts
        score_map: dict[int, int] = {}
        rationale_map: dict[int, str] = {}
        for entry in scores_list:
            if isinstance(entry, dict) and "id" in entry and "score" in entry:
                score_map[int(entry["id"])] = int(entry["score"])
                rationale_map[int(entry["id"])] = str(
                    entry.get("rationale", "")
                )

        for idx, p in enumerate(papers):
            p["relevance_score"] = max(0, min(100, score_map.get(idx, 50)))
            p["score_rationale"] = rationale_map.get(idx, "")

        papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        return papers

    # ---- Heuristic scorer ----------------------------------------- #

    @staticmethod
    def _heuristic_score(
        papers: list[dict], user_query: str
    ) -> list[dict]:
        """
        Deterministic fallback scorer.

        Components (total = 100):
        - Keyword overlap with query    → up to 50 pts
        - Recency (years since 2024)    → up to 25 pts
        - Citation count (log-scaled)   → up to 25 pts

        Also generates a synthetic ``score_rationale`` string from
        the component breakdown.
        """
        import math

        query_tokens = set(user_query.lower().split())
        current_year = datetime.now().year

        for p in papers:
            # --- keyword overlap (0-50) ---
            title = (p.get("title") or "").lower()
            abstract = (p.get("abstract") or "").lower()
            combined_tokens = set(title.split()) | set(abstract.split())
            if query_tokens:
                overlap = len(query_tokens & combined_tokens) / len(query_tokens)
            else:
                overlap = 0.0
            kw_score = min(50, int(overlap * 50))

            # --- recency (0-25) ---
            year = p.get("year") or (current_year - 5)
            age = max(0, current_year - int(year))
            recency_score = max(0, 25 - age * 3)

            # --- citations (0-25) ---
            cites = p.get("citationCount") or 0
            cite_score = min(25, int(math.log2(max(1, cites)) * 3))

            total = kw_score + recency_score + cite_score
            p["relevance_score"] = total
            p["score_rationale"] = (
                f"Heuristic — Keyword overlap: {kw_score}/50, "
                f"Recency: {recency_score}/25, "
                f"Citations: {cite_score}/25"
            )

        papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        return papers

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        return text.strip("`").strip()

    @staticmethod
    def _parse_json_array(text: str) -> list | None:
        """
        Robustly extract a JSON array from LLM output that may contain
        markdown fences or surrounding prose.
        """
        text = ResearcherAgent._strip_markdown_fences(text)

        # Try direct parse first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to locate a JSON array substring
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        """
        Robustly extract a JSON object from LLM output that may contain
        markdown fences or surrounding prose.
        """
        text = ResearcherAgent._strip_markdown_fences(text)

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
