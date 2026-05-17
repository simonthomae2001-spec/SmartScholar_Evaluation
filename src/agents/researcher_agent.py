"""
ResearcherAgent: Agentic query expansion and multi-query research orchestration.

This agent takes a user's natural-language research topic, uses an LLM to expand it
into 3-5 precise scientific search queries, executes each against the Semantic Scholar
API via ScholarTool, and deduplicates the aggregated results by paperId.

Extended with:
    enhance_prompt()      – profile-aware wrapper around generate_search_queries.
    evaluate_papers()     – LLM-based relevance scoring (0-100) with heuristic fallback.
"""

import json
import re
from datetime import datetime

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
        "Given a user's research topic, generate a JSON array of 3 to 5 highly "
        "specific, scientific search queries that would surface the most relevant "
        "papers on the Semantic Scholar API.\n\n"
        "Rules:\n"
        "- Each query should target a different facet, sub-topic, or methodology "
        "related to the user's topic.\n"
        "- Use precise scientific terminology.\n"
        "- Return ONLY a valid JSON array of strings — no markdown, no explanation.\n\n"
        "Example output:\n"
        '[\"deep reinforcement learning for robotic manipulation\", '
        '\"sim-to-real transfer policy learning\", '
        '\"sample efficient model-based RL robotics\"]\n\n'
        "User topic: {user_input}\n\n"
        "JSON array:"
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
        "Return ONLY a valid JSON array of objects with keys \"id\" (the paper index) "
        "and \"score\" (integer 0-100). No markdown, no explanation.\n\n"
        "JSON array:"
    )

    def __init__(self):
        self.llm = ModelFactory.get_model()

    # ------------------------------------------------------------------ #
    #  Query expansion via LLM
    # ------------------------------------------------------------------ #
    def generate_search_queries(self, user_input: str) -> list[str]:
        """
        Prompt the LLM to produce 3-5 refined search queries for the given
        topic. Falls back to the original user input on any parsing error.
        """
        prompt = self.QUERY_EXPANSION_PROMPT.format(user_input=user_input)

        try:
            response = self.llm.complete(prompt)
            raw_text = str(response).strip()

            # Try to extract JSON array from the response
            queries = self._parse_json_array(raw_text)

            if queries and isinstance(queries, list) and len(queries) >= 1:
                # Enforce the 3-5 range
                return [str(q) for q in queries[:5]]

        except Exception as e:
            print(f"[ResearcherAgent] Query expansion failed: {e}")

        # Fallback: use the original query as-is
        return [user_input]

    # ------------------------------------------------------------------ #
    #  Profile-aware prompt enhancement  (NEW)
    # ------------------------------------------------------------------ #
    def enhance_prompt(self, user_query: str, profile_config) -> list[str]:
        """
        Expand a user query into refined academic search terms, respecting
        the profile's ``max_queries`` cap.

        This is the primary entry-point used by the LangGraph
        ``researcher_enhance_node``.

        Parameters
        ----------
        user_query : str
            The raw topic the user typed.
        profile_config : ResearchConfig
            A :class:`~src.core.graph_state.ResearchConfig` instance that
            carries the ``max_queries`` limit for the active profile.

        Returns
        -------
        list[str]
            Refined search queries (length ≤ ``profile_config.max_queries``).
        """
        queries = self.generate_search_queries(user_query)
        return queries[: profile_config.max_queries]

    # ------------------------------------------------------------------ #
    #  Full research pipeline
    # ------------------------------------------------------------------ #
    def execute_research(
        self,
        user_input: str,
        queries: list[str] | None = None,
        limit_per_query: int = 5,
        status_callback=None,
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
    ) -> list[dict]:
        """
        Score and rank papers by relevance to the user's research topic.

        Attempts an LLM-based batch evaluation first.  If that fails
        (timeout, JSON parse error, etc.), falls back to a deterministic
        heuristic that combines keyword overlap, recency, and citation
        count.

        Each paper dict gets a new key ``relevance_score`` (int, 0-100).
        The returned list is sorted **descending** by score.

        Parameters
        ----------
        papers : list[dict]
            Raw paper dicts from the Semantic Scholar API.
        user_query : str
            The original user topic (used for relevance comparison).

        Returns
        -------
        list[dict]
            The same papers, augmented with ``relevance_score`` and sorted.
        """
        if not papers:
            return []

        # Try LLM-based scoring first
        try:
            scored = self._llm_score(papers, user_query)
            if scored:
                return scored
        except Exception as e:
            print(f"[ResearcherAgent] LLM scoring failed, using heuristic: {e}")

        # Fallback: heuristic scoring
        return self._heuristic_score(papers, user_query)

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

        # Map scores back onto the original paper dicts
        score_map: dict[int, int] = {}
        for entry in scores_list:
            if isinstance(entry, dict) and "id" in entry and "score" in entry:
                score_map[int(entry["id"])] = int(entry["score"])

        for idx, p in enumerate(papers):
            p["relevance_score"] = max(0, min(100, score_map.get(idx, 50)))

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

            p["relevance_score"] = kw_score + recency_score + cite_score

        papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        return papers

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json_array(text: str) -> list | None:
        """
        Robustly extract a JSON array from LLM output that may contain
        markdown fences or surrounding prose.
        """
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.strip("`").strip()

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
