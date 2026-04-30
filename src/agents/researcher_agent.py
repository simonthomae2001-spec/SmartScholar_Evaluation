"""
ResearcherAgent: Agentic query expansion and multi-query research orchestration.

This agent takes a user's natural-language research topic, uses an LLM to expand it
into 3-5 precise scientific search queries, executes each against the Semantic Scholar
API via ScholarTool, and deduplicates the aggregated results by paperId.
"""

import json
import re

from core.model_factory import ModelFactory
from tools.scholar_tool import ScholarTool


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
