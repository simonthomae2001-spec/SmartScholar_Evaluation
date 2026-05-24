"""
gatekeeper_agent.py — Input validation gate (Steps 2 & 9).

The Gatekeeper inspects incoming user queries and decides whether they
are valid, researchable academic topics. Invalid queries are rejected
with a human-readable reason.

Current implementation: **stub** that unconditionally accepts all
queries. Replace the body of ``validate_input`` with LLM-based
classification when ready.
"""

from __future__ import annotations


class GatekeeperAgent:
    """Validates whether a user query is a legitimate research topic."""

    def validate_input(self, query: str) -> tuple[bool, str]:
        """
        Check whether *query* is a valid academic research question.

        Parameters
        ----------
        query : str
            The raw user input.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if the query is accepted.
            ``(False, reason)`` if the query is rejected.
        """
        # ---- STUB: accept everything -------------------------------- #
        return (True, "")
