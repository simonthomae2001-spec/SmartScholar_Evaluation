"""
graph_state.py — Central state schema for the SmartScholar LangGraph pipeline.

Defines:
    GraphState      – the comprehensive TypedDict that flows through
                      every LangGraph node across all 16 pipeline steps.
"""

from __future__ import annotations

from typing import TypedDict
from src.core.vector_store import VectorEngine

# ------------------------------------------------------------------ #
#  Comprehensive LangGraph state (all 16 steps)
# ------------------------------------------------------------------ #

class GraphState(TypedDict, total=False):
    """
    The shared state dictionary that travels through every LangGraph
    node in the SmartScholar pipeline.

    Covers all 16 conceptual steps of the multi-agent system.

    Attributes
    ----------
    user_query : str
        The raw research topic entered by the user.
    config_profile : str
        One of ``"fast"``, ``"medium"``, ``"pro"``.
    is_valid : bool
        Set by the Gatekeeper. ``True`` → query is acceptable.
    validation_reason : str
        Human-readable reason if the query was rejected.
    gatekeeper_needs_confirmation : bool
        ``True`` if the Gatekeeper rejected a safe but unclear request and
        the UI may offer a confirmation retry.
    gatekeeper_confirmed : bool
        ``True`` when the user explicitly confirmed an unclear request.
    gatekeeper_severity : str
        ``"none"``, ``"content_issue"``, or ``"security_critical"``.
    gatekeeper_can_override : bool
        ``True`` only when the Gatekeeper rejection may be overridden by
        explicit user confirmation.
    gatekeeper_follow_up_question : str
        Optional user-facing follow-up question for overrideable content issues.
    gatekeeper_override_allowed : bool
        ``True`` only for a retry that was created from an overrideable prior
        Gatekeeper decision.
    search_queries : list[str]
        Refined academic search terms produced by the Researcher.
    query_strategy : str
        Agent's internal monologue explaining why it chose specific
        search angles / facets for query expansion.
    active_papers : list[dict]
        Papers selected for deep analysis.
    discarded_papers : list[dict]
        Papers that scored below the cut-off.
    paper_analysis_data : list[dict]
        Structured analysis output from the Analyst.
    critic_feedback : str
        Textual feedback from the Critic agent.
    loop_count : int
        Current critic→analyst iteration count.
    final_review : str
        The literature review / synthesis produced by the Synthesizer.
    """

    user_query: str
    config_profile: str
    is_valid: bool
    validation_reason: str
    gatekeeper_needs_confirmation: bool
    gatekeeper_confirmed: bool
    gatekeeper_severity: str
    gatekeeper_can_override: bool
    gatekeeper_follow_up_question: str
    gatekeeper_override_allowed: bool
    search_queries: list[str]
    query_strategy: str
    active_papers: list[dict]
    discarded_papers: list[dict]
    vectorEngine: VectorEngine
    paper_analysis_data: list[dict]
    critic_feedback: str
    loop_count: int
    final_review: str
