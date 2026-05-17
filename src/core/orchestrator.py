"""
orchestrator.py — LangGraph StateGraph for the SmartScholar pipeline.

Wires together all agent nodes into a directed graph.  The *static*
graph currently runs:

    START → gatekeeper_node → researcher_enhance_node → END

Additional nodes (``researcher_search_node``, ``analyst_node``) are
defined here but invoked **imperatively** from the Streamlit UI after
human review steps.  This keeps the Human-in-the-Loop control loop
inside Streamlit while still flowing data through the LangGraph state.

Replacing a placeholder
-----------------------
To swap in a real agent for ``gatekeeper_node`` or ``analyst_node``:

1. Import your agent class at the top of this file.
2. Replace the body of the corresponding ``*_node`` function.
3. Make sure your function signature is ``(state: GraphState) -> dict``
   and returns a dict of the state keys you want to update.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.core.graph_state import GraphState, ResearchConfig
from src.agents.researcher_agent import ResearcherAgent


# ------------------------------------------------------------------ #
#  Shared agent instance (created once, reused across invocations)
# ------------------------------------------------------------------ #

_researcher: ResearcherAgent | None = None


def _get_researcher() -> ResearcherAgent:
    """Lazy-initialise the ResearcherAgent singleton."""
    global _researcher
    if _researcher is None:
        _researcher = ResearcherAgent()
    return _researcher


# ------------------------------------------------------------------ #
#  Node functions
# ------------------------------------------------------------------ #

def gatekeeper_node(state: GraphState) -> dict:
    """
    **PLACEHOLDER — GatekeeperAgent stub.**

    In production this node should:
    • Classify whether the user query is a valid, researchable academic
      topic.
    • Set ``is_valid = False`` and optionally populate a rejection
      reason if the query is off-topic, too vague, or harmful.

    For now it unconditionally approves every query.

    Parameters
    ----------
    state : GraphState

    Returns
    -------
    dict
        Updated state keys: ``is_valid``, ``research_config``.
    """
    profile = state.get("config_profile", "medium")
    config = ResearchConfig.from_profile(profile)
    return {
        "is_valid": True,
        "research_config": config.to_dict(),
    }


def researcher_enhance_node(state: GraphState) -> dict:
    """
    Use the ResearcherAgent to expand the user query into refined
    academic search terms.

    Reads
    -----
    ``user_query``, ``research_config``

    Writes
    ------
    ``search_queries``
    """
    agent = _get_researcher()
    config = ResearchConfig.from_dict(state["research_config"])
    queries = agent.enhance_prompt(
        user_query=state["user_query"],
        profile_config=config,
    )
    return {"search_queries": queries}


def researcher_search_node(state: GraphState) -> dict:
    """
    Execute multi-query search against Semantic Scholar and score /
    rank all retrieved papers.

    This node is **not** wired into the static graph — it is called
    imperatively from the Streamlit UI after the human has reviewed
    and accepted the search queries.

    Reads
    -----
    ``user_query``, ``search_queries``, ``research_config``

    Writes
    ------
    ``active_papers``, ``discarded_papers``
    """
    agent = _get_researcher()
    config = ResearchConfig.from_dict(state["research_config"])

    # 1. Search
    papers = agent.execute_research(
        user_input=state["user_query"],
        queries=state["search_queries"],
        limit_per_query=config.results_per_query,
    )

    # 2. Score & rank
    scored = agent.evaluate_papers(papers, state["user_query"])

    # 3. Split into active vs. discarded
    active = scored[: config.active_paper_count]
    discarded = scored[config.active_paper_count:]

    return {
        "active_papers": active,
        "discarded_papers": discarded,
    }


def analyst_node(state: GraphState) -> dict:
    """
    **PLACEHOLDER — AnalystAgent stub.**

    In production this node should:
    • Receive the curated ``active_papers`` list.
    • Synthesise a full literature review citing each paper with its
      assigned citation ID (``[1]``, ``[2]``, …).
    • Write the review text to ``final_review``.

    For now it generates a simple fallback summary so the app runs
    end-to-end.

    Parameters
    ----------
    state : GraphState

    Returns
    -------
    dict
        Updated state key: ``final_review``.
    """
    papers = state.get("active_papers", [])
    if not papers:
        return {"final_review": "No papers were selected for the review."}

    lines = ["# Literature Review (Auto-Generated Stub)\n"]
    lines.append(
        "> **Note:** This is a placeholder summary. Replace the "
        "`analyst_node` in `orchestrator.py` with a real AnalystAgent "
        "to get a full synthesis.\n"
    )
    for p in papers:
        cid = p.get("citation_id", "")
        title = p.get("title", "Untitled")
        year = p.get("year", "n.d.")
        abstract = (p.get("abstract") or "No abstract available.")[:200]
        lines.append(f"**{cid} {title}** ({year})")
        lines.append(f"> {abstract}…\n")

    return {"final_review": "\n".join(lines)}


# ------------------------------------------------------------------ #
#  Graph construction
# ------------------------------------------------------------------ #

def build_graph() -> StateGraph:
    """
    Construct and compile the SmartScholar LangGraph.

    Returns
    -------
    langgraph.graph.StateGraph
        A compiled graph ready to be invoked.
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("gatekeeper_node", gatekeeper_node)
    graph.add_node("researcher_enhance_node", researcher_enhance_node)

    # Edges
    graph.add_edge(START, "gatekeeper_node")
    graph.add_edge("gatekeeper_node", "researcher_enhance_node")
    graph.add_edge("researcher_enhance_node", END)

    return graph.compile()
