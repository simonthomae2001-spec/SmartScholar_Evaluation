"""
orchestrator.py — LangGraph StateGraph for the SmartScholar pipeline.

Wires all agent nodes into a directed graph that covers all 16
conceptual steps of the multi-agent research system.

Graph topology
--------------
    START
      → gatekeeper_node
          ─── (invalid) ──→ END
          ─── (valid)   ──→ researcher_enhance_node
      → researcher_enhance_node → researcher_search_node
      → researcher_search_node  → ingestor_node
      → ingestor_node           → analyst_node
      → analyst_node            → critic_node
      → critic_node
          ─── (passed OR loops exhausted) ──→ synthesizer_node
          ─── (failed AND loops remain)   ──→ analyst_node
      → synthesizer_node → END

CRITICAL: All numeric limits (max_queries, top_n_papers, max_loops,
chunk_size, read_depth) are fetched dynamically from
``src.core.config.get_config()`` — NO magic numbers.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Generator

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from src.core.graph_state import GraphState
from src.core.config import get_config

from src.agents.gatekeeper_agent import GatekeeperAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.ingestor_agent import IngestorAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.critic_agent import CriticAgent
from src.agents.synthesizer_agent import SynthesizerAgent

# ------------------------------------------------------------------ #
#  Shared agent singletons (created once, reused across invocations)
# ------------------------------------------------------------------ #

_gatekeeper: GatekeeperAgent | None = None
_researcher: ResearcherAgent | None = None
_ingestor: IngestorAgent | None = None
_analyst: AnalystAgent | None = None
_critic: CriticAgent | None = None
_synthesizer: SynthesizerAgent | None = None


def _get_gatekeeper() -> GatekeeperAgent:
    global _gatekeeper
    if _gatekeeper is None:
        _gatekeeper = GatekeeperAgent()
    return _gatekeeper


def _get_researcher() -> ResearcherAgent:
    global _researcher
    if _researcher is None:
        _researcher = ResearcherAgent()
    return _researcher


def _get_ingestor() -> IngestorAgent:
    global _ingestor
    if _ingestor is None:
        _ingestor = IngestorAgent()
    return _ingestor


def _get_analyst() -> AnalystAgent:
    global _analyst
    if _analyst is None:
        _analyst = AnalystAgent()
    return _analyst


def _get_critic() -> CriticAgent:
    global _critic
    if _critic is None:
        _critic = CriticAgent()
    return _critic


def _get_synthesizer() -> SynthesizerAgent:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = SynthesizerAgent()
    return _synthesizer


# ------------------------------------------------------------------ #
#  Helper: resolve config from state
# ------------------------------------------------------------------ #

def _resolve_config(state: GraphState) -> dict:
    """Fetch the config dict for the profile stored in state."""
    profile = state.get("config_profile", "medium")
    return get_config(profile)


def _ui_log(config: dict, msg: str) -> None:
    """Push a trace message to the UI queue if it exists."""
    q = config.get("configurable", {}).get("ui_queue")
    if q:
        q.put(("ui_msg", msg))


# ------------------------------------------------------------------ #
#  Node functions — each returns a dict of state-key updates
# ------------------------------------------------------------------ #

def gatekeeper_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Steps 2 & 9 — Validate the user query.

    Sets ``is_valid``, ``validation_reason``, and Gatekeeper confirmation
    metadata used by the UI retry flow.
    """
    _ui_log(config, "⚙️ [System] Starting Gatekeeper Agent...")

    if state.get("gatekeeper_confirmed", False) and state.get(
            "gatekeeper_override_allowed", False
    ):
        reason = "Durch Benutzerbestätigung akzeptiert."
        _ui_log(config, f"✅ [Gatekeeper] Query accepted: {reason}")
        return {
            "is_valid": True,
            "validation_reason": reason,
            "gatekeeper_needs_confirmation": False,
            "gatekeeper_confirmed": True,
            "gatekeeper_severity": "none",
            "gatekeeper_can_override": False,
            "gatekeeper_follow_up_question": None,
            "gatekeeper_override_allowed": False,
        }

    if state.get("gatekeeper_confirmed", False):
        reason = "Diese Ablehnung kann nicht per Benutzerbestätigung überschrieben werden."
        _ui_log(config, f"❌ [Gatekeeper] Query rejected: {reason}")
        return {
            "is_valid": False,
            "validation_reason": reason,
            "gatekeeper_needs_confirmation": False,
            "gatekeeper_confirmed": False,
            "gatekeeper_severity": "security_critical",
            "gatekeeper_can_override": False,
            "gatekeeper_follow_up_question": None,
            "gatekeeper_override_allowed": False,
        }

    agent = _get_gatekeeper()
    decision = agent.evaluate_input(
        user_input=state.get("user_query", ""),
    )
    is_valid = bool(decision.get("is_valid", False))
    reason = str(decision.get("reason", ""))
    severity = str(decision.get("severity", "none" if is_valid else "content_issue"))
    can_override = bool(decision.get("can_override", False))
    follow_up_question = decision.get("follow_up_question")

    if severity == "security_critical":
        can_override = False

    needs_confirmation = bool(not is_valid and can_override)

    if is_valid:
        _ui_log(config, f"✅ [Gatekeeper] Query accepted: {reason}")
    elif needs_confirmation:
        _ui_log(config, f"❓ [Gatekeeper] Query needs confirmation: {reason}")
    else:
        _ui_log(config, f"❌ [Gatekeeper] Query rejected: {reason}")

    return {
        "is_valid": is_valid,
        "validation_reason": reason,
        "gatekeeper_needs_confirmation": needs_confirmation,
        "gatekeeper_confirmed": state.get("gatekeeper_confirmed", False),
        "gatekeeper_severity": severity,
        "gatekeeper_can_override": can_override,
        "gatekeeper_follow_up_question": follow_up_question,
        "gatekeeper_override_allowed": False,
    }


def researcher_enhance_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Step 3 — Expand the user query into refined academic search terms.

    The number of search phrases is strictly bounded by
    ``config["max_queries"]``.
    """
    _ui_log(config, "⚙️ [System] Starting Researcher Agent (Query Expansion)...")
    agent = _get_researcher()
    cfg = _resolve_config(state)

    queries, strategy = agent.enhance_prompt(
        user_query=state["user_query"],
        config=cfg,
    )

    if strategy:
        _ui_log(config, f"🧠 [Researcher] Strategy: {strategy}")
    _ui_log(config, f"✅ [System] Generated {len(queries)} search queries.")

    return {"search_queries": queries, "query_strategy": strategy}


def researcher_search_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Steps 5, 6, 7 — Execute multi-query search, score, and split
    papers into active vs. discarded sets.

    The split point is ``config["top_n_papers"]``.
    """
    _ui_log(config, "⚙️ [System] Starting Researcher Agent (Search)...")
    agent = _get_researcher()
    cfg = _resolve_config(state)

    def _agent_cb(msg: str):
        _ui_log(config, msg)

    # 1. Search
    papers = agent.execute_research(
        user_input=state["user_query"],
        queries=state["search_queries"],
        limit_per_query=cfg["top_n_papers"],
        status_callback=_agent_cb,
    )

    # 2. Score, rank, and split
    _ui_log(config, f"🏆 [System] Scoring & ranking {len(papers)} retrieved papers...")
    active, discarded = agent.evaluate_papers(papers, state["user_query"], cfg)

    _ui_log(config, f"✅ [System] Retained {len(active)} active papers ({len(discarded)} in reserve pool).")

    return {
        "active_papers": active,
        "discarded_papers": discarded,
    }


def ingestor_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Steps 10A, 10B, 11B — Ingest paper content at the configured
    ``read_depth`` with the configured ``chunk_size``.
    """
    _ui_log(config, "⚙️ [System] Starting Ingestor Agent...")
    agent = _get_ingestor()
    cfg = _resolve_config(state)

    # Bridge the agent's progress messages to the Streamlit UI queue, so the
    # UI doesn't freeze during (potentially slow) ingestion.
    def _agent_cb(msg: str):
        _ui_log(config, msg)

    papers = state.get("active_papers", [])
    ingested, vector_engine = agent.ingest_knowledge(papers, cfg, status_callback=_agent_cb)
    _ui_log(config, "✅ [System] Ingestion complete.")

    return {"active_papers": ingested, "vectorEngine": vector_engine}


def analyst_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Steps 12 & 13 — Produce structured analysis records for each paper.
    """
    _ui_log(config, "⚙️ [System] Starting Analyst Agent...")
    agent = _get_analyst()
    papers = state.get("active_papers", [])
    query = state.get("user_query", "")
    vector_engine = state.get("vectorEngine")

    def _agent_cb(msg: str):
        _ui_log(config, msg)

    analysis_data = agent.analyze_papers(papers, query, vector_engine, _agent_cb)
    _ui_log(config, "✅ [System] Analysis complete.")

    return {"paper_analysis_data": analysis_data}


def critic_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Steps 14 & 15 — Verify the analysis and decide whether to loop.
    """
    _ui_log(config, "⚙️ [System] Starting Critic Agent...")
    agent = _get_critic()
    cfg = _resolve_config(state)
    loop_count = state.get("loop_count", 0)
    analysis_data = state.get("paper_analysis_data", [])

    def _agent_cb(msg: str):
        _ui_log(config, msg)

    passed, feedback = agent.verify_facts(
        analysis_data, cfg, loop_count, status_callback=_agent_cb,
    )

    updates: dict = {
        "critic_feedback": feedback,
        "loop_count": loop_count + 1,
    }

    # Store the pass/fail decision for conditional routing
    updates["_critic_passed"] = passed
    return updates


def synthesizer_node(state: GraphState, config: RunnableConfig) -> dict:
    """Step 16 — Compose the final literature review from approved claims."""
    def _agent_cb(msg: str):
        _ui_log(config, msg)

    _agent_cb("📝 [System] Starting final synthesis from real data...")

    agent = _get_synthesizer()

    agent_output = agent.synthesize_review(state, config, status_callback=_agent_cb)

    _agent_cb("✅ [System] Test-synthesis finalized sucessfully.")

    return agent_output


# ------------------------------------------------------------------ #
#  Conditional routing functions
# ------------------------------------------------------------------ #

def _route_after_gatekeeper(state: GraphState) -> str:
    """Route to researcher or END based on gatekeeper verdict."""
    if state.get("is_valid", False):
        return "researcher_enhance_node"
    return END


def _route_after_critic(state: GraphState) -> str:
    """
    Route to analyst (loop back) or synthesizer based on critic result.

    Uses ``config["max_loops"]`` from the profile — no hardcoded limit.
    """
    passed = state.get("_critic_passed", False)
    if passed:
        return "synthesizer_node"

    # Check loop budget
    config = _resolve_config(state)
    loop_count = state.get("loop_count", 0)

    if loop_count < config["max_loops"]:
        return "analyst_node"

    # Budget exhausted → proceed to synthesis
    return "synthesizer_node"


# ------------------------------------------------------------------ #
#  Graph construction
# ------------------------------------------------------------------ #

def build_enhance_graph():
    """
    Construct a **partial** graph used by the Streamlit Human-in-the-Loop
    UI for the query-generation step only.

    Topology:  START → gatekeeper → researcher_enhance → END

    The remaining nodes (search, ingest, analyst, critic, synthesizer)
    are called **imperatively** from ``app.py`` after human review.
    """
    graph = StateGraph(GraphState)

    graph.add_node("gatekeeper_node", gatekeeper_node)
    graph.add_node("researcher_enhance_node", researcher_enhance_node)

    graph.add_edge(START, "gatekeeper_node")
    graph.add_conditional_edges(
        "gatekeeper_node",
        _route_after_gatekeeper,
        {
            "researcher_enhance_node": "researcher_enhance_node",
            END: END,
        },
    )
    graph.add_edge("researcher_enhance_node", END)

    return graph.compile()


def build_regenerate_graph():
    """
    Construct a **partial** graph used by the Streamlit UI to regenerate 
    queries directly, skipping the gatekeeper.
    
    Topology: START -> researcher_enhance_node -> END
    """
    graph = StateGraph(GraphState)

    graph.add_node("researcher_enhance_node", researcher_enhance_node)

    graph.add_edge(START, "researcher_enhance_node")
    graph.add_edge("researcher_enhance_node", END)

    return graph.compile()


def build_search_graph():
    """
    Construct a **partial** graph used by the Streamlit UI to run just
    the multi-query search.
    """
    graph = StateGraph(GraphState)
    graph.add_node("researcher_search_node", researcher_search_node)
    graph.add_edge(START, "researcher_search_node")
    graph.add_edge("researcher_search_node", END)
    return graph.compile()


def build_analysis_graph():
    """
    Construct a **partial** graph used by the Streamlit UI after the user has
    finalised the papers: first ingest the selected papers into ChromaDB,
    then run the analyst, critic, and synthesizer.

    Topology: START → ingestor_node → analyst_node → critic_node ↺ synthesizer_node → END
    """
    graph = StateGraph(GraphState)
    graph.add_node("ingestor_node", ingestor_node)
    graph.add_node("analyst_node", analyst_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("synthesizer_node", synthesizer_node)
    
    graph.add_edge(START, "ingestor_node")
    graph.add_edge("ingestor_node", "analyst_node")
    graph.add_edge("analyst_node", "critic_node")
    graph.add_conditional_edges(
        "critic_node",
        _route_after_critic,
        {
            "analyst_node": "analyst_node",
            "synthesizer_node": "synthesizer_node",
        },
    )
    graph.add_edge("synthesizer_node", END)
    
    return graph.compile()


def build_synthesizer_graph():
    """
    Builds one isolated, linear graph just for genereating texts.
    """
    builder = StateGraph(GraphState)

    # Node registrieren
    builder.add_node("synthesizer_node", synthesizer_node)

    # Linearen Ablauf definieren: START -> Synthesizer -> END
    builder.add_edge(START, "synthesizer_node")
    builder.add_edge("synthesizer_node", END)

    return builder.compile()


def build_graph():
    """
    Construct and compile the **full** SmartScholar LangGraph covering
    all 16 steps (autonomous mode).

    Returns
    -------
    CompiledStateGraph
        A compiled graph ready to be invoked via ``graph.invoke(state)``.
    """
    graph = StateGraph(GraphState)

    # ---- Register all nodes -------------------------------------- #
    graph.add_node("gatekeeper_node", gatekeeper_node)
    graph.add_node("researcher_enhance_node", researcher_enhance_node)
    graph.add_node("researcher_search_node", researcher_search_node)
    graph.add_node("ingestor_node", ingestor_node)
    graph.add_node("analyst_node", analyst_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("synthesizer_node", synthesizer_node)

    # ---- Edges --------------------------------------------------- #

    # START → Gatekeeper
    graph.add_edge(START, "gatekeeper_node")

    # Gatekeeper → (conditional) Researcher or END
    graph.add_conditional_edges(
        "gatekeeper_node",
        _route_after_gatekeeper,
        {
            "researcher_enhance_node": "researcher_enhance_node",
            END: END,
        },
    )

    # Researcher enhance → Researcher search
    graph.add_edge("researcher_enhance_node", "researcher_search_node")

    # Researcher search → Ingestor
    graph.add_edge("researcher_search_node", "ingestor_node")

    # Ingestor → Analyst
    graph.add_edge("ingestor_node", "analyst_node")

    # Analyst → Critic
    graph.add_edge("analyst_node", "critic_node")

    # Critic → (conditional) Analyst (loop) or Synthesizer
    graph.add_conditional_edges(
        "critic_node",
        _route_after_critic,
        {
            "analyst_node": "analyst_node",
            "synthesizer_node": "synthesizer_node",
        },
    )

    # Synthesizer → END
    graph.add_edge("synthesizer_node", END)

    return graph.compile()


# ------------------------------------------------------------------ #
#  Streaming Wrappers for Streamlit UI
# ------------------------------------------------------------------ #

def _run_graph_with_queue(graph, state: GraphState) -> Generator[str | dict, None, None]:
    """
    Helper to run a graph in a background thread and yield real-time 
    string messages from the `ui_queue`, finally yielding the GraphState dict.
    """
    q = queue.Queue()
    final_state_holder = []

    def worker():
        try:
            # Pass the queue to the nodes via config
            cfg = {"configurable": {"ui_queue": q}}
            # Graph.invoke will block until finished, but nodes will push to queue
            result = graph.invoke(state, config=cfg)
            final_state_holder.append(result)
        except Exception as e:
            q.put(("error", e))
        finally:
            q.put(("done", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        msg_type, msg = q.get()
        if msg_type == "ui_msg":
            yield msg
        elif msg_type == "error":
            raise msg
        elif msg_type == "done":
            break

    if final_state_holder:
        yield final_state_holder[0]


def stream_enhance_flow(state: GraphState, regenerate: bool = False) -> Generator[str | dict, None, None]:
    """Stream the gatekeeper + enhance nodes."""
    if regenerate:
        graph = build_regenerate_graph()
    else:
        graph = build_enhance_graph()
    yield from _run_graph_with_queue(graph, state)


def stream_search_flow(state: GraphState) -> Generator[str | dict, None, None]:
    """Stream the search + evaluate nodes."""
    graph = build_search_graph()
    yield from _run_graph_with_queue(graph, state)


def stream_analysis_flow(state: GraphState) -> Generator[str | dict, None, None]:
    """Stream the analyst stub node."""
    graph = build_analysis_graph()
    yield from _run_graph_with_queue(graph, state)


def stream_synthesis_flow(state: GraphState) -> Generator[str | dict, None, None]:
    """Stream the analyst stub node."""
    graph = build_synthesizer_graph()
    yield from _run_graph_with_queue(graph, state)