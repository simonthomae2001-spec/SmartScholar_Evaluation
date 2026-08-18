"""
SmartScholar — Agentic Research Copilot (Human-in-the-Loop UI)

This Streamlit application drives a step-by-step research workflow
orchestrated by LangGraph.  The user reviews and refines search
queries, curates the paper selection, and controls when to advance
to the next stage.

Workflow steps (mapped to ``st.session_state.workflow_step``):
    idle          → waiting for input
    enhancing     → LangGraph graph is running (gatekeeper → enhance)
    query_review  → user edits / accepts generated queries
    searching     → Semantic Scholar search + scoring in progress
    paper_review  → user curates the ranked paper list
    done          → final selection with citation IDs assigned
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
from datetime import datetime
import json

# ------------------------------------------------------------------ #
#  Path setup
# ------------------------------------------------------------------ #
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.core.model_factory import ModelFactory
from src.core.graph_state import GraphState
from src.core.config import get_config
from src.core.orchestrator import (
    stream_enhance_flow,
    stream_search_flow,
    stream_analysis_flow,
    stream_synthesis_flow
)

from src.core.vector_store import clear_collection

# ------------------------------------------------------------------ #
#  Page configuration
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="SmartScholar",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Gemini Aesthetics — Custom CSS
# ------------------------------------------------------------------ #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 600;
        background: linear-gradient(75deg, #1a73e8, #8ab4f8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 16px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
        background-color: #ffffff;
        padding: 1.5rem;
    }

    /* User avatar icon — match the blue brand palette */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #1a73e8 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        background-color: #1a73e8 !important;
    }

    /* Modern Polish for Status & Expander */
    .stStatusContainer, .stExpander {
        padding: 1.2rem !important;
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #dadce0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .score-high   { background: #e6f4ea; color: #1e7e34; }
    .score-mid    { background: #fef7e0; color: #b45309; }
    .score-low    { background: #fce8e6; color: #c5221f; }

    /* Profile chip */
    .profile-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        background: linear-gradient(135deg, #e8eaf6, #c5cae9);
        color: #283593;
        margin-top: 4px;
    }

    /* Step indicator */
    .step-indicator {
        text-align: center;
        padding: 8px 0;
        color: #5f6368;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .step-indicator .current-step {
        color: #1a73e8;
        font-weight: 600;
    }

    /* Limit max height of trace elements to make them scrollable */
    div[data-testid="stExpanderDetails"]:has(.trace-marker) {
        max-height: 400px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        resize: vertical !important;
        display: block !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Session-state initialisation
# ------------------------------------------------------------------ #
_DEFAULTS = {
    "workflow_step": "idle",
    "current_query": None,
    "config_profile": "medium",
    "graph_state": None,  # dict — mirrors GraphState
    "search_queries_edit": None,  # list — mutable copy for the review UI
    "trace_steps": [],
    "query_gen": 0,  # bumped on every Regenerate to force fresh widget keys
    "gatekeeper_error": None,  # str — last rejection reason shown on idle
    "gatekeeper_pending_query": None,
    "gatekeeper_confirmed": False,
    "gatekeeper_override_allowed": False,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Clear stale ChromaDB data once per session, so the first run starts clean.
if "_kb_cleared" not in st.session_state:
    clear_collection()
    st.session_state["_kb_cleared"] = True


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _score_badge(score: int) -> str:
    """Return an HTML badge coloured by score tier."""
    if score >= 70:
        cls = "score-high"
    elif score >= 40:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score}/100</span>'


def _reset_workflow():
    """Clear all workflow state and return to idle."""
    # Bump run_id to guarantee fresh widget keys
    run_id = st.session_state.get("run_id", 0) + 1
    
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default
        
    st.session_state["run_id"] = run_id
        
    # Clear dynamically generated Streamlit widget keys to prevent ghost state
    keys_to_delete = [
        k for k in st.session_state.keys()
        if k.startswith(("paper_keep_", "q_enable_", "q_text_"))
    ]
    for k in keys_to_delete:
        del st.session_state[k]


def _is_gatekeeper_confirmation(text: str) -> bool:
    """Return True when the user confirms a pending Gatekeeper override."""
    normalized = text.strip().lower()
    return normalized in {
        "ja",
        "yes",
        "y",
        "ok",
        "okay",
        "mach weiter",
        "weiter",
        "fortfahren",
        "trotzdem fortfahren",
        "trotzdem weitermachen",
        "weiter machen",
        "continue",
        "proceed",
    }


def _confirm_pending_gatekeeper_query() -> None:
    """Continue with the previously rejected/unclear query after user confirmation."""
    if not st.session_state.gatekeeper_pending_query:
        return

    st.session_state.gatekeeper_error = None
    st.session_state.current_query = st.session_state.gatekeeper_pending_query
    st.session_state.gatekeeper_pending_query = None
    st.session_state.gatekeeper_confirmed = True
    st.session_state.gatekeeper_override_allowed = True
    st.session_state.graph_state = None
    st.session_state.search_queries_edit = None
    st.session_state.trace_steps = []
    st.session_state.query_gen = 0
    st.session_state.workflow_step = "enhancing"


def _gatekeeper_rejection(state: GraphState | None) -> str | None:
    """Return the rejection reason if the Gatekeeper explicitly rejected."""
    if state and state.get("is_valid") is False:
        return state.get("validation_reason") or "The query was rejected by the Gatekeeper."
    return None


def _reset_after_gatekeeper_rejection(state: GraphState, trace: list[str]) -> None:
    """Store rejection details and reset the workflow so the user can retry."""
    needs_confirmation = state.get("gatekeeper_needs_confirmation", False)
    was_confirmed = state.get("gatekeeper_confirmed", False)
    can_override = state.get("gatekeeper_can_override", False)
    follow_up = state.get("gatekeeper_follow_up_question")

    st.session_state.graph_state = state
    st.session_state.search_queries_edit = None
    st.session_state.trace_steps = trace
    st.session_state.query_gen = 0
    rejection = _gatekeeper_rejection(state)
    st.session_state.gatekeeper_error = (
        f"{rejection}\n\n{follow_up}"
        if follow_up and rejection and follow_up != rejection
        else follow_up or rejection
    )
    st.session_state.gatekeeper_pending_query = (
        state.get("user_query")
        if needs_confirmation and can_override and not was_confirmed
        else None
    )
    st.session_state.gatekeeper_confirmed = False
    st.session_state.gatekeeper_override_allowed = False
    st.session_state.workflow_step = "idle"


def _rerun_if_gatekeeper_rejected() -> None:
    """Prevent stale invalid graph state from reaching later workflow steps."""
    state = st.session_state.graph_state
    if _gatekeeper_rejection(state):
        _reset_after_gatekeeper_rejection(state, st.session_state.trace_steps)
        st.rerun()


def _inject_scroll_hack():
    """Injects JS to add smart auto-scrolling to trace containers."""
    st.html("""
    <script>
    const parentDoc = window.document;
    
    function attachScroll() {
        const markers = parentDoc.querySelectorAll('.trace-marker');
        markers.forEach(marker => {
            const c = marker.closest('div[data-testid="stExpanderDetails"]');
            if (!c || c.dataset.scrollInit) return;
            c.dataset.scrollInit = 'true';
            c.dataset.isUserScrolling = 'false';
            
            c.addEventListener('scroll', () => {
                const atBottom = c.scrollTop + c.clientHeight >= c.scrollHeight - 10;
                c.dataset.isUserScrolling = atBottom ? 'false' : 'true';
            });
            
            // Local observer: ONLY fires when THIS specific trace container gets new text
            const innerObserver = new MutationObserver(() => {
                if (c.dataset.isUserScrolling !== 'true') {
                    c.scrollTop = c.scrollHeight;
                }
            });
            innerObserver.observe(c, { childList: true, subtree: true, characterData: true });
        });
    }

    // Global observer: ONLY checks for NEW trace containers, does not force scroll
    const globalObserver = new MutationObserver(() => {
        attachScroll();
    });
    globalObserver.observe(parentDoc.body, { childList: true, subtree: true });

    // Handle expander open event: reset scroll to bottom
    parentDoc.addEventListener('toggle', (e) => {
        if (e.target && e.target.open) {
            setTimeout(() => {
                const markers = e.target.querySelectorAll('.trace-marker');
                markers.forEach(marker => {
                    const c = marker.closest('div[data-testid="stExpanderDetails"]');
                    if (c && c.dataset.scrollInit === 'true') {
                        c.dataset.isUserScrolling = 'false';
                        c.scrollTop = c.scrollHeight;
                    }
                });
            }, 50);
        }
    }, true);
    </script>
    """)


def append_to_evaluation_logs(final_state: dict, filepath: str = "data/experiment_baseline_logs.json"):
    """
    Extracts readable text data and writes it into a json file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 1. Cleaned active papers (mit Fallback, falls Keys fehlen)
    cleaned_papers = []
    for paper in final_state.get("active_papers", []):
        cleaned_papers.append({
            "title": str(paper.get("title", "")),
            "abstract": str(paper.get("abstract", "")),
            # Falls ingested_content leer/None ist, wird auf abstract zurückgegriffen
            "ingested_content": str(paper.get("ingested_content") or paper.get("abstract") or "")
        })

    # 2. Extract analysis data (KRITISCH für die Synthesizer-Evaluation!)
    # Das ist exakt der Kontext, den der Synthesizer verarbeitet hat
    analysis_data = final_state.get("paper_analysis_data", [])

    serializable_state = {
        "user_query": str(final_state.get("user_query", "")),
        "final_review": str(final_state.get("final_review", "")),
        "active_papers": cleaned_papers,
        "paper_analysis_data": analysis_data  # <--- HIER ERGÄNZT
    }

    existing_logs = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_logs = []

    existing_logs.append(serializable_state)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, indent=4, ensure_ascii=False)
    print(f"Erfolgreich bereinigte Daten in {filepath} geloggt!")


def run_synthesis_from_log(log_entry: dict):
    """
    Replay-Modus: Überspringt Retrieval, Ingestor & Analyst komplett.
    Startet direkt und ausschließlich mit der Synthesizer-Phase.
    """
    mocked_state = {
        "user_query": log_entry.get("user_query", ""),
        "paper_analysis_data": log_entry.get("paper_analysis_data", []),
        "active_papers": log_entry.get("active_papers", []),
        "config_profile": st.session_state.get("config_profile", "medium")
    }

    st.info(f"⚡ Replay-Modus gestartet für Query: '{mocked_state['user_query']}'")

    final_output_state = mocked_state.copy()

    with st.status("🧠 Multi-Agent Synthesizer erzeugt Literatur-Review...", expanded=True) as status:
        st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)

        # JETZT: Ruft direkt und isoliert den Synthesizer auf!
        for event in stream_synthesis_flow(mocked_state):
            if isinstance(event, str):
                st.write(event)
            elif isinstance(event, dict):
                final_output_state = event

        status.update(label="Synthese abgeschlossen!", state="complete", expanded=False)

    return final_output_state

# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #
st.sidebar.title("Settings")

try:
    model_name = ModelFactory.get_model_name()
    if "llm_is_ready" not in st.session_state or "llm_status_msg" not in st.session_state:
        is_ready, status_msg = ModelFactory.check_availability(model_name)
        st.session_state.llm_is_ready = is_ready
        st.session_state.llm_status_msg = status_msg
        st.session_state.llm_offline = not is_ready

    is_ready = st.session_state.llm_is_ready
    status_msg = st.session_state.llm_status_msg

    if is_ready:
        st.sidebar.success(f"🟢 **{status_msg}**")
    else:
        st.sidebar.error(f"🔴 **{status_msg}**")

    if st.sidebar.button("🔄 Refresh Status", key="refresh_llm_status", use_container_width=True):
        is_ready, status_msg = ModelFactory.check_availability(model_name)
        st.session_state.llm_is_ready = is_ready
        st.session_state.llm_status_msg = status_msg
        st.session_state.llm_offline = not is_ready
        st.rerun()

except Exception as e:
    st.sidebar.error(f"Error loading model name: {e}")
    st.session_state.llm_offline = True

st.sidebar.markdown("---")
st.sidebar.subheader("Research Profile")

profile_labels = {
    "fast": "⚡ Fast — quick scan",
    "medium": "⚖️ Medium — balanced",
    "pro": "🔬 Pro — exhaustive",
}
selected_profile = st.sidebar.radio(
    "Choose depth",
    options=list(profile_labels.keys()),
    format_func=lambda k: profile_labels[k],
    index=list(profile_labels.keys()).index(st.session_state.config_profile),
    key="profile_radio",
    disabled=st.session_state.workflow_step != "idle",
)
# Sync back (only when idle to avoid mid-run changes)
if st.session_state.workflow_step == "idle":
    st.session_state.config_profile = selected_profile

cfg = get_config(st.session_state.config_profile)
st.sidebar.markdown(
    f"<div class='profile-chip'>"
    f"Queries: {cfg['max_queries']} &nbsp;·&nbsp; "
    f"Top papers: {cfg['top_n_papers']}</div>",
    unsafe_allow_html=True,
)

# Reset button (always available when not idle)
if st.session_state.workflow_step != "idle":
    st.sidebar.markdown("---")
    if st.sidebar.button("Start Over", use_container_width=True):
        clear_collection()
        _reset_workflow()
        st.rerun()

# ------------------------------------------------------------------ #
#  Sidebar: Replay & Evaluation Panel
# ------------------------------------------------------------------ #
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Evaluation Replay Mode")

# Saubere Datenbasis mit den 5 Test-Fragen
testset_file = "data/evaluation_testset.json"

if os.path.exists(testset_file):
    try:
        with open(testset_file, "r", encoding="utf-8") as f:
            test_entries = json.load(f)

        if test_entries:
            # Erstelle Dropdown-Optionen: Index + Query Name
            options = [f"Frage {i + 1}: {entry.get('user_query', '')[:35]}..." for i, entry in enumerate(test_entries)]
            selected_idx = st.sidebar.selectbox(
                "Testfall auswählen:",
                range(len(options)),
                format_func=lambda x: options[x],
                key="replay_select_box"
            )

            if st.sidebar.button("🚀 Re-Run Synthesizer", use_container_width=True, type="primary"):
                # Hole exakt das gewählte JSON-Objekt
                selected_test = test_entries[selected_idx]

                # Starte die Replay-Synthese
                new_state = run_synthesis_from_log(selected_test)

                # Speichere das neue Multi-Agent Ergebnis automatisch in einer eigenen Log-Datei
                #append_to_evaluation_logs(new_state, filepath="data/experiment_multiagent_logs.json")

                # Setze State auf 'done', um das Ergebnis direkt im Hauptfenster anzuzeigen
                st.session_state.graph_state = new_state
                st.session_state.workflow_step = "done"
                st.rerun()
        else:
            st.sidebar.warning("`evaluation_testset.json` ist leer.")
    except Exception as e:
        st.sidebar.error(f"Fehler beim Laden des Testsets: {e}")
else:
    st.sidebar.caption("💡 Lege `data/evaluation_testset.json` an, um den Replay-Modus zu nutzen.")

# ------------------------------------------------------------------ #
#  Header
# ------------------------------------------------------------------ #
st.markdown('<h1 class="main-title">SmartScholar</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #5f6368;">'
    "Agentic Research Copilot</p>",
    unsafe_allow_html=True,
)

# Step progress indicator
_STEP_LABELS = {
    "idle": "Ready",
    "enhancing": "Generating queries…",
    "query_review": "Review Queries",
    "searching": "Searching papers…",
    "paper_review": "Review Papers",
    "ingesting": "Analysing papers…",
    "done": "Research Finalised ✓",
}
current_label = _STEP_LABELS.get(st.session_state.workflow_step, "")
if st.session_state.workflow_step != "idle":
    st.markdown(
        f'<div class="step-indicator">Step: '
        f'<span class="current-step">{current_label}</span></div>',
        unsafe_allow_html=True,
    )

# Inject the auto-scroll JS globally
_inject_scroll_hack()

# ================================================================== #
#  STEP 0 — Idle: accept user input
# ================================================================== #
if st.session_state.workflow_step == "idle":
    if st.session_state.gatekeeper_error:
        if st.session_state.gatekeeper_pending_query:
            st.warning(st.session_state.gatekeeper_error)
            if st.button("Als Research-Task bestätigen", type="primary"):
                _confirm_pending_gatekeeper_query()
                st.rerun()
        else:
            st.error(st.session_state.gatekeeper_error)

    query = st.chat_input("What would you like to research today?")
    if query:
        if st.session_state.get("llm_offline", False):
            st.error(
                "\u2715 [System] Cannot start research: The LLM server is unreachable "
                "or the configured model is missing. Please check your settings "
                "in the sidebar."
            )
            st.stop()
            
        if (
                st.session_state.gatekeeper_pending_query
                and _is_gatekeeper_confirmation(query)
        ):
            _confirm_pending_gatekeeper_query()
            st.rerun()

        st.session_state.gatekeeper_error = None
        st.session_state.gatekeeper_pending_query = None
        st.session_state.gatekeeper_confirmed = False
        st.session_state.gatekeeper_override_allowed = False
        st.session_state.graph_state = None
        st.session_state.search_queries_edit = None
        st.session_state.trace_steps = []
        st.session_state.query_gen = 0
        st.session_state.current_query = query
        st.session_state.workflow_step = "enhancing"
        st.rerun()


# ================================================================== #
#  STEP 1 — Enhancing: run the LangGraph (gatekeeper → enhance)
# ================================================================== #
elif st.session_state.workflow_step == "enhancing":
    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    with st.status("🧠 Expanding your query into academic search terms…", expanded=True) as status:
        st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)
        trace = list(st.session_state.trace_steps)
        result: GraphState = {}

        # Pre-fill historical trace logs
        for step in trace:
            st.write(step)

        def _log(msg):
            st.write(msg)
            trace.append(msg)

        if st.session_state.query_gen == 0:
            initial_state: GraphState = {
                "user_query": st.session_state.current_query,
                "config_profile": st.session_state.config_profile,
                "gatekeeper_confirmed": st.session_state.gatekeeper_confirmed,
                "gatekeeper_override_allowed": st.session_state.gatekeeper_override_allowed,
            }
            stream = stream_enhance_flow(initial_state, regenerate=False)
        else:
            initial_state = st.session_state.graph_state
            stream = stream_enhance_flow(initial_state, regenerate=True)

        for event in stream:
            if isinstance(event, str):
                _log(event)
            elif isinstance(event, dict):
                # Final graph state
                result = event

        rejection = _gatekeeper_rejection(result)
        if rejection:
            _reset_after_gatekeeper_rejection(result, trace)
            status.update(label="Query rejected by Gatekeeper", state="error", expanded=False)
            st.rerun()

        queries = result.get("search_queries", [])
        for i, q in enumerate(queries, 1):
            _log(f"   {i}. `{q}`")

        st.session_state.graph_state = result
        st.session_state.search_queries_edit = list(queries)
        st.session_state.trace_steps = trace
        st.session_state.gatekeeper_confirmed = False
        st.session_state.gatekeeper_override_allowed = False
        status.update(label="Queries Generated", state="complete", expanded=False)

    st.session_state.workflow_step = "query_review"
    st.rerun()


# ================================================================== #
#  STEP 2 — Query Review: human edits / accepts queries
# ================================================================== #
elif st.session_state.workflow_step == "query_review":
    _rerun_if_gatekeeper_rejected()

    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    # Replay trace
    with st.status("Queries Generated", state="complete", expanded=False):
        st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)
        for step in st.session_state.trace_steps:
            st.write(step)

    gs = st.session_state.graph_state
    strategy = gs.get("query_strategy", "")
    if strategy:
        st.info(f"🧠 **Search Strategy:** {strategy}")

    st.markdown("### Review Search Queries")
    st.markdown("**Foundational Query (Always Included):**")
    st.info(f"**`{st.session_state.current_query}`**")

    st.markdown("#### Optional Suggested Expansions")
    st.caption(
        "Edit the optional queries below, uncheck any you want to skip, "
        "then accept or regenerate."
    )

    # Filter out any query that exactly matches the foundational query to avoid redundancy
    queries = [q for q in st.session_state.search_queries_edit if
               q.strip().lower() != st.session_state.current_query.strip().lower()]
    updated_queries: list[str] = []
    enabled_flags: list[bool] = []

    if not queries:
        st.info(
            "No additional enhanced queries were generated. The search will proceed with only your foundational query.")
    else:
        for i, q in enumerate(queries):
            col_check, col_input = st.columns([0.08, 0.92])
            with col_check:
                gen = st.session_state.query_gen
                enabled = st.checkbox(
                    "Use", value=True, key=f"q_enable_{gen}_{i}", label_visibility="collapsed"
                )
            with col_input:
                gen = st.session_state.query_gen
                edited = st.text_input(
                    f"Query {i + 1}",
                    value=q,
                    key=f"q_text_{gen}_{i}",
                    label_visibility="collapsed",
                )
            enabled_flags.append(enabled)
            updated_queries.append(edited)

    col_regen, col_accept = st.columns(2)

    with col_regen:
        if st.button("Regenerate", use_container_width=True):
            # Bump generation counter so the next render creates fresh widget keys
            st.session_state.query_gen += 1
            st.session_state.workflow_step = "enhancing"
            st.rerun()

    with col_accept:
        if st.button("Accept & Search", type="primary", use_container_width=True):
            # Keep only checked queries
            accepted = [
                q for q, ok in zip(updated_queries, enabled_flags) if ok and q.strip()
            ]
            gs = st.session_state.graph_state
            gs["search_queries"] = [st.session_state.current_query] + accepted
            st.session_state.graph_state = gs
            st.session_state.workflow_step = "searching"
            st.rerun()


# ================================================================== #
#  STEP 3 — Searching: fetch + score papers
# ================================================================== #
elif st.session_state.workflow_step == "searching":
    _rerun_if_gatekeeper_rejected()

    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    with st.status("Searching Semantic Scholar & scoring papers…", expanded=True) as status:
        st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)
        trace = list(st.session_state.trace_steps)

        # Pre-fill historical trace logs
        for step in trace:
            st.write(step)

        def _log_s(msg):
            st.write(msg)
            trace.append(msg)

        gs = st.session_state.graph_state

        for event in stream_search_flow(gs):
            if isinstance(event, str):
                _log_s(event)
            elif isinstance(event, dict):
                gs = event

        st.session_state.graph_state = gs
        st.session_state.trace_steps = trace

        status.update(label="Papers Ranked", state="complete", expanded=False)

    st.session_state.workflow_step = "paper_review"
    st.rerun()


# ================================================================== #
#  STEP 4 — Paper Review: curate the active set
# ================================================================== #
elif st.session_state.workflow_step == "paper_review":
    _rerun_if_gatekeeper_rejected()

    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    # Show search trace
    with st.status("Papers Ranked", state="complete", expanded=False):
        st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)
        for step in st.session_state.trace_steps:
            st.write(step)

    gs = st.session_state.graph_state
    active = gs.get("active_papers", [])
    discarded = gs.get("discarded_papers", [])
    # ---- Guard: zero search results ---- #
    if not active and not discarded:
        st.error(
            "\u2715 [System] No papers found. Possible causes: "
            "API rate limit (Semantic Scholar), network errors, "
            "or overly specific search queries."
        )
        error_detail = gs.get("error") or gs.get("search_error")
        if error_detail:
            st.code(error_detail, language="text")

        col_retry, col_back = st.columns(2)
        with col_retry:
            if st.button("Retry Search", use_container_width=True):
                st.session_state.workflow_step = "searching"
                st.rerun()
        with col_back:
            if st.button("Back to Query Review", use_container_width=True):
                st.session_state.workflow_step = "query_review"
                st.rerun()
        st.stop()

    st.markdown("### Review Active Papers")
    st.caption(
        f"Top {len(active)} papers ranked by relevance. "
        "Uncheck to remove, or add from the reserve pool."
    )

    keep_flags: list[bool] = []

    for i, paper in enumerate(active):
        score = paper.get("relevance_score", 0)
        title = paper.get("title", "Untitled")
        year = paper.get("year", "n.d.")
        cites = paper.get("citationCount", 0)
        abstract = paper.get("abstract") or "No abstract available."
        authors = paper.get("authors", [])
        url = paper.get("url", "")
        badge = _score_badge(score)

        with st.expander(f"**{title}** ({year}) — Score: {score}/100", expanded=False):
            st.markdown(
                f"**Authors:** {', '.join(authors[:5])}"
                f"{'…' if len(authors) > 5 else ''}"
            )
            st.markdown(f"**Citations:** {cites} &nbsp;·&nbsp; **Year:** {year} &nbsp;·&nbsp; {badge}",
                        unsafe_allow_html=True)

            rationale = paper.get("score_rationale", "")
            if rationale:
                st.markdown(f"🧠 **Rationale:** {rationale}")

            if url:
                st.markdown(f"[View on Semantic Scholar]({url})")
            st.markdown("---")
            st.markdown(f"**Abstract:** {abstract}")

        run_id = st.session_state.get("run_id", 0)
        keep = st.checkbox(
            f"Include _{title[:60]}_",
            value=True,
            key=f"paper_keep_{run_id}_{i}",
        )
        keep_flags.append(keep)

    st.markdown("---")

    # ---- Add alternative from discarded pool ---- #
    col_add, col_fin = st.columns(2)

    with col_add:
        add_disabled = len(discarded) == 0
        if st.button(
                "Add Alternative Paper",
                use_container_width=True,
                disabled=add_disabled,
                help="Pull the next-highest-scored paper from the reserve pool",
        ):
            if discarded:
                promoted = discarded.pop(0)
                active.append(promoted)
                gs["active_papers"] = active
                gs["discarded_papers"] = discarded
                st.session_state.graph_state = gs
                st.rerun()

    with col_fin:
        selected_count = sum(1 for keep in keep_flags if keep)
        if st.button(
                "Finalize Research",
                type="primary",
                use_container_width=True,
                disabled=(selected_count == 0),
        ):
            if selected_count == 0:
                st.warning(
                    "\u26a0 [System] You must select at least one paper "
                    "to start the analysis!"
                )
            else:
                # Remove unchecked papers
                final_active = []
                newly_discarded = list(discarded)  # copy
                for paper, keep in zip(active, keep_flags):
                    if keep:
                        final_active.append(paper)
                    else:
                        newly_discarded.insert(0, paper)

                # Assign citation IDs
                for idx, paper in enumerate(final_active, start=1):
                    paper["citation_id"] = f"[{idx}]"

                gs["active_papers"] = final_active
                gs["discarded_papers"] = newly_discarded
                st.session_state.graph_state = gs

                # Hand off to a dedicated 'ingesting' step. This makes the page
                # re-render from the top and run the pipeline in the main trace
                # container, consistent with the other agents.
                st.session_state.workflow_step = "ingesting"
                st.rerun()

    # Show reserve pool size
    if discarded:
        st.caption(f"Reserve pool: {len(discarded)} papers available")

# ================================================================== #
#  STEP 4.5 — Ingesting: ingest papers + run analyst pipeline
# ================================================================== #
elif st.session_state.workflow_step == "ingesting":
    _rerun_if_gatekeeper_rejected()

    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    with st.status(
            "🧠 Agent Pipeline (Ingestion, Analysis & Synthesis)…", expanded=True
    ) as status:
        st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)
        trace = list(st.session_state.trace_steps)

        # Pre-fill historical trace logs
        for step in trace:
            st.write(step)

        def _log_i(msg):
            st.write(msg)
            trace.append(msg)

        gs = st.session_state.graph_state

        for event in stream_analysis_flow(gs):
            if isinstance(event, str):
                _log_i(event)
            elif isinstance(event, dict):
                gs = event

        st.session_state.graph_state = gs
        st.session_state.trace_steps = trace

        status.update(
            label="Pipeline Completed!",
            state="complete",
            expanded=False,
        )

    st.session_state.workflow_step = "done"
    st.rerun()

# ================================================================== #
#  STEP 5 — Done: show final selection with citation IDs
# ================================================================== #
elif st.session_state.workflow_step == "done":
    _rerun_if_gatekeeper_rejected()

    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    # Persist the trace logs so the user can review the entire thought process
    if st.session_state.trace_steps:
        with st.expander("Execution Trace", expanded=False):
            st.markdown('<span class="trace-marker"></span>', unsafe_allow_html=True)
            for step in st.session_state.trace_steps:
                st.write(step)

    gs = st.session_state.graph_state
    active = gs.get("active_papers", [])
    review = gs.get("final_review", "")

    # Log Graphstate
    append_to_evaluation_logs(gs)

    st.markdown("### Research Finalised")
    st.success(
        f"**{len(active)} papers** selected and assigned citation IDs. "
    )

    # Citation table
    st.markdown("#### Final Paper Selection")
    for paper in active:
        cid = paper.get("citation_id", "")
        title = paper.get("title", "Untitled")
        year = paper.get("year", "n.d.")
        score = paper.get("relevance_score", 0)
        url = paper.get("url", "")
        abstract = paper.get("abstract") or "No abstract available."
        authors = paper.get("authors", [])
        cites = paper.get("citationCount", 0)
        badge = _score_badge(score)

        with st.expander(f"{cid} **{title}** ({year}) — Score: {score}/100", expanded=False):
            st.markdown(
                f"**Authors:** {', '.join(authors[:5])}"
                f"{'…' if len(authors) > 5 else ''}"
            )
            st.markdown(f"**Citations:** {cites} &nbsp;·&nbsp; {badge}", unsafe_allow_html=True)

            rationale = paper.get("score_rationale", "")
            if rationale:
                st.markdown(f"🧠 **Rationale:** {rationale}")

            if url:
                st.markdown(f"[View on Semantic Scholar]({url})")
            st.markdown("---")
            st.markdown(f"{abstract}")

    # Final Review Output
    if review:
        st.markdown("---")
        st.markdown("### Final Literature Review")
        st.markdown(review)

    # Download final review as Markdown
    st.markdown("---")
    today = datetime.now().strftime("%Y-%m-%d")
    review_text = gs.get("final_review", "") or "No review generated."
    st.download_button(
        label="Download Literature Review (.md)",
        data=review_text,
        file_name=f"smartscholar_review_{today}.md",
        mime="text/markdown",
    )
