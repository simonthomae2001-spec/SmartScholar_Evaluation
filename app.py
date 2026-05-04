import streamlit as st
import sys
import os
from datetime import datetime

# Add src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.core.model_factory import ModelFactory
from src.agents.researcher_agent import ResearcherAgent
from src.core.vector_store import VectorEngine

# Page configuration
st.set_page_config(
    page_title="SmartScholar",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Gemini Aesthetics - Custom CSS
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
    
    /* Modern Polish for Status & Expander */

    .stStatusContainer, .stExpander {
        padding: 1.2rem !important;
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #dadce0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
    }

    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "research_review" not in st.session_state:
    st.session_state.research_review = None
if "papers" not in st.session_state:
    st.session_state.papers = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "trace_steps" not in st.session_state:
    st.session_state.trace_steps = []

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
try:
    model_name = ModelFactory.get_model_name()
    st.sidebar.success(f"**Active Model:** `{model_name}`")
except Exception as e:
    st.sidebar.error(f"Error loading model name: {e}")

# Main UI
st.markdown('<h1 class="main-title">SmartScholar</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #5f6368;">Agentic Research Copilot</p>', unsafe_allow_html=True)

# Chat-style input for research questions
query = st.chat_input("What would you like to research today?", disabled=st.session_state.is_running)

# Interaction Flow & Execution
if query:
    st.session_state.is_running = True
    st.session_state.has_run = False
    st.session_state.current_query = query
    st.session_state.research_review = None
    st.session_state.papers = []
    st.session_state.trace_steps = []
    st.rerun()

if st.session_state.current_query:
    if st.session_state.is_running:
        # Inject CSS for the Stop button
        st.markdown("""
            <style>
            .st-key-stop_btn button {
                background-color: #ea4335 !important;
                color: white !important;
                border-radius: 4px !important;
                border: none !important;
                padding: 6px 16px !important;
                font-size: 14px !important;
                margin-bottom: 10px !important;
            }
            .st-key-stop_btn button:hover {
                background-color: #d93025 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
        if st.button("⏹ Stop", key="stop_btn", help="Stop Generation"):
            st.session_state.is_running = False
            st.session_state.has_run = False
            st.session_state.current_query = None
            st.session_state.research_review = None
            st.session_state.papers = []
            st.session_state.trace_steps = []
            st.rerun()

    # Display the user's query
    with st.chat_message("user"):
        st.write(st.session_state.current_query)

    if st.session_state.is_running:
        # Agent-driven research pipeline
        with st.status("🔍 SmartScholar is investigating...", expanded=True) as status:
            trace = []

            def log(msg):
                """Write to the live status container AND record for replay."""
                st.write(msg)
                trace.append(msg)

            # --- Step 1: Query Expansion ---
            log("🧩 **Step 1 — Generating refined search queries...**")
            agent = ResearcherAgent()
            queries = agent.generate_search_queries(st.session_state.current_query)
            log(f"   ↳ Expanded into **{len(queries)}** queries.")

            # --- Step 2: Multi-query search + deduplication ---
            log("🔎 **Step 2 — Searching Semantic Scholar...**")
            papers = agent.execute_research(
                user_input=st.session_state.current_query,
                queries=queries,
                limit_per_query=5,
                status_callback=log,
            )
            st.session_state.papers = papers

            if papers:
                # --- Step 3: Index into ChromaDB ---
                log(f"📥 **Step 3 — Indexing {len(papers)} papers into ChromaDB...**")
                vector_engine = VectorEngine()
                index = vector_engine.index_papers(papers)

                # --- Step 4: Literature Synthesis ---
                log("🧠 **Step 4 — Generating Literature Synthesis...**")
                query_engine = vector_engine.get_query_engine(index)
                response = query_engine.query(
                    f"Provide a comprehensive literature review summarizing "
                    f"the key findings from these papers regarding: "
                    f"{st.session_state.current_query}"
                )

                st.session_state.research_review = str(response)
                status.update(label="Research Complete", state="complete", expanded=False)
                st.session_state.has_run = True
            else:
                log("❌ **No relevant papers found.**")
                status.update(label="Search Failed", state="error", expanded=True)
                st.session_state.has_run = True

            st.session_state.trace_steps = trace

        st.session_state.is_running = False
        st.rerun()

    elif st.session_state.has_run:
        # Replay the recorded trace steps in a static status container
        if st.session_state.research_review:
            with st.status("Research Complete", expanded=False, state="complete"):
                for step in st.session_state.trace_steps:
                    st.write(step)
        else:
            with st.status("Search Failed", expanded=True, state="error"):
                for step in st.session_state.trace_steps:
                    st.write(step)

# Persistent Result Architecture: Literature Review
if st.session_state.research_review:
    st.markdown("---")
    with st.expander("📄 Final Literature Review", expanded=True):
        st.markdown(st.session_state.research_review)

        st.markdown("---")

        # Download Functionality
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"research_review_{today}.md"

        st.download_button(
            label="📥 Download Review (.md)",
            data=st.session_state.research_review,
            file_name=filename,
            mime="text/markdown"
        )

        # Sources
        st.markdown("### 📚 Referenced Sources")
        for paper in st.session_state.papers:
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "Unknown Year")
            url = paper.get("url")
            if url:
                st.markdown(f"- **[{title}]({url})** ({year})")
            else:
                st.markdown(f"- **{title}** ({year})")
