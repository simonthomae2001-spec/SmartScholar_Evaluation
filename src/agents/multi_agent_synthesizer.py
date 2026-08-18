import logging
from typing import Callable, List, Dict
from src.core.model_factory import ModelFactory

logger = logging.getLogger(__name__)

# --- PROMPTS ---

# 1. Outline Prompt: Plant die Struktur basierend auf den Daten
OUTLINE_PROMPT = """You are an expert lead researcher. 
Based on the provided research papers, create a detailed outline for a literature review.

RESEARCH TOPIC: "{user_query}"
PROVIDED PAPERS:
{paper_data}

Output ONLY a JSON array of section objects, each with a 'title' and a 'description' of what specific paper findings should be synthesized in that section.
Format:
[
  {{"title": "Introduction & Overview", "description": "Cover topic X and introduction to sources [1], [2]."}},
  {{"title": "Methodological Approaches", "description": "Compare methodologies from [1] and [3]."}},
  {{"title": "Critical Limitations & Gaps", "description": "Analyze limitations found in [2] and [3]."}},
  {{"title": "Conclusion", "description": "Synthesize main conclusions and future directions."}}
]
"""

# 2. Section Generator Prompt: Generiert EINEN einzelnen Abschnitt
SECTION_PROMPT = """You are an academic writer. Write the literature review section for: "{section_title}".

SECTION INSTRUCTIONS: {section_description}

FULL SOURCE DATA (Use ONLY these facts and preserve citation IDs like [1], [2]):
{paper_data}

CONSTRAINTS:
1. Write 2-5 well-developed paragraphs specifically for this section.
2. Ground EVERY claim in the provided source data using inline citations [X].
3. Write in the same language as the topic.
"""

# 3. Stitching / Coherence Prompt: Verbindet die Bausteine
STITCHING_PROMPT = """You are a senior academic editor.
You have been provided with individual draft sections of a literature review.

DRAFT SECTIONS:
{draft_sections}

YOUR TASK:
Review and smooth out the transitions between sections to create a single, cohesive, publication-ready literature review in Markdown.

CRITICAL RULES:
1. DO NOT change, remove, or hallucinate any facts or inline citations (e.g. [1], [2]).
2. DO NOT delete any existing sections or content. Only improve flow, tone, and transitions.
3. Ensure proper Markdown formatting (# Literature Review, ## Section Titles).
"""


class MultiAgentSynthesizerAgent:
    def __init__(self):
        self.llm = ModelFactory.get_model()
        logger.info("MultiAgentSynthesizerAgent initialisiert.")

    def synthesize_review(self, state: dict, config: dict,
                          status_callback: Callable[[str], None] | None = None) -> dict:
        def _log(msg: str):
            if status_callback:
                status_callback(msg)
            logger.info(msg)

        user_query = state.get("user_query", "")
        analysis_data = state.get("paper_analysis_data", [])

        if not analysis_data:
            _log("⚠ [Multi-Synthesizer] No analysis data found.")
            return {"final_review": "Error: No analysed papers were found."}

        formatted_data = self._format_paper_data(analysis_data)

        try:
            # --- PHASE 1: AGENT 1A - Section Planning ---
            _log("🧠 [Multi-Synthesizer] Agent 1: Planning structure & sections...")
            outline = self._plan_sections(user_query, formatted_data, config)

            # --- PHASE 2: AGENT 1B - Section Generation (Chunking Loop) ---
            generated_sections = []
            for idx, sec in enumerate(outline, 1):
                _log(f"✍ [Multi-Synthesizer] Agent 1: Writing section {idx}/{len(outline)}: '{sec.get('title')}'...")
                section_text = self._generate_section(sec, formatted_data, config)
                generated_sections.append(f"## {sec.get('title')}\n\n{section_text}")

            full_draft = "\n\n".join(generated_sections)

            # --- PHASE 3: AGENT 2 - Coherence & Stitching ---
            _log("🔍 [Multi-Synthesizer] Agent 2: Editing and stitching sections for cohesion...")
            final_review = self._stitch_sections(full_draft, config)

            _log("✓ [Multi-Synthesizer] Multi-Agent Synthesis complete!")
            return {"final_review": final_review}

        except Exception as e:
            _log(f"✕ [Multi-Synthesizer] Error: {str(e)}")
            return {"final_review": f"Error during multi-agent generation: {str(e)}"}

    def _plan_sections(self, query: str, paper_data: str, config: dict) -> List[Dict[str, str]]:
        # Ausführen des Planners (gibt z. B. Liste von Abschnitten zurück)
        prompt = OUTLINE_PROMPT.format(user_query=query, paper_data=paper_data)
        resp = self.llm.complete(prompt, temperature=0.2)

        # Fallback-Struktur, falls das LLM kein valides JSON zurückgibt
        import json
        try:
            return json.loads(str(resp))
        except Exception:
            return [
                {"title": "Introduction", "description": "Introduce the query and general background."},
                {"title": "Thematic Analysis", "description": "Synthesize main findings across papers."},
                {"title": "Methodology & Limitations", "description": "Compare methods and discuss limitations."},
                {"title": "Conclusion", "description": "Summarize key findings."}
            ]

    def _generate_section(self, section_info: dict, paper_data: str, config: dict) -> str:
        prompt = SECTION_PROMPT.format(
            section_title=section_info.get("title", ""),
            section_description=section_info.get("description", ""),
            paper_data=paper_data
        )
        temp = config.get("llm", {}).get("temperature_creative", 0.4)
        num_predict = config.get("llm", {}).get("max_tokens_short", 1000)

        resp = self.llm.complete(prompt, temperature=temp, num_predict=num_predict)
        return str(resp)

    def _stitch_sections(self, full_draft: str, config: dict) -> str:
        prompt = STITCHING_PROMPT.format(draft_sections=full_draft)
        temp = config.get("llm", {}).get("temperature_analytical", 0.1)

        resp = self.llm.complete(prompt, temperature=temp)
        return str(resp)

    def _format_paper_data(self, analysis_data: list[dict]) -> str:
        formatted_string = ""
        for paper in analysis_data:
            citation_id = paper.get("citation_id", "[?]")
            methodology = paper.get("methodology", "N/A")
            findings = paper.get("findings", "N/A")
            limitations = paper.get("limitations", "N/A")
            relevance = paper.get("user_relevance", "N/A")

            formatted_string += f"Citation ID: {citation_id}\n"
            formatted_string += f"User Relevance Context: {relevance}\n"
            formatted_string += f"Methodology: {methodology}\n"
            formatted_string += f"Findings: {findings}\n"
            formatted_string += f"Limitations: {limitations}\n"
            formatted_string += "-" * 40 + "\n\n"
        return formatted_string