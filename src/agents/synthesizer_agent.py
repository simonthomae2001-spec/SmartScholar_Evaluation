from src.core.model_factory import ModelFactory
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Der System-Prompt als Konstante (auf Englisch für bessere LLM-Performance)
SYNTHESIS_PROMPT = """You are an expert academic researcher and highly skilled technical writer. 
Your task is to synthesize a comprehensive, critical literature review based EXCLUSIVELY on the provided extracted paper data.

RESEARCH TOPIC (USER QUERY):
"{user_query}"

PROVIDED PAPER DATA:
{paper_data}

INSTRUCTIONS & CONSTRAINTS:
1. THEMATIC SYNTHESIS: Do NOT just list paper by paper (e.g., do not write "Paper 1 says... Paper 2 says..."). You must group the findings by themes, concepts, or methodologies.
2. STRICT GROUNDING: Make NO assumptions and include NO external knowledge. Every claim you make must be directly supported by the "PROVIDED PAPER DATA". 
3. INLINE CITATIONS: Every claim, finding, or methodology mentioned must be cited immediately using the provided citation IDs (e.g., "Recent studies show X [1][3].").
4. LANGUAGE: Write the review in the language of the RESEARCH TOPIC. If the topic is German, write in German. If English, write in English.

REQUIRED STRUCTURE (Use Markdown):
# Literature Review

## Introduction
Provide a brief overview of the research topic and what this review covers based on the provided papers.

## Thematic Analysis
(Create 2-3 logical sub-headings based on the findings. Synthesize how the papers agree, disagree, or complement each other).

## Methodological Critique
Critically compare the methodologies and explicitly discuss the 'limitations' mentioned in the provided data.

## Conclusion & Future Research
Summarize the current state of research. Explicitly identify research gaps based on the limitations and findings.

## Analyzed Sources
List the citation IDs and a one-sentence summary of their contribution based on the provided data.
"""

class SynthesizerAgent:
    def __init__(self):
        # Initialisiert das LLM über die zentrale Factory, genau wie der ResearcherAgent
        self.llm = ModelFactory.get_model()
        logger.info("SynthesizerAgent initialisiert.")

    def synthesize_review(self, state: dict, config: dict, status_callback: Callable[[str], None] | None = None) -> dict:
        """
        Generiert die finale Literature Review aus den strukturierten Analyse-Daten.
        """
        def _log(msg: str):
            if status_callback:
                status_callback(msg)
            logger.info(msg)

        # 1. Daten aus dem GraphState extrahieren
        user_query = state.get("user_query", "")
        analysis_data = state.get("paper_analysis_data", [])

        # Fallback, falls keine Daten ankommen
        if not analysis_data:
            _log("⚠️ [Synthesizer] Keine Analyse-Daten im State gefunden. Breche Synthese ab.")
            return {"final_review": "Fehler: Es wurden keine analysierten Paper für die Synthese gefunden."}

        # 2. Analyse-Daten für den Prompt formatieren
        formatted_data = self._format_paper_data(analysis_data)

        # 3. Prompt zusammenbauen
        prompt = SYNTHESIS_PROMPT.format(
            user_query=user_query,
            paper_data=formatted_data
        )

        _log(f"📝 [Synthesizer] Starte LLM-Synthese für {len(analysis_data)} Paper...")

        try:
            # 4. LLM aufrufen (LlamaIndex-Standard für Ollama-Aufrufe)
            response = self.llm.complete(prompt)
            
            # Die Antwort als String extrahieren (.text ist bei LlamaIndex oft nötig)
            final_review_text = str(response)
            
            _log("✅ [Synthesizer] Synthese erfolgreich abgeschlossen.")
            # 5. Partial Update für den GraphState zurückgeben
            return {"final_review": final_review_text}
            
        except Exception as e:
            _log(f"❌ [Synthesizer] Fehler bei der Generierung der Review: {str(e)}")
            return {"final_review": f"Fehler bei der Generierung der Review: {str(e)}"}

    def _format_paper_data(self, analysis_data: list[dict]) -> str:
        """
        Hilfsmethode, um die Liste der Dictionaries in einen sauberen Text-Block zu verwandeln.
        """
        formatted_string = ""
        for paper in analysis_data:
            # Nutzt die exakten Keys aus eurer Analyst-Stub-Spezifikation
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
