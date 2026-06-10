from src.core.model_factory import ModelFactory
import logging

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

    def synthesize_review(self, state: dict, config: dict) -> dict:
        """
        Generiert die finale Literature Review aus den strukturierten Analyse-Daten.
        """
        # 1. Daten aus dem GraphState extrahieren
        user_query = state.get("user_query", "")
        analysis_data = state.get("paper_analysis_data", [])

        # Fallback, falls keine Daten ankommen
        if not analysis_data:
            logger.warning("Keine Analyse-Daten im State gefunden. Breche Synthese ab.")
            return {"final_review": "Fehler: Es wurden keine analysierten Paper für die Synthese gefunden."}

        # 2. Analyse-Daten für den Prompt formatieren
        formatted_data = self._format_paper_data(analysis_data)

        # 3. Prompt zusammenbauen
        prompt = SYNTHESIS_PROMPT.format(
            user_query=user_query,
            paper_data=formatted_data
        )

        logger.info(f"Starte LLM-Synthese für {len(analysis_data)} Paper...")

        try:
            # 4. LLM aufrufen (LlamaIndex-Standard für Ollama-Aufrufe)
            response = self.llm.complete(prompt)
            
            # Die Antwort als String extrahieren (.text ist bei LlamaIndex oft nötig)
            final_review_text = str(response)
            
            # 5. Partial Update für den GraphState zurückgeben
            return {"final_review": final_review_text}
            
        except Exception as e:
            logger.error(f"Fehler bei der LLM-Generierung im Synthesizer: {e}")
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

if __name__ == "__main__":
    # --- PROVISORISCHER LOKALER TEST ---

    # 1. Wir definieren unseren Mock-GraphState
    mock_graph_state = {
        "user_query": "Impact of Agentic RAG pipelines on the efficiency and accuracy of scientific literature reviews",
        "config_profile": "medium",
        "paper_analysis_data": [
            {
                "citation_id": "[1]",
                "methodology": "Comparative study evaluating Agentic RAG pipelines against traditional keyword search across 500 scientific papers using Llama-3.",
                "findings": "Agentic RAG reduced manual curation time by 40% and improved thematic relevance scores by 25%. Multi-agent loops with a 'Critic' node caught 80% more factual inconsistencies.",
                "limitations": "The study only evaluated open-source LLMs and limited vector stores (ChromaDB), ignoring proprietary models like GPT-4.",
                "user_relevance": "Highly relevant. Directly quantifies the efficiency and accuracy improvements of Agentic RAG asked in the query."
            },
            {
                "citation_id": "[2]",
                "methodology": "Systematic review of human-in-the-loop (HITL) breakpoints in multi-agent research systems.",
                "findings": "Incorporating HITL breakpoints between query expansion and paper selection increased user trust by 60% and prevented LLM topic drift in 90% of edge cases.",
                "limitations": "Small sample size of human evaluators (n=15). Did not measure raw time-to-completion.",
                "user_relevance": "Relevant context. Shows that while accuracy increases, human breakpoints might impact total automation efficiency."
            },
            {
                "citation_id": "[3]",
                "methodology": "Benchmarking study on chunking strategies (abstract vs. full-text) in scientific RAG applications.",
                "findings": "Full-text chunking increased hallucination rates by 12% compared to hybrid abstract-first approaches, due to noise in methodology sections. Abstract-only search was fastest but missed key limitations.",
                "limitations": "Did not test semantic chunking, only static character-limit chunking (512 vs 1024 chars).",
                "user_relevance": "Moderately relevant. Explains underlying mechanics of why Agentic RAG might struggle with accuracy depending on ingestion depth."
            }
        ]
    }

    # 2. Agent instanziieren (holt sich Llama3 über eure Factory)
    print("Starte lokalen Test des SynthesizerAgents...\n")
    agent = SynthesizerAgent()

    # 3. Synthese mit Mock-Daten anstoßen
    print("Sende Prompt an das LLM (das kann ein paar Sekunden dauern)...\n")
    result = agent.synthesize_review(state=mock_graph_state, config={})

    # 4. Ergebnis wunderschön in die Konsole drucken
    print("=" * 50)
    print(" GENERIERTE LITERATURE REVIEW ")
    print("=" * 50)
    print(result.get("final_review", "Fehler: Kein Output generiert."))