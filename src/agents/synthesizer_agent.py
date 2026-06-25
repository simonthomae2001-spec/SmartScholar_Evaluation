from src.core.model_factory import ModelFactory
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# System-Prompt as a constant
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

ADDITIONAL GENERATION CONSTRAINTS:
1. Elaboration: Do not just list or summarize the findings. For each thematic section, provide a deep synthesis, explaining *how* the methodologies achieved their results and *why* certain limitations occurred.
2. Section Depth: Each thematic subsection (e.g., Clinical Applications, Technical Methodologies) must consist of at least two to three fully articulated paragraphs.
3. Analytical Tone: Write in a comprehensive, academic, and flowing textbook style. Avoid brief bullet points or rushed conclusions.

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
        # Initializes LLM via the central ModelFactory
        self.llm = ModelFactory.get_model()
        logger.info("SynthesizerAgent initialisiert.")

    def synthesize_review(self, state: dict, config: dict, status_callback: Callable[[str], None] | None = None) -> dict:
        """
        Generates the final literature review based on the structured analysis data.
        """
        def _log(msg: str):
            if status_callback:
                status_callback(msg)
            logger.info(msg)

        # 1. Extracting data from GraphState
        user_query = state.get("user_query", "")
        analysis_data = state.get("paper_analysis_data", [])

        # Fallback if no data is received
        if not analysis_data:
            _log("⚠️ [Synthesizer] No analysis data found in the state. Aborting synthesis.")
            return {"final_review": "Error: No analysed papers were found for the synthesis."}

        # 2. Format the analysis data for the prompt
        formatted_data = self._format_paper_data(analysis_data)

        # 3. Build Prompt
        prompt = SYNTHESIS_PROMPT.format(
            user_query=user_query,
            paper_data=formatted_data
        )

        _log(f"📝 [Synthesizer] start LLM-Synthesis for {len(analysis_data)} Paper...")

        try:
            # 4. Call LLM (LlamaIndex standard for Ollama calls)
            response = self.llm.complete(prompt)
            
            # Extract the response as a string (the .text suffix is often required in LlamaIndex)
            final_review_text = str(response)
            
            _log("✅ [Synthesizer] Synthesis successfully completed.")
            # 5. Return a partial update for the graph state
            return {"final_review": final_review_text}
            
        except Exception as e:
            _log(f"❌ [Synthesizer] Error during review generation: {str(e)}")
            return {"final_review": f"Error during review generation: {str(e)}"}

    def _format_paper_data(self, analysis_data: list[dict]) -> str:
        """
        A method for converting the list of dictionaries into a clean block of text.
        """
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

