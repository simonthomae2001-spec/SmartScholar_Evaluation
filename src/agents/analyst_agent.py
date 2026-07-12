"""
analyst_agent.py — Structured paper analysis (Steps 12 & 13).

The Analyst receives ingested papers and produces a structured analysis
record for each, covering methodology, key findings, limitations, and
relevance to the user's original query.

Current implementation: **stub** returning mock structured data.
"""

from __future__ import annotations

from typing import Callable, List, Dict

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from pydantic import BaseModel, Field

from src.core.model_factory import ModelFactory
from src.core.vector_store import VectorEngine
from src.tools.llm_tools import structured_predicted_with_retries
from src.core.config import get_system_config

class AnalystAgent:
    """Performs structured analysis on ingested papers."""

    class AnalystQuestions(BaseModel):
        questions: List[str] = Field(default_factory=list, min_length=3, max_length=30, description="A list of questions to ask the vector database to retrieve relevant information about the paper.")

    class AnalysisRecord(BaseModel): 
        methodology: str = Field(description="A concise summary of the paper's methodology.")
        findings: str = Field(description="A concise summary of the paper's key findings.")
        limitations: str = Field(description="A concise summary of the paper's limitations.")
        user_relevance: str = Field(description="An assessment of the paper's relevance to the user's original query.")
    
    FALLBACK_QUESTIONS = AnalystQuestions(questions=["What methodology is used?", "What are the key findings?", "What limitations where mentioned?"])
    FALLBACK_RECORD = AnalysisRecord(methodology="Nothing found", findings="Nothing found", limitations="Nothing found", user_relevance="not relevant")


    SEARCH_PROMPT = (
        "You have to generate a list of strings."
        "The strings should contains questions one can ask about an academic paper to find out about the methodology used, key findings, limitations." 
        "Try to keep the questions short and designe them in a way that they can be used as queries to a vector database."

        "The title of the paper is: "
        "{title}"
    )
    SUMMARY_PROMPT = (
        "You have to fill out all the sections of an analysis record based on the following extracted passages from an academic paper."
        " The sections are: methodology, key findings, limitations and overall relevance for the users original query. "
        " The original user query was:"
        " {user_query}"
        "Provided passages:"
        "{passages}"
    )
    FEEDBACK_PROMPT = (
        "You have to filled out all the sections of an analysis record based on the following extracted passages from an academic paper."
        " The sections are: methodology, key findings, limitations and overall relevance for the users original query. "
        " The original user query was:"
        " {user_query}"
        "Provided passages:"
        "{passages}"
        "Your output was reviewed an needs improvements."
        "Here is the feedback provided:"
        "{feedback}"
    )



    def __init__(self, collection_name: str = "scholar_papers"):
        self.llm = ModelFactory.get_model()
        self.vector_engine:VectorEngine =None
        self._analysis_data: Dict[dict] = {}
        self.config = {}

    def _generate_questions(self, title:str) ->List[str]:
        """
        Generates a list of questions in order to extract 
        methodology, key findings, limitations of a given paper.

        Parameters
        ----------
        title (str) : title of the paper to examine

        Returns : List[str]
        """
        prompt_text = self.SEARCH_PROMPT.format(title=title)
        questions = structured_predicted_with_retries(
            llm= self.llm,
            output_cls= self.AnalystQuestions,
            messages=[ChatMessage(role=MessageRole.USER, content=prompt_text)],
            logger=self._log,
            msg= "Failed generating AnalystQuestions! Retrying...",
            llm_kwargs={
                "temperature": self.config.get("llm", {}).get("temperature_analytical", 0.1),
                "num_predict": self.config.get("llm", {}).get("max_tokens_short", 500)
            }
        )
        if questions is None:
            self._log("⚠ [Analyst] Failed to generate questions, using fallback")
            return self.FALLBACK_QUESTIONS
        
        return questions.questions

    @staticmethod
    def _normalize_id(cid: str | int) -> str:
        s_id = str(cid).strip()
        if not s_id.startswith("["):
            s_id = f"[{s_id}]"
        if not s_id.endswith("]"):
            s_id = f"{s_id}]"
        return s_id

    # TODO: Filter for duplicates
    def _query_vector_db(self, title:str, questions:List[str], citation_id: str | int) ->List[str]:
        """
        Searches the vector database with a set of questions an and returns
        the passages matching the Queries.

        Parameters
        ----------
        title (str) : title of the paper to examine
        questions (List[str]) : questions used as queries for the vector database

        Returns : List[str]
        """
        results = set()
        # Create filter for this paper's citation_id
        norm_id = self._normalize_id(citation_id)
        filters = MetadataFilters(filters=[MetadataFilter(
            key="citation_id",
            value=norm_id
        )])
        top_k = self.config.get("rag", {}).get("analyst_top_k", 5)
        retriever = self.vector_engine.get_retriever(
            similarity_top_k=top_k,
            filters=filters
        )
        
        for question in questions:
            # Query returns list of nodes with text content
            nodes = retriever.retrieve(question)
            # Extract text from retrieved nodes
            for node in nodes:
                results.add(node.get_content())
        return list(results)
    
    def _generate_record(self, query_results: List[str], orig_query:str) ->AnalysisRecord:
        """
        Builds an AnalysisRecord that based on the information contained in the query_results.

        Parameters
        ----------
        query_results (str) : title of the paper to examine
        orig_query (str) : the original user query

        Returns : AnalysisRecord
        """
        prompt = self.SUMMARY_PROMPT.format(user_query=orig_query, passages=query_results)
        record = structured_predicted_with_retries(
            llm= self.llm,
            output_cls= self.AnalysisRecord,
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            logger=self._log,
            msg= "Failed generating AnalysisRecord! Retrying...",
            llm_kwargs={
                "temperature": self.config.get("llm", {}).get("temperature_analytical", 0.1),
                "num_predict": self.config.get("llm", {}).get("max_tokens_long", 1000)
            }
        )
        if record is None:
            self._log("⚠ [Analyst] Failed to generate record, using fallback")
            return self.FALLBACK_RECORD
        
        return record

            

    def _extract_information(self, title:str, orig_query:str, citation_id: str | int) ->AnalysisRecord:
        """
        Extracts information about metrology, key findings or limitations and relevance.

        Parameters
        ----------
        title (str) : title of the paper to examine
        orig_query (str) : the original user query

        Returns : AnalysisRecord
        """
        questions = self._generate_questions(title)
        self._log(f"🧠 [Analyst] Generated {len(questions)} questions:")
        for idx, q in enumerate(questions):
            self._log(f"  ↳ {idx}. {q}")

        passages = self._query_vector_db(title, questions, citation_id)
        self._log(f"  ↳ Retrieved {len(passages)} passages from ChromaDB")

        self._log(f"🧠 [Analyst] Generating analysis record...")
        record = self._generate_record(passages, orig_query)
        sys_cfg = get_system_config()
        trunc_len = sys_cfg.get("ui", {}).get("log_snippet_truncation", 120)
        self._log(f"  ↳ Methodology: {record.methodology[:trunc_len]}")
        self._log(f"  ↳ Findings: {record.findings[:trunc_len]}")
        self._log(f"  ↳ Limitations: {record.limitations[:trunc_len]}")
        
        return record
    
    def _process_feedback(self, title: str, orig_query:str, feedback: Dict, citation_id: str | int)-> AnalysisRecord:
        questions = self._generate_questions(title)
        passages = self._query_vector_db(title, questions, citation_id)
        prompt = self.FEEDBACK_PROMPT.format(user_query=orig_query, passages=passages, feedback=str(feedback))
        record = structured_predicted_with_retries(
            llm= self.llm,
            output_cls= self.AnalysisRecord,
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            logger=self._log,
            msg= "Failed generating AnalysisRecord! Retrying...",
            llm_kwargs={
                "temperature": self.config.get("llm", {}).get("temperature_analytical", 0.1),
                "num_predict": self.config.get("llm", {}).get("max_tokens_long", 1000)
            }
        )
        if record is None:
            self._log("⚠ [Analyst] Failed to generate record, using fallback")
            return self.FALLBACK_RECORD
        
        return record
    
    def _log(self, msg:str):
        if not self.status_callback is None:
            self.status_callback(msg)

    def analyze_papers(
        self,
        papers: list[dict],
        query: str,
        vector_engine: VectorEngine,
        feedback: Dict[int, dict],
        config: dict,
        status_callback: Callable[[str], None] | None = None,
    ) -> list[dict]:
        """
        Produce a structured analysis record for each paper.

        Parameters
        ----------
        papers : list[dict]
            Papers with ingested content (from the IngestorAgent).
        query : str
            The original user query (used to assess relevance).

        Returns
        -------
        list[dict]
            One record per paper with keys:
            ``citation_id``, ``methodology``, ``findings``,
            ``limitations``, ``user_relevance``.
        """
        self.status_callback = status_callback
        self.vector_engine = vector_engine
        self.config = config
        
        # Clear stale records from previous runs or UI resets
        self._analysis_data.clear()
        
        # Defensively re-hydrate the vector engine to ensure binding to the active collection
        if hasattr(self.vector_engine, 'rebind'):
            self.vector_engine.rebind()

        for idx, paper in enumerate(papers, start=1):
            title = paper.get("title")

            if len(feedback) == 0:
                self._log(f"🧠 [Analyst] Analyzing paper: {title}")
                pre_record = self._extract_information(title, query, idx)
            else:
                self._log(f"🧠 [Analyst] Revising record for: {title}")
                pre_record = self._process_feedback(title, query, feedback[idx], idx)

                        
            record = {
                "citation_id": idx,
                "methodology": pre_record.methodology,
                "findings": pre_record.findings,
                "limitations": pre_record.limitations,
                "user_relevance": (
                    f"Relevance to '{query[:80]}' — "
                    f"{pre_record.user_relevance}"
                ),
            }
            self._analysis_data[idx] = record

        return list(self._analysis_data.values())
