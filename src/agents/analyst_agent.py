"""
analyst_agent.py — Structured paper analysis (Steps 12 & 13).

The Analyst receives ingested papers and produces a structured analysis
record for each, covering methodology, key findings, limitations, and
relevance to the user's original query.

Current implementation: **stub** returning mock structured data.
"""

from __future__ import annotations
from typing import Callable, List
from pydantic import BaseModel, Field
from src.core.vector_store import VectorEngine
from src.core.model_factory import ModelFactory
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage


class AnalystAgent:
    """Performs structured analysis on ingested papers."""

    class AnalystQuestions(BaseModel):
        questions: List[str] = Field(default_factory=list, min_length=10, max_length=30, description="A list of questions to ask the vector database to retrieve relevant information about the paper.")

    class AnalysisRecord(BaseModel): 
        methodology: str = Field(description="A concise summary of the paper's methodology.")
        findings: str = Field(description="A concise summary of the paper's key findings.")
        limitations: str = Field(description="A concise summary of the paper's limitations.")
        user_relevance: str = Field(description="An assessment of the paper's relevance to the user's original query.")
    
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



    def __init__(self, collection_name: str = "scholar_papers"):
        self.llm = ModelFactory.get_model()
        self.vector_engine:VectorEngine =None

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
        prompt_template = PromptTemplate(template=prompt_text)
        questions = self.llm.structured_predict(
            output_cls=self.AnalystQuestions,
            prompt=prompt_template
        )
        return questions.questions

    # TODO: Filter for duplicates
    def _query_vector_db(self, title:str, questions:List[str]) ->List[str]:
        """
        Searches the vector database with a set of questions an and returns
        the passages matching the Queries.

        Parameters
        ----------
        title (str) : title of the paper to examine
        questions (List[str]) : questions used as queries for the vector database

        Returns : List[str]
        """
        results = []
        # Create filter for this paper's title
        filters = MetadataFilters(filters=[MetadataFilter(
            key="title",
            value=title
        )])
        retriever = self.vector_engine.get_retriever(
            similarity_top_k=5,
            filters=filters
        )
        
        for question in questions:
            # Query returns list of nodes with text content
            nodes = retriever.retrieve(question)
            # Extract text from retrieved nodes
            passages = [node.get_content() for node in nodes]
            results.append(" ".join(passages))
        return results
    
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
        prompt_template = PromptTemplate(template=prompt)
        record = self.llm.structured_predict(
            output_cls=self.AnalysisRecord,
            prompt=prompt_template
        )
        return record
    
    def _extract_information(self, title:str, orig_query:str) ->AnalysisRecord:
        """
        Extracts information about metrology, key findings or limitations and relevance.

        Parameters
        ----------
        title (str) : title of the paper to examine
        orig_query (str) : the original user query

        Returns : AnalysisRecord
        """
        questions = self._generate_questions(title)
        self._log(f"❓Generated questions: ")
        for q in questions:
            self._log(f"   + {q}")

        passages = self._query_vector_db(title, questions)
        self._log(f"📖Retrieved passages from chroma: {len(passages)} passages found.")

        self._log(f"Generating record...")
        record = self._generate_record(passages, orig_query)
        self._log(f"\n Metrology:\n ``{record.methodology}``\n")
        self._log(f"\n Findings:\n ``{record.findings}``\n")
        self._log(f"\n Limitations:\n ``{record.limitations}``\n")
        
        return record

    def analyze_papers(
        self,
        papers: list[dict],
        query: str,
        vector_engine: VectorEngine,
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
        self._log = lambda msg: status_callback(msg) if status_callback else None
        self.vector_engine = vector_engine
        analysis_data: list[dict] = []

        for idx, paper in enumerate(papers, start=1):
            title = paper.get("title")
            self._log(f"Analyzing paper: {title}")
            pre_record = self._extract_information(title, query)
            
            
            self._log(f"Generating analysis record...")
            record = {
                "citation_id": f"[{idx}]",
                "methodology": pre_record.methodology,
                "findings": pre_record.findings,
                "limitations": pre_record.limitations,
                "user_relevance": (
                    f"Relevance to '{query[:80]}' — "
                    f"{pre_record.user_relevance}"
                ),
            }
            analysis_data.append(record)

        return analysis_data
