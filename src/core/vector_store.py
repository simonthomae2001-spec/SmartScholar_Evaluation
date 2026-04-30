import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings

from llama_index.core.schema import Document
from core.model_factory import ModelFactory

class VectorEngine:
    def __init__(self, collection_name: str = "scholar_papers"):
        # Ensure data directory exists
        os.makedirs("./data", exist_ok=True)
        
        # Initialize ChromaDB client pointing to ./data/chroma_db
        self.db = chromadb.PersistentClient(path="./data/chroma_db")
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        
        # Initialize LlamaIndex ChromaVectorStore
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Get LLM and Embedding model from ModelFactory
        self.llm = ModelFactory.get_model()
        self.embed_model = ModelFactory.get_embedding_model()
        
        # Set global settings to prevent default OpenAI fallback
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model


    def index_papers(self, papers: list) -> VectorStoreIndex:
        """
        Converts the list of papers into LlamaIndex Document objects (with metadata)
        and creates/updates the index.
        """
        documents = []
        for paper in papers:
            # Use abstract as the text content to be indexed. 
            # Fallback to title if abstract is not available.
            text = paper.get("abstract") or f"Title: {paper.get('title')}"
            
            # Ensure metadata values are simple types (str, int, float, bool)
            metadata = {
                "title": str(paper.get("title") or "Unknown Title"),
                "authors": ", ".join(paper.get("authors") or []),
                "year": str(paper.get("year") or "Unknown"),
                "url": str(paper.get("url") or ""),
                "openAccessPdf": str(paper.get("openAccessPdf") or "")
            }
            
            doc = Document(
                text=text,
                metadata=metadata,
                excluded_llm_metadata_keys=["url", "openAccessPdf"],
                excluded_embed_metadata_keys=["url", "openAccessPdf"]
            )
            documents.append(doc)
            
        # Create the index from documents
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            llm=self.llm,
            embed_model=self.embed_model
        )
        return index

    def get_query_engine(self, index: VectorStoreIndex = None):
        """
        Returns a query engine for the index.
        """
        if index is None:
            # Load index from existing vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                llm=self.llm,
                embed_model=self.embed_model
            )
        return index.as_query_engine(llm=self.llm)

