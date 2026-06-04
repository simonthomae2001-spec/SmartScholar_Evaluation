"""
vector_store.py — ChromaDB + LlamaIndex vector engine for SmartScholar.

The VectorEngine is the "shelf": it owns chunking, embedding, and storage.
The IngestorAgent (the "librarian") calls ``index_paper`` per paper and gets
back the list of ``chunk_ids`` it then attaches to the paper dict in
``active_papers``.

Key contract (decided with the team):
- Large content (chunks + embeddings) lives ONLY in ChromaDB.
- ``active_papers`` stays lean — it only carries chunk_ids (a foreign-key
  reference into the DB) plus an ingestion_status.
- Chunk IDs are predictable: f"{citation_id}_chunk_{i}" (e.g. "[1]_chunk_0").
- Every chunk carries ``citation_id`` in its metadata — the backward
  reference the Critic uses to trace a retrieved chunk back to its paper.

Chunking is config-driven (SSOT): ``chunk_size`` comes from the active
profile and is measured in TOKENS (SentenceSplitter's unit), matching the
architecture doc (fast 1024 / medium 512 / pro 256).
"""

import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.core.model_factory import ModelFactory


# Explicit overlap as a fraction of chunk_size, so small (pro) chunks don't
# drown in the SentenceSplitter default overlap of 200 tokens.
_CHUNK_OVERLAP_RATIO = 0.15


class VectorEngine:
    """Owns chunking, embedding, and ChromaDB storage for one research run."""

    def __init__(self, collection_name: str = "scholar_papers"):
        # Ensure the data directory exists.
        os.makedirs("./data", exist_ok=True)

        # ChromaDB client persisted under ./data/chroma_db
        self.db = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection_name = collection_name
        self.chroma_collection = self.db.get_or_create_collection(collection_name)

        # LlamaIndex vector store on top of the Chroma collection.
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # LLM + embedding model from the project factory.
        self.llm = ModelFactory.get_model()
        self.embed_model = ModelFactory.get_embedding_model()

        # Global settings prevent LlamaIndex from falling back to OpenAI.
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # A single index bound to the vector store. We insert nodes into it
        # per paper (insert_nodes does NOT re-chunk — it stores our nodes and
        # IDs verbatim, which is exactly what we need).
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    # ------------------------------------------------------------------ #
    #  Core ingestion entry point
    # ------------------------------------------------------------------ #
    def index_paper(
        self,
        text: str,
        citation_id: str,
        chunk_size: int,
        metadata: dict | None = None,
    ) -> list[str]:
        """
        Chunk ``text`` and store the chunks in ChromaDB.

        Steps
        -----
        1. Split ``text`` into token-sized chunks with a SentenceSplitter
           (so words/sentences aren't cut mid-way), using the profile's
           ``chunk_size`` and a 15 % overlap.
        2. Build a ``TextNode`` per chunk with a predictable id
           ``f"{citation_id}_chunk_{i}"`` and metadata that ALWAYS contains
           ``citation_id`` (the backward reference for the Critic).
        3. Embed + store the nodes via ``insert_nodes`` (no re-chunking).

        Parameters
        ----------
        text : str
            The text to ingest (an abstract, selected sections, or full PDF
            text — the Ingestor decides which, per profile).
        citation_id : str
            The paper's citation id, e.g. "[1]". Used as the id prefix and
            stored in every chunk's metadata.
        chunk_size : int
            Tokens per chunk, from ``config["chunk_size"]``.
        metadata : dict | None
            Extra flat metadata (title, year, page, ...). Values must be
            simple types (str/int/float) — ChromaDB rejects lists/dicts.

        Returns
        -------
        list[str]
            The chunk_ids that were created and stored, in order. Empty list
            if ``text`` is empty.
        """
        if not text or not text.strip():
            return []

        overlap = max(1, int(chunk_size * _CHUNK_OVERLAP_RATIO))
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)

        # citation_id is the backward reference (DB hit -> which paper).
        base_meta = dict(metadata or {})
        base_meta["citation_id"] = citation_id

        nodes: list[TextNode] = []
        chunk_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{citation_id}_chunk_{i}"
            nodes.append(
                TextNode(text=chunk, id_=chunk_id, metadata=dict(base_meta))
            )
            chunk_ids.append(chunk_id)

        if nodes:
            self._index.insert_nodes(nodes)

        return chunk_ids

    # ------------------------------------------------------------------ #
    #  Retrieval (used later by the Analyst / Critic)
    # ------------------------------------------------------------------ #
    def get_query_engine(self):
        """Return a query engine over the current collection."""
        return self._index.as_query_engine(llm=self.llm)

    def get_retriever(self, similarity_top_k: int = 3):
        """Return a retriever (vector search only, no LLM synthesis)."""
        return self._index.as_retriever(similarity_top_k=similarity_top_k)