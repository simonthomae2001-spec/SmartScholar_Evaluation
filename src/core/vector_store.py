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

Lifecycle (decision 0.C): the collection is reset per research run via
``reset_collection()`` so ChromaDB contains exactly the papers of the
current run — retrievals can't return hits from a previous topic.
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

        # ChromaDB client persisted under ./data/chroma_db (created once).
        self.db = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection_name = collection_name

        # LLM + embedding model from the project factory (created once).
        self.llm = ModelFactory.get_model()
        self.embed_model = ModelFactory.get_embedding_model()

        # Global settings prevent LlamaIndex from falling back to OpenAI.
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Bind the collection -> vector store -> index chain.
        self._build_index()

    # ------------------------------------------------------------------ #
    #  Index lifecycle
    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        """
        (Re)bind the Chroma collection -> vector store -> index chain.

        Shared by ``__init__`` and ``reset_collection`` so the whole chain is
        always rebuilt consistently. After a ``delete_collection`` the old
        ``chroma_collection`` / ``vector_store`` / ``_index`` references are
        stale and MUST be rebuilt — that's exactly what this method does.
        """
        self.chroma_collection = self.db.get_or_create_collection(
            self.collection_name
        )
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        # insert_nodes does NOT re-chunk — it stores our nodes and IDs verbatim.
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def reset_collection(self) -> None:
        """
        Drop all chunks from previous runs and start with a fresh collection.

        The Ingestor calls this at the START of a run (before indexing the
        first paper), guaranteeing ChromaDB holds exactly the finalised papers
        of the current research run.
        """
        try:
            self.db.delete_collection(self.collection_name)
        except Exception:
            # Collection didn't exist yet → nothing to delete.
            pass
        self._build_index()

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