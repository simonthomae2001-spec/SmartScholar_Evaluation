"""
vector_store.py — ChromaDB + LlamaIndex vector engine for SmartScholar.

The VectorEngine is the "shelf": it owns chunking, embedding, and storage.
The IngestorAgent (the "librarian") calls index_paper / index_paper_by_pages
per paper and gets back the list of ``chunk_ids`` it then attaches to the
paper dict in ``active_papers``.

Key contract (decided with the team):
- Large content (chunks + embeddings) lives ONLY in ChromaDB.
- ``active_papers`` stays lean — chunk_ids + ingestion_status only.
- Chunk IDs are predictable: f"{citation_id}_chunk_{i}" (e.g. "[1]_chunk_0").
- Every chunk carries ``citation_id`` in its metadata (backward reference).
  PRO chunks additionally carry ``page_number`` (fine-grained reference for
  the Critic: "claim from [3], page 7").

Chunking is config-driven (SSOT): ``chunk_size`` (TOKENS) comes from the
active profile (fast 1024 / medium 512 / pro 256).

Lifecycle: the collection is cleared by the UI (session start / "Start Over")
via the module-level ``clear_collection()`` — a lightweight delete that does
NOT load any model. The Ingestor then re-binds to the (empty) collection via
``VectorEngine.rebind()`` before ingesting.
"""

import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.core.model_factory import ModelFactory


# Explicit overlap as a fraction of chunk_size is removed.
# Chunking is config-driven and overlap is now provided explicitly.

# Defaults shared with clear_collection() so the UI can clear the exact same
# collection the VectorEngine uses, without instantiating a VectorEngine.
_DEFAULT_COLLECTION = "scholar_papers"
_DEFAULT_DB_PATH = "./data/chroma_db"


class VectorEngine:
    """Owns chunking, embedding, and ChromaDB storage for one research run."""

    def __init__(self, collection_name: str = _DEFAULT_COLLECTION):
        os.makedirs("./data", exist_ok=True)

        self.db = chromadb.PersistentClient(path=_DEFAULT_DB_PATH)
        self.collection_name = collection_name

        self.llm = ModelFactory.get_model()
        self.embed_model = ModelFactory.get_embedding_model()
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self._build_index()

    # ------------------------------------------------------------------ #
    #  Index lifecycle
    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        """(Re)bind the Chroma collection -> vector store -> index chain."""
        self.chroma_collection = self.db.get_or_create_collection(
            self.collection_name
        )
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def reset_collection(self) -> None:
        """Drop all chunks from previous runs and start with a fresh collection."""
        try:
            self.db.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection didn't exist yet → nothing to delete.
        self._build_index()

    def rebind(self) -> None:
        """
        Re-bind to the current ChromaDB collection, recreating it empty if it
        was deleted out from under us (e.g. the UI cleared it on 'Start Over').

        Cheap — it only refreshes the collection / index handles; the embedding
        model stays loaded on ``self`` and is NOT reloaded. The Ingestor calls
        this before each run so its writes land in the freshly-cleared store.
        """
        self._build_index()

    # ------------------------------------------------------------------ #
    #  Internal: build a SentenceSplitter for a given chunk_size
    # ------------------------------------------------------------------ #
    def _make_splitter(self, chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # ------------------------------------------------------------------ #
    #  Ingestion: single text blob (FAST abstract, MEDIUM hybrid)
    # ------------------------------------------------------------------ #
    def index_paper(
        self,
        text: str,
        citation_id: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict | None = None,
    ) -> list[str]:
        """
        Chunk a single text blob and store the chunks in ChromaDB.

        Every chunk gets a predictable id ``f"{citation_id}_chunk_{i}"`` and
        metadata that ALWAYS contains ``citation_id``. Returns the chunk_ids
        (empty list if ``text`` is empty).
        """
        if not text or not text.strip():
            return []

        splitter = self._make_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)

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
    #  Ingestion: page-aware (PRO full_pdf)
    # ------------------------------------------------------------------ #
    def index_paper_by_pages(
        self,
        pages: list[tuple[int, str]],
        citation_id: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict | None = None,
    ) -> list[str]:
        """
        Chunk each page separately so every chunk carries its 1-based
        ``page_number`` in metadata — the fine-grained backward reference the
        Critic uses ("claim from [3], page 7").

        Chunk IDs stay globally unique across the whole paper:
        ``f"{citation_id}_chunk_{i}"`` with ``i`` running over ALL chunks of
        all pages (not resetting per page). Returns the chunk_ids.
        """
        if not pages:
            return []

        splitter = self._make_splitter(chunk_size, chunk_overlap)

        base_meta = dict(metadata or {})
        base_meta["citation_id"] = citation_id

        nodes: list[TextNode] = []
        chunk_ids: list[str] = []
        global_i = 0
        for page_number, page_text in pages:
            page_text = (page_text or "").strip()
            if not page_text:
                continue
            for chunk in splitter.split_text(page_text):
                chunk_id = f"{citation_id}_chunk_{global_i}"
                meta = dict(base_meta)
                meta["page_number"] = int(page_number)  # fine-grained ref
                nodes.append(TextNode(text=chunk, id_=chunk_id, metadata=meta))
                chunk_ids.append(chunk_id)
                global_i += 1

        if nodes:
            self._index.insert_nodes(nodes)

        return chunk_ids

    # ------------------------------------------------------------------ #
    #  Retrieval (used later by the Analyst / Critic)
    # ------------------------------------------------------------------ #
    def get_query_engine(self):
        """Return a query engine over the current collection."""
        return self._index.as_query_engine(llm=self.llm)

    def get_retriever(self, similarity_top_k: int = 3, filters=None):
        """Return a retriever (vector search only, no LLM synthesis).

        Parameters
        ----------
        similarity_top_k : int
            Number of top results to return.
        filters : MetadataFilters, optional
            Metadata filters to apply (e.g., filter by title).
        """
        return self._index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=filters
        )


# ====================================================================== #
#  Module-level: lightweight collection clear for the UI
# ====================================================================== #

def clear_collection(
    collection_name: str = _DEFAULT_COLLECTION,
    db_path: str = _DEFAULT_DB_PATH,
) -> None:
    """
    Delete the ChromaDB collection WITHOUT loading any models.

    The UI calls this on session start and on 'Start Over', so the knowledge
    base is emptied immediately (page stays fast) instead of lazily at the next
    ingestion. The Ingestor's VectorEngine.rebind() then recreates the (empty)
    collection on the next run. Best-effort — never raises.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        client.delete_collection(collection_name)
    except Exception:
        pass  # didn't exist → nothing to clear