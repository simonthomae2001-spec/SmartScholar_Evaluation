from src.core.config import get_config
from src.agents.ingestor_agent import IngestorAgent

# Fake "finalisierte" Paper, wie der Researcher sie uebergeben wuerde
papers = [
    {
        "paperId": "p1",
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": "The dominant sequence transduction models are based on "
                    "complex recurrent or convolutional neural networks. We "
                    "propose the Transformer, based solely on attention "
                    "mechanisms, dispensing with recurrence entirely.",
    },
    {
        "paperId": "p2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "year": 2018,
        "abstract": "We introduce BERT, a language representation model that "
                    "pre-trains deep bidirectional representations from "
                    "unlabeled text.",
    },
    {
        "paperId": "p3",
        "title": "Paper ohne Abstract",
        "year": 2020,
        "abstract": None,   # Edge Case: kein Abstract
    },
]

cfg = get_config("fast")   # read_depth=abstract, chunk_size=1024

ing = IngestorAgent(collection_name="test_fast_ingest")
result = ing.ingest_knowledge(papers, cfg, status_callback=print)

print("\n--- Ergebnis (schlankes active_papers) ---")
for p in result:
    print(
        p["citation_id"],
        "| status:", p["ingestion_status"],
        "| depth:", p["ingested_depth"],
        "| chunk_ids:", p["chunk_ids"],
    )

print("\n--- Retrieval-Gegencheck (Inhalt wirklich in ChromaDB?) ---")
for r in ing.vector_engine.get_retriever(similarity_top_k=2).retrieve("transformer attention"):
    print("Chunk:", r.node.id_, "| citation_id:", r.node.metadata.get("citation_id"))