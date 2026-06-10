from src.core.config import get_config
from src.agents.ingestor_agent import IngestorAgent

papers = [
    {  # echtes PDF -> page-aware Ingestion (full_pdf)
        "paperId": "attn1",
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": "We propose the Transformer.",
        "openAccessPdf": "https://arxiv.org/pdf/1706.03762",
    },
    {  # kein PDF -> Fallback auf Abstract
        "paperId": "nopdf1",
        "title": "Paper hinter Paywall",
        "year": 2019,
        "abstract": "Abstract vorhanden, aber kein PDF.",
        "openAccessPdf": None,
    },
]

cfg = get_config("pro")   # read_depth=full_pdf, chunk_size=256

ing = IngestorAgent(collection_name="test_pro_ingest")
result = ing.ingest_knowledge(papers, cfg, status_callback=print)

print("\n--- Ergebnis ---")
for p in result:
    print(p["citation_id"], "| status:", p["ingestion_status"],
          "| depth:", p["ingested_depth"], "| #chunks:", len(p["chunk_ids"]))

# Beweis: tragen die Chunks Seitenzahlen?
print("\n--- Retrieval mit Seitenzahl ---")
for r in ing.vector_engine.get_retriever(similarity_top_k=3).retrieve("attention mechanism"):
    print("Chunk:", r.node.id_, "| page:", r.node.metadata.get("page_number"))