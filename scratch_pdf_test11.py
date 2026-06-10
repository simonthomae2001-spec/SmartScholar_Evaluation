from src.core.config import get_config
from src.agents.ingestor_agent import IngestorAgent

papers = [
    {  # echtes Open-Access-PDF -> sollte PDF indizieren (hybrid)
        "paperId": "attn1",
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": "We propose the Transformer, based solely on attention mechanisms.",
        "openAccessPdf": "https://arxiv.org/pdf/1706.03762",
    },
    {  # kein PDF -> lautloser Fallback auf Abstract
        "paperId": "nopdf1",
        "title": "Paper hinter Paywall",
        "year": 2019,
        "abstract": "This paper has an abstract but no open-access PDF.",
        "openAccessPdf": None,
    },
    {  # weder PDF noch Abstract -> no_content
        "paperId": "empty1",
        "title": "Paper ohne alles",
        "year": 2020,
        "abstract": None,
        "openAccessPdf": None,
    },
]

cfg = get_config("medium")   # read_depth=hybrid, chunk_size=512

ing = IngestorAgent(collection_name="test_medium_ingest")
result = ing.ingest_knowledge(papers, cfg, status_callback=print)

print("\n--- Ergebnis ---")
for p in result:
    print(p["citation_id"], "| status:", p["ingestion_status"],
          "| depth:", p["ingested_depth"], "| #chunks:", len(p["chunk_ids"]))