from src.core.orchestrator import build_analysis_graph

state = {
    "user_query": "transformer attention",
    "config_profile": "fast",
    "active_papers": [
        {"paperId": "p1", "title": "Attention Is All You Need", "year": 2017,
         "abstract": "We propose the Transformer, based solely on attention mechanisms.",
         "citation_id": "[1]"},
        {"paperId": "p2", "title": "BERT", "year": 2018,
         "abstract": "BERT pre-trains deep bidirectional representations from unlabeled text.",
         "citation_id": "[2]"},
    ],
}

graph = build_analysis_graph()
result = graph.invoke(state)

print("--- active_papers nach Graph-Lauf ---")
for p in result["active_papers"]:
    print(p["citation_id"], "| status:", p.get("ingestion_status"),
          "| depth:", p.get("ingested_depth"), "| chunk_ids:", p.get("chunk_ids"))

print("\npaper_analysis_data vorhanden:", "paper_analysis_data" in result, "(Analyst-Stub lief)")