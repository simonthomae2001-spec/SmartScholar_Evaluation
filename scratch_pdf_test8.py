from src.core.vector_store import VectorEngine

eng = VectorEngine(collection_name="test_lifecycle")
text = "Self-attention weighs all tokens at once. " * 20

# 1) Befuellen
ids1 = eng.index_paper(text, "[1]", chunk_size=50, metadata={"title": "Paper A"})
treffer_vorher = eng.get_retriever(similarity_top_k=5).retrieve("self-attention")
print("Vor Reset  -> chunk_ids:", len(ids1), "| Treffer:", len(treffer_vorher))

# 2) Reset
eng.reset_collection()
treffer_nachher = eng.get_retriever(similarity_top_k=5).retrieve("self-attention")
print("Nach Reset -> Treffer:", len(treffer_nachher), "(sollte 0 sein)")

# 3) Frisch wieder befuellbar?
ids2 = eng.index_paper(text, "[1]", chunk_size=50, metadata={"title": "Paper A"})
treffer_neu = eng.get_retriever(similarity_top_k=5).retrieve("self-attention")
print("Neu gefuellt -> chunk_ids:", len(ids2), "| Treffer:", len(treffer_neu))