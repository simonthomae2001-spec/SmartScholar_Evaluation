from src.core.vector_store import VectorEngine

eng = VectorEngine(collection_name="test_ingest")

# Genug Text, damit bei kleiner chunk_size mehrere Chunks entstehen
text = (
    "Transformers rely on self-attention. "
    "Self-attention lets the model weigh all tokens at once. "
    "This avoids recurrence and improves parallelism during training. "
) * 8

ids = eng.index_paper(
    text=text,
    citation_id="[1]",
    chunk_size=50,   # klein, um mehrere Chunks zu erzwingen
    metadata={"title": "Attention Is All You Need", "year": "2017"},
)

print("Erzeugte chunk_ids:", ids)
print("Anzahl Chunks:", len(ids))

# Rueckwaerts-Referenz pruefen: kommt die citation_id beim Retrieval mit?
print("\n--- Retrieval-Test ---")
retriever = eng.get_retriever(similarity_top_k=3)
for r in retriever.retrieve("What is self-attention?"):
    print(
        "Chunk-ID:", r.node.id_,
        "| citation_id:", r.node.metadata.get("citation_id"),
        "| title:", r.node.metadata.get("title"),
    )