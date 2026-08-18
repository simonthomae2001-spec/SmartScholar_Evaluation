import os
import json
import pandas as pd
from datasets import Dataset

# Ragas Core & Metrics
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextUtilization
from ragas.run_config import RunConfig

# LangChain wrapper for Ragas (to connect Ollama)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import your SmartScholar infrastructure
from src.core.model_factory import ModelFactory
from langchain_community.chat_models import ChatOllama

from ragas.run_config import RunConfig

# Extrem wichtig für lokale Modelle: max_workers=1 erzwingt DURCHGEHEND serielle Verarbeitung!
ollama_run_config = RunConfig(
    max_workers=1,   # Genau 1 LLM-Aufruf nach dem anderen!
    timeout=600.0,   # 10 Minuten Puffer pro Einzel-Aufruf
    max_retries=2
)


def load_smartscholar_logs(log_file_path: str) -> Dataset:
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"No log file found at {log_file_path}!")

    with open(log_file_path, "r", encoding="utf-8") as f:
        log_entries = json.load(f)

    questions = []
    contexts_list = []
    answers = []

    for state in log_entries:
        questions.append(state.get("user_query", ""))

        # 1. Bevorzugt: Tatsächlicher Synthesizer-Input (paper_analysis_data)
        analysis_data = state.get("paper_analysis_data", [])

        if analysis_data:
            paper_chunks = [
                f"Citation ID: [{p.get('citation_id', '?')}]\n"
                f"Relevance: {p.get('user_relevance', '')}\n"
                f"Methodology: {p.get('methodology', '')}\n"
                f"Findings: {p.get('findings', '')}\n"
                f"Limitations: {p.get('limitations', '')}"
                for p in analysis_data
            ]
        else:
            # 2. Fallback: Abstract / Ingested Content aus active_papers
            paper_chunks = [
                p.get("ingested_content") or p.get("abstract", "")
                for p in state.get("active_papers", [])
            ]

        contexts_list.append(paper_chunks)
        answers.append(state.get("final_review", ""))

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "answer": answers
    })


def main():
    print("=== Starting SmartScholar Evaluation Pipeline ===")

    # 1. Connect to local Ollama model (OHNE format="json"!)
    from src.core.config import get_system_config
    cfg = get_system_config()
    temp = cfg.get("llm", {}).get("temperature_analytical", 0.0)

    ollama_llm = ChatOllama(
        model=ModelFactory.get_model_name(),
        temperature=temp,
        # 'format="json"' REMOVED: Cause of infinite loops / timeouts in Ragas!
    )

    ragas_llm = LangchainLLMWrapper(ollama_llm)

    # 2. Setup Metrics with higher timeout
    metric_faithfulness = Faithfulness(llm=ragas_llm)
    metric_context_utilization = ContextUtilization(llm=ragas_llm)

    # 300 seconds timeout for large reviews & multi-statement verification
    ollama_config = RunConfig(max_workers=1, timeout=300.0)

    metric_faithfulness.__dict__["run_config"] = ollama_config
    metric_context_utilization.__dict__["run_config"] = ollama_config

    metrics = [metric_faithfulness, metric_context_utilization]

    # 3. Load Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "experiment_baseline_logs_multi_correction.json"))

    print(f"Loading logs from: {log_path}")
    full_dataset = load_smartscholar_logs(log_path)
    total_items = len(full_dataset)
    print(f"Successfully loaded {total_items} test cases.")

    # 4. Process in Batches (e.g. 3 items per iteration)
    BATCH_SIZE = 3
    all_results_df = pd.DataFrame()

    output_csv = os.path.normpath(os.path.join(log_path, "..", "evaluation_results_multi_agent_correction2.csv"))

    for i in range(0, total_items, BATCH_SIZE):
        batch_dataset = full_dataset.select(range(i, min(i + BATCH_SIZE, total_items)))
        print(
            f"\n--- Processing Batch {i // BATCH_SIZE + 1} / {(total_items + BATCH_SIZE - 1) // BATCH_SIZE} (Items {i + 1} to {min(i + BATCH_SIZE, total_items)}) ---")

        try:
            # Hier übergeben wir run_config direkt als Parameter
            results = evaluate(
                dataset=batch_dataset,
                metrics=metrics,
                run_config=ollama_run_config
            )

            batch_df = results.to_pandas()
            all_results_df = pd.concat([all_results_df, batch_df], ignore_index=True)

            # Zwischenspeichern
            all_results_df.to_csv(output_csv, index=False)
            print(f"✓ Batch saved. Current progress: {len(all_results_df)}/{total_items} items.")

        except Exception as e:
            print(f"❌ Error during batch evaluation: {e}")

    # 5. Summary
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(f"Available columns: {list(all_results_df.columns)}")
    print(all_results_df.to_string())

    print("\n=== MEAN VALUES ===")
    for col in all_results_df.columns:
        if all_results_df[col].dtype in ['float64', 'int64']:
            print(f"Average {col}: {all_results_df[col].mean():.4f}")

    print(f"\nSaved final results to '{output_csv}'.")


if __name__ == "__main__":
    main()