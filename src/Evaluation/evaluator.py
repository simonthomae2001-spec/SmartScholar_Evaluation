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


def load_smartscholar_logs(log_file_path: str) -> Dataset:
    """
    Loads your logged GraphState data from your logs (e.g., JSON/JSONL)
    and transforms them into the exact input format required by Ragas.
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"No log file found at {log_file_path}!")

    with open(log_file_path, "r", encoding="utf-8") as f:
        # Assumes that you logged a list of GraphState dicts
        log_entries = json.load(f)

    questions = []
    contexts_list = []
    answers = []

    for state in log_entries:
        # 1. Extract the original research question
        questions.append(state.get("user_query", ""))

        # 2. Collect the retrieved paper chunks (content of active papers)
        # In your Ingestor/Analyst state, you store the extracted text.
        # Ragas expects a LIST of strings per question (['chunk1', 'chunk2', ...])
        paper_chunks = []
        for paper in state.get("active_papers", []):
            # Use the field where you write your paper texts/abstracts
            chunk_text = paper.get("abstract") or paper.get("ingested_content", "")
            if chunk_text:
                paper_chunks.append(chunk_text)

        contexts_list.append(paper_chunks)

        # 3. Fetch the final generated literature review (Markdown)
        answers.append(state.get("final_review", ""))

    # Creation of the Hugging Face Dataset required by Ragas
    evaluation_data = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers
    }

    return Dataset.from_dict(evaluation_data)


def main():
    print("=== Starting SmartScholar Evaluation Pipeline ===")

    # 1. Connect to your local Ollama models
    print("Initializing evaluator models via ModelFactory...")

    # Use your existing SmartScholar Factory to fetch LLM and Embeddings
    ollama_llm = ChatOllama(
        model="llama3",  # Or whichever model you have installed locally in Ollama
        temperature=0.0,
        format="json"
    )
    ollama_embeddings = ModelFactory.get_embedding_model()

    # Wrap into the LangChain wrappers required by Ragas
    ragas_llm = LangchainLLMWrapper(ollama_llm)

    # 2. The main() function: Create metrics and throttle via "backdoor"
    print("Initializing Ragas metrics...")

    # Create metrics empty without arguments
    metric_faithfulness = Faithfulness(llm=ragas_llm)
    metric_context_utilization = ContextUtilization(llm=ragas_llm)

    print("Enforcing single-thread mode for Ollama via object injection...")
    # Our single-thread throttle for Ollama
    ollama_config = RunConfig(max_workers=1, timeout=60)

    # The trick: We write the RunConfig directly into the internal
    # property dictionary of the metric objects. This bypasses any error messages!
    metric_faithfulness.__dict__["run_config"] = ollama_config
    metric_context_utilization.__dict__["run_config"] = ollama_config

    metrics = [metric_faithfulness, metric_context_utilization]

    # 3. Load your experiment data (WITH ABSOLUTE PATH FIX)
    # Determine the directory where THIS script is located
    # Find the "Evaluate/" folder
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one folder to the main project and then enter "data/"
    log_path = os.path.join(current_dir, "..", "..", "data", "experiment_baseline_logs.json")

    # Clean up the path (remove the "..")
    log_path = os.path.normpath(log_path)

    print(f"Searching for log file at: {log_path}")

    dataset = load_smartscholar_logs(log_path)
    print(f"Successfully loaded {len(dataset)} test cases from logs.")

    # 4. Run evaluation (NOW ABSOLUTELY CLEAN, NO WRONG KEYWORDS)
    print("Running LLM-as-a-Judge evaluation (Ollama is now running serially)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics
        # No extra keywords here anymore – Ragas gets the info via the metrics!
    )

    # 5. Evaluate and save results (DYNAMIC FIX FOR KEYERROR)
    print("\n=== EVALUATION RESULTS ===")
    df = results.to_pandas()

    # Check live what Ragas wrote into the DataFrame
    print(f"Available columns in result: {list(df.columns)}")

    # Search for the matching question column (often 'question' or 'user_input')
    frage_col = 'user_input' if 'user_input' in df.columns else (
        'question' if 'question' in df.columns else df.columns[0])

    # Print the entire DataFrame so nothing can go wrong!
    print("\nAll test results:")
    print(df.to_string())

    # Calculate average scores (dynamically searching for numerical columns)
    print("\n=== MEAN VALUES ===")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            print(f"Average {col}: {df[col].mean():.2f}")

    # Export results as CSV
    output_csv = os.path.join(log_path, "..", "evaluation_results_summary.csv")
    output_csv = os.path.normpath(output_csv)

    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to '{output_csv}'.")


if __name__ == "__main__":
    main()