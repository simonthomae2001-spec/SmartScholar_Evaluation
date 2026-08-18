import os
import json
import re
import pandas as pd
from langchain_community.chat_models import ChatOllama
from src.core.model_factory import ModelFactory

# 1. Dateipfade anpassen falls nötig
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON_PATH = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "experiment_baseline_logs_multi_correction.json"))

OUTPUT_CSV_PATH = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "evaluation_results_multi_agent_correction_2.csv"))


def extract_context(entry: dict) -> str:
    """Extrahiert den Kontext als Freitext aus einem Log-Eintrag."""
    analysis_data = entry.get("paper_analysis_data", [])
    if analysis_data:
        return "\n\n".join([
            f"Citation ID: [{p.get('citation_id', '?')}]\n"
            f"Findings: {p.get('findings', '')}\n"
            f"Methodology: {p.get('methodology', '')}\n"
            f"Limitations: {p.get('limitations', '')}"
            for p in analysis_data
        ])
    else:
        return "\n\n".join([
            p.get("ingested_content") or p.get("abstract", "")
            for p in entry.get("active_papers", [])
        ])


def evaluate_faithfulness(llm, context: str, review: str) -> float:
    """Extrahiert Kernaussagen und prüft sie direkt über Ollama auf Faithfulness."""
    # Schritt 1: Aussagen extrahieren
    prompt_extract = f"""Extract all individual factual statements made in the following literature review.
List each claim on a new line starting with '- '. Do NOT add introductory text.

[Literature Review]:
{review}
"""
    res_extract = llm.invoke(prompt_extract)
    text_extract = res_extract.content if hasattr(res_extract, 'content') else str(res_extract)

    # Aussagen aus Freitext filtern
    claims = [
        line.strip("- *").strip()
        for line in text_extract.split("\n")
        if line.strip().startswith(("-", "*")) or (len(line.strip()) > 10 and line.strip()[0].isdigit())
    ]
    if not claims:
        claims = [line.strip() for line in text_extract.split("\n") if len(line.strip()) > 15]

    if not claims:
        return 1.0

    print(f"   -> {len(claims)} Aussagen extrahiert. Prüfe Belege...")

    # Schritt 2: Aussagen gegen Kontext abgleichen
    claims_formatted = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(claims)])
    prompt_verify = f"""Compare the following claims against the provided context.

[Context]:
{context}

[Claims]:
{claims_formatted}

Task: For each claim, determine if it is supported by the context.
Format output strictly like:
Claim 1: YES
Claim 2: NO
"""
    res_verify = llm.invoke(prompt_verify)
    text_verify = res_verify.content if hasattr(res_verify, 'content') else str(res_verify)

    # Treffer zählen
    yes_count = len(re.findall(r'Claim \d+:?\s*YES', text_verify, re.IGNORECASE))
    total_claims = len(claims)

    score = round(yes_count / total_claims, 4) if total_claims > 0 else 1.0
    print(f"   -> Belegt: {yes_count}/{total_claims} | Score: {score}")
    return min(score, 1.0)


def main():
    print("=== Direct Faithfulness Evaluation (JSON Input) ===")

    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ Datei nicht gefunden: {INPUT_JSON_PATH}")
        return

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Gefundene Einträge in Datei: {len(entries)}")

    ollama_llm = ChatOllama(
        model=ModelFactory.get_model_name(),
        temperature=0.0,
        num_ctx=8192
    )

    results = []

    # Verarbeite jeden Eintrag aus der Datei
    for idx, entry in enumerate(entries):
        query = entry.get("user_query", "")
        print(f"\n--- Verarbeite Eintrag {idx + 1}/{len(entries)} ---")
        print(f"Frage: '{query[:60]}...'")

        context = extract_context(entry)
        review = entry.get("final_review", "")

        score = evaluate_faithfulness(ollama_llm, context, review)
        results.append({
            "index": idx,
            "query": query,
            "faithfulness": score
        })

    print("\n=== FAITHFULNESS ERGEBNISSE ===")
    for r in results:
        print(f"Eintrag {r['index'] + 1}: Faithfulness = {r['faithfulness']}")

    # Automatisch in CSV eintragen, wenn die CSV exakt gleich viele leere Felder hat
    if os.path.exists(OUTPUT_CSV_PATH):
        df = pd.read_csv(OUTPUT_CSV_PATH)
        nan_indices = df[df['faithfulness'].isna()].index.tolist()

        if len(nan_indices) == len(results):
            for i, res in enumerate(results):
                target_idx = nan_indices[i]
                df.loc[target_idx, 'faithfulness'] = res['faithfulness']

            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\n✓ Scores wurden direkt in '{OUTPUT_CSV_PATH}' eingetragen!")
        else:
            print(
                f"\n💡 Hinweis: Die CSV hat {len(nan_indices)} leere Feldern, aber es wurden {len(results)} Ergebnisse berechnet. Du kannst die Werte oben manuell eintragen.")


if __name__ == "__main__":
    main()