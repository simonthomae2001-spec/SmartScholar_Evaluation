import os
import re
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "evaluation_results_multi_agent.csv"))
OUTPUT_CSV = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "evaluation_results_multi_agent_clean.csv"))

print(f"Lese Datei: {INPUT_CSV}")

with open(INPUT_CSV, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

# Sucht gezielt nach zwei aufeinanderfolgenden Fließkommazahlen: ,0.84623464323,0.999999999975
pattern = re.compile(r',\s*([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)')
score_pairs = pattern.findall(raw_text)

print(f"✓ Gefundene valide Score-Paare: {len(score_pairs)}")

questions_base = [
    "What are the primary architectural differences, performance trade-offs, and scaling limits between Transformer-based Large Language Models and State Space Models (e.g., Mamba) in natural language processing?",
    "What is the current scientific consensus on the pathological mechanisms of Tau protein aggregation versus Amyloid-beta accumulation in the progression of Alzheimer's disease?",
    "How do microstructural defects and grain boundary dynamics affect the ionic conductivity of solid-state halide electrolytes in lithium-metal batteries?",
    "What are the specific computational challenges and delay-compensation strategies in Model Predictive Control (MPC) when applied to real-time autonomous vehicle trajectory tracking under adverse weather conditions?",
    "How are Deep Learning generative models being integrated with microfluidic chip design to accelerate high-throughput screening in synthetic biology?",
    "What are the economic, ecological, and legal implications of implementing Remote Sensing and AI-driven satellite monitoring for international carbon credit verification in tropical forestry?",
]

# 6 Fragen * 3 Durchläufe (18 Einträge)
questions_full = [q for q in questions_base for _ in range(3)]
min_len = min(len(score_pairs), len(questions_full))

clean_data = [
    {
        "user_input": questions_full[i],
        "faithfulness": float(score_pairs[i][0]),
        "context_utilization": float(score_pairs[i][1])
    }
    for i in range(min_len)
]

df_clean = pd.DataFrame(clean_data)

print("\n--- Vorschau der geretteten Daten ---")
print(df_clean)

df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saubere CSV erfolgreich gespeichert unter: {OUTPUT_CSV}")