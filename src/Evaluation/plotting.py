import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# CSVs laden (nur die relevanten Spalten)
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_PATH = os.path.normpath(os.path.join(current_dir, "..", "..", "data", "evaluation_results_single_agent.csv"))

df_single = pd.read_csv(INPUT_CSV_PATH)[['user_input', 'faithfulness', 'context_utilization']]
df_single['Architecture'] = 'Multi-Agent'

# Kurze Fragenamen für hübsche Plots erstellen
question_map = {
    q: f"Q{i+1}" for i, q in enumerate(df_single['user_input'].unique())
}
df_single['Question_ID'] = df_single['user_input'].map(question_map)

# Melt für Seaborn Facet / Categorical Plot
df_melted = df_single.melt(
    id_vars=['Question_ID', 'Architecture'],
    value_vars=['faithfulness', 'context_utilization'],
    var_name='Metric', value_name='Score'
)

# Plot erstellen
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df_melted,
    x='Question_ID',
    y='Score',
    hue='Metric',
    capsize=0.1,
    err_kws={'linewidth': 1.5}
)

plt.title("SmartScholar Evaluation Scores per Question (Single-Agent)", fontsize=18)
plt.ylim(0, 1.05)
plt.ylabel("RAGAS Score", fontsize=16)
plt.xlabel("Test Questions (Q1-Q6)", fontsize=16)
plt.legend(title="Metric")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Grafik als PNG für deine Arbeit speichern
plt.savefig("evaluation_single_agent_plot.png", dpi=300)
plt.show()