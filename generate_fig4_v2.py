"""Regenerates Figure 4 (cross-model heatmap) from the canonical master_auc_table.csv,
fixing stale/buggy numbers in the previous version (e.g. a mislabeled Qwen2.5-1.5B row)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("master_results/master_auc_table.csv")

rows = [
    ("Qwen2.5-3B (HaluEval)", "HaluEval (Qwen2.5-3B)"),
    ("Qwen2.5-1.5B (HaluEval)", "HaluEval (Qwen2.5-1.5B)"),
    ("Qwen2.5-3B (TruthfulQA)", "TruthfulQA (Qwen2.5-3B)"),
    ("SmolLM-1.7B (TruthfulQA)", "TruthfulQA (SmolLM-1.7B)"),
    ("Phi-3-mini-3.8B (TruthfulQA)", "TruthfulQA (Phi-3-mini-3.8B)"),
    ("Mistral-7B (TruthfulQA)", "TruthfulQA (Mistral-7B)"),
    ("TinyLlama-1.1B (TruthfulQA)", "TruthfulQA (TinyLlama-1.1B)"),
]
cols = [
    ("Length\nBaseline", "Length Baseline"),
    ("MST\nProxy", "MST Per-Layer Proxy"),
    ("0D Only\n(=MST)", "0D Only"),
    ("1D Raw\n(confounded)", "1D Only (Raw)"),
    ("1D Norm.\n(Z/N)", "1D Only (Normalized, Z/N)"),
    ("0D+1D\nNorm.", "0D + 1D (Normalized, Z/N)"),
]

mat = np.full((len(rows), len(cols)), np.nan)
for i, (_, bm) in enumerate(rows):
    sub = df[df["Benchmark"] == bm].set_index("Feature Set")
    for j, (_, bank) in enumerate(cols):
        if bank in sub.index:
            mat[i, j] = sub.loc[bank, "AUC_full"]

fig, ax = plt.subplots(figsize=(11, 4.0))
im = ax.imshow(mat, cmap="YlGnBu", vmin=0.45, vmax=0.95, aspect="auto")

ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c[0] for c in cols], fontsize=10)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, fontweight="bold")

for i in range(len(rows)):
    for j in range(len(cols)):
        v = mat[i, j]
        if not np.isnan(v):
            color = "white" if v > 0.72 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)

ax.set_title("Audited ROC-AUC Across Architectures and Feature Banks (Full Benchmark, Canonical Pipeline)",
              fontsize=12.5, fontweight="bold", pad=12)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("ROC-AUC", fontsize=10)

fig.tight_layout()
fig.savefig("paper/fig4_cross_model_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved paper/fig4_cross_model_heatmap.png")
