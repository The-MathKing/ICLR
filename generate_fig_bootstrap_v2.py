"""Regenerates the appendix cluster-bootstrap distribution figure from REAL
bootstrap resamples saved by master_pipeline.py (master_results/bootstrap_diffs_*.npy),
replacing a previous version that plotted np.random.normal(...) placeholder draws
never connected to any real data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'

panels = [
    ("HaluEval_Qwen2.5-3B", "HaluEval (Qwen2.5-3B):\n0D vs. 0D+1D Norm ($+0.0156$, REJECTED @ $\\epsilon{=}0.015$)", "#2980b9"),
    ("TruthfulQA_SmolLM-1.7B", "TruthfulQA (SmolLM-1.7B):\n0D vs. 0D+1D Norm ($+0.0103$, CONFIRMED only @ $\\epsilon{=}0.025$)", "#27ae60"),
    ("TruthfulQA_Mistral-7B", "TruthfulQA (Mistral-7B):\n0D vs. 0D+1D Norm ($-0.0002$, CONFIRMED @ $\\epsilon{=}0.010$)", "#8e44ad"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=300)
for ax, (fname, title, color) in zip(axes, panels):
    diffs = np.load(f"master_results/bootstrap_diffs_{fname}.npy")
    sns.histplot(diffs, kde=True, color=color, ax=ax, bins=40)
    ax.axvline(0, color='red', linestyle='--', lw=1.5, label='$\\Delta=0$')
    ax.axvline(diffs.mean(), color='black', linestyle='-', lw=1.2, label=f'observed $\\Delta={diffs.mean():+.4f}$')
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_xlabel('$\\Delta$ ROC-AUC (10,000 group-cluster resamples)', fontweight='bold', fontsize=8.5)
    ax.legend(fontsize=7)

fig.suptitle("Real Cluster-Bootstrap $\\Delta$AUC Distributions (0D vs. 0D+1D, from master_pipeline.py)", fontweight='bold', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("paper/fig_appx_power_bootstrap.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved paper/fig_appx_power_bootstrap.png from real bootstrap data")
