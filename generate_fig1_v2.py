"""
generate_fig1_v2.py
====================
Rebuilds Figure 1 (teaser) with an honest, internally-consistent story:
panel (a) shows what actually happens across three evaluation corrections
(naive same-fold CV -> grouped CV -> grouped+length-matched CV) for each
feature bank on HaluEval QA (Qwen2.5-3B); panel (b) keeps the MST<->0D
equivalence diagram (Theorem 1), redrawn to fix text clipping.

Numbers are taken directly from the canonical scaled pipeline
(master_pipeline.py) -- no separate/inconsistent analysis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(15, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.32)

# ---------------------------------------------------------------------------
# Panel (a): honest waterfall across evaluation corrections
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0])

banks = ["Length\n(raw seq. len.)", "1D Persistence\n(raw)", "0D Persistence\n(MST-equiv.)", "0D + 1D\n(normalized)"]
naive =   [0.584, 0.684, 0.816, 0.829]
grouped = [0.585, 0.695, 0.822, 0.837]
matched = [0.495, 0.668, 0.819, 0.831]

x = np.arange(len(banks))
width = 0.25

colors = {"naive": "#c94c4c", "grouped": "#4c78a8", "matched": "#54a24b"}

b1 = ax.bar(x - width, naive, width, label="Naive CV (ungrouped)", color=colors["naive"])
b2 = ax.bar(x, grouped, width, label="Grouped CV", color=colors["grouped"])
b3 = ax.bar(x + width, matched, width, label="Grouped CV + Length-Matched", color=colors["matched"])

for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.set_ylabel("ROC-AUC")
ax.set_xticks(x)
ax.set_xticklabels(banks)
ax.set_ylim(0.4, 0.92)
ax.set_title("(a) What Actually Collapses Under Correction\n(HaluEval QA, Qwen2.5-3B)", fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=8.5, frameon=False)

ax.annotate("Length-matching kills\nthe length confound (only)", xy=(0.25, 0.51), xytext=(0.75, 0.60),
            fontsize=8.5, ha="left", color=colors["matched"],
            arrowprops=dict(arrowstyle="->", color=colors["matched"], lw=1.2))
ax.annotate("0D is robust to\nall three corrections", xy=(2.25, 0.822), xytext=(1.55, 0.865),
            fontsize=8.5, ha="left", color=colors["grouped"],
            arrowprops=dict(arrowstyle="->", color=colors["grouped"], lw=1.2))

# ---------------------------------------------------------------------------
# Panel (b): MST <-> 0D equivalence (Theorem 1), redrawn
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

box1 = dict(boxstyle="round,pad=0.6", facecolor="#eaf1fb", edgecolor="#4c78a8", linewidth=1.5)
box2 = dict(boxstyle="round,pad=0.6", facecolor="#eaf7ea", edgecolor="#54a24b", linewidth=1.5)

ax2.text(0.5, 0.86, "Symmetric Attention\nDistance Matrix", ha="center", va="center",
         fontsize=11, fontweight="bold", bbox=box1)
ax2.text(0.5, 0.86 - 0.14, r"$D_{ij} = 1 - \max(A_{ij}, A_{ji})$", ha="center", va="center", fontsize=10)

ax2.annotate("", xy=(0.5, 0.42), xytext=(0.5, 0.63),
             arrowprops=dict(arrowstyle="-|>", color="#c94c4c", lw=2.2))
ax2.text(0.56, 0.525, "Theorem 1\n(exact identity)", fontsize=9.5, color="#c94c4c", va="center")

ax2.text(0.5, 0.30, "Kruskal's Single-Linkage\nMinimum Spanning Tree", ha="center", va="center",
         fontsize=11, fontweight="bold", bbox=box2)
ax2.text(0.5, 0.30 - 0.13, r"$\sum_i (d_i-b_i) \equiv \sum_{e\in \mathrm{MST}} w(e)$", ha="center", va="center", fontsize=10)

ax2.text(0.5, 0.01, r"verified exact (0 diff.) on 52,288 real attention" "\n" r"graphs (Mistral-7B); skips 1D homology, $>\!1{,}200\times$" "\n" r"cheaper than full 0D+1D Ripser (not vs. 0D alone)",
         ha="center", va="bottom", fontsize=8.5, style="italic", color="#333333")
ax2.set_title("(b) 0D Persistence Is Exactly MST", fontsize=12, fontweight="bold")

fig.suptitle("Attention TDA: An Exact, Cheaper Replacement -- and an Honest Audit of the Rest",
             fontsize=13, fontweight="bold", y=1.02)

fig.savefig("paper/fig1_teaser_audit.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved paper/fig1_teaser_audit.png")
