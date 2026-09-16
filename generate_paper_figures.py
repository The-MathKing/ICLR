import os

# Figure output directory. Override with the PAPER_DIR env var; defaults to ./paper
PAPER_DIR = os.environ.get("PAPER_DIR",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))
os.makedirs(PAPER_DIR, exist_ok=True)
"""
generate_paper_figures.py
Generates high-resolution publication-quality figures for the ICLR manuscript:
1. fig1_teaser_audit.png: Page 1 Hook Teaser Figure (The Illusion vs Demystification vs Audited Reality)
2. fig2_pipeline_scaffolding.png: Methodological Audit Pipeline Diagram
3. fig4_cross_model_heatmap.png: Multi-Model, Multi-Dataset Performance Heatmap
4. fig_appx_power_bootstrap.png: Cluster-Bootstrap Significance Distributions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import pandas as pd

# Set academic typography and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

def create_fig1_teaser():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)
    
    # ---------------- Panel 1: The Illusion ----------------
    ax = axes[0]
    methods = ['Length (Raw)', 'TOHA (MTop)', '0D Persistence', '1D Persistence\n(Raw Cycles)', '0D + 1D (Raw)']
    scores = [0.967, 0.597, 0.790, 0.693, 0.791]
    
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, scores, color=['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#2ecc71'], alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontweight='bold', fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Reported ROC-AUC (Naive CV)', fontweight='bold')
    ax.set_title('(a) The Illusion\n(Naive Cross-Validation & Confounded Length)', fontweight='bold', pad=10, fontsize=10)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.6, label='Random Chance (0.50)')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=8, fontweight='bold')
    
    # Annotation box
    ax.text(0.52, 0.15, 'Length Confound\n$r=0.614, p<10^{-10}$', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee8e7', edgecolor='#e74c3c', alpha=0.9),
            fontsize=8, fontweight='bold', color='#c0392b')

    # ---------------- Panel 2: The Demystification ----------------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(b) The Mathematical Demystification\n(0D Persistence = Attention MST)', fontweight='bold', pad=10, fontsize=10)
    
    # Box for Symmetrized Distance
    rect1 = patches.FancyBboxPatch((0.5, 6.5), 9.0, 2.5, boxstyle="round,pad=0.2", facecolor='#ebf5fb', edgecolor='#2980b9', linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(5.0, 8.1, r"Symmetric Attention Distance Matrix", ha='center', va='center', fontweight='bold', fontsize=9, color='#1b4f72')
    ax.text(5.0, 7.1, r"$D_{ij} = 1 - \max(A_{ij}, A_{ji}), \quad D_{ii} = 0$", ha='center', va='center', fontsize=9.5, color='#2c3e50')
    
    # Equivalence Arrow
    ax.annotate('', xy=(5.0, 4.4), xytext=(5.0, 6.3),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.4", color='#e74c3c', lw=2.5))
    ax.text(5.15, 5.35, r"$\mathbf{Isomorphism}$", color='#e74c3c', fontweight='bold', fontsize=8.5, ha='left')
    
    # Box for MST Equivalence
    rect2 = patches.FancyBboxPatch((0.5, 0.8), 9.0, 3.4, boxstyle="round,pad=0.2", facecolor='#eafaf1', edgecolor='#27ae60', linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(5.0, 3.4, r"Kruskal's Single-Linkage Minimum Spanning Tree", ha='center', va='center', fontweight='bold', fontsize=9, color='#145a32')
    ax.text(5.0, 2.4, r"$\sum_i (d_i - b_i) \equiv \sum_{e \in \text{MST}(D)} w(e)$", ha='center', va='center', fontsize=10, fontweight='bold', color='#1e8449')
    ax.text(5.0, 1.4, r"0D Persistence $\equiv$ Basic Attention Marginals ($p_{\text{adj}} = 1.0$)", ha='center', va='center', fontsize=8, color='#27ae60')

    # ---------------- Panel 3: The Audited Reality ----------------
    ax = axes[2]
    categories = ['Length\nBaseline', 'MST Weight\n(0D Proxy)', '0D Only\n(VR Bank)', '1D Normalized\n(Length-Free)', '0D + 1D Norm\n(Combined)']
    auc_full = [0.495, 0.732, 0.780, 0.613, 0.780]
    
    x = np.arange(len(categories))
    bars3 = ax.bar(x, auc_full, color=['#95a5a6', '#f39c12', '#2980b9', '#9b59b6', '#27ae60'], alpha=0.9, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, fontweight='bold')
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel('Audited ROC-AUC (Length-Matched)', fontweight='bold')
    ax.set_title('(c) The Audited Reality\n(Strict Grouped CV + Length-Matched)', fontweight='bold', pad=10, fontsize=10)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6)
    
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax.annotate(r'$\mathbf{\Delta = +0.000}$' + '\n' + r'($p_{\text{adj}} = 1.00$)', xy=(4, 0.785), xytext=(3.3, 0.83),
                arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5),
                fontsize=8, fontweight='bold', color='#c0392b')

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, 'fig1_teaser_audit.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig1_teaser_audit.png")

def create_fig2_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # 1. Forward Pass Box
    p1 = patches.FancyBboxPatch((0.2, 1.2), 2.2, 2.6, boxstyle="round,pad=0.2", facecolor='#f4f6f7', edgecolor='#34495e', lw=1.5)
    ax.add_patch(p1)
    ax.text(1.3, 3.3, "1. LLM Forward Pass", ha='center', fontweight='bold', fontsize=9.5, color='#2c3e50')
    ax.text(1.3, 2.6, "Teacher-Forced Input:\n$X = [x_1, \dots, x_N]$\nLayerwise Attn:\n$A^{(l)} \in \mathbb{R}^{N \\times N}$", ha='center', fontsize=8.5)
    
    # Arrow 1->2
    ax.annotate('', xy=(2.9, 2.5), xytext=(2.4, 2.5), arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2))
    
    # 2. Graph Construction Box
    p2 = patches.FancyBboxPatch((3.0, 1.2), 2.3, 2.6, boxstyle="round,pad=0.2", facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.5)
    ax.add_patch(p2)
    ax.text(4.15, 3.3, "2. Attention Graph", ha='center', fontweight='bold', fontsize=9.5, color='#1b4f72')
    ax.text(4.15, 2.6, "Symmetrization:\n$W^{(l)} = \max(A, A^T)$\nDistance Metric:\n$D_{ij}^{(l)} = 1 - W_{ij}^{(l)}$", ha='center', fontsize=8.5)
    
    # Arrow 2->3
    ax.annotate('', xy=(5.8, 2.5), xytext=(5.3, 2.5), arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2))
    
    # 3. Topological Filtration Box
    p3 = patches.FancyBboxPatch((5.9, 1.2), 2.3, 2.6, boxstyle="round,pad=0.2", facecolor='#fef9e7', edgecolor='#f39c12', lw=1.5)
    ax.add_patch(p3)
    ax.text(7.05, 3.3, "3. Feature Extraction", ha='center', fontweight='bold', fontsize=9.5, color='#7e5109')
    ax.text(7.05, 2.6, "0D: MST Edge Weights\n1D: VR Simplices ($H_1$)\nBaselines: Mean/Max Attn,\nSink Mass, MSP Prob", ha='center', fontsize=8.5)
    
    # Arrow 3->4
    ax.annotate('', xy=(8.7, 2.5), xytext=(8.2, 2.5), arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2))
    
    # 4. Rigorous Audit Filters
    p4 = patches.FancyBboxPatch((8.8, 1.2), 2.0, 2.6, boxstyle="round,pad=0.2", facecolor='#eafaf1', edgecolor='#27ae60', lw=1.5)
    ax.add_patch(p4)
    ax.text(9.8, 3.3, "4. Auditing Filters", ha='center', fontweight='bold', fontsize=9.5, color='#145a32')
    ax.text(9.8, 2.5, "• Grouped 10-Fold CV\n• Exact Length Match\n• Token Normalization\n• Cluster Bootstrap\n  (Holm-Bonferroni)", ha='center', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, 'fig2_pipeline_scaffolding.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig2_pipeline_scaffolding.png")

def create_fig4_heatmap():
    # Model x Feature performance matrix
    models = ['Qwen2.5-3B (HaluEval)', 'Qwen2.5-3B (TruthfulQA)', 'SmolLM-1.7B (TruthfulQA)', 'Phi-3-mini-3.8B (TruthfulQA)', 'Qwen2.5-1.5B (HaluEval Control)']
    features = ['Length Baseline', 'MSP Probability', 'TOHA (MTop)', 'MST 0D Proxy', '0D Only (VR)', '1D Raw (Confounded)', '1D Normalized', '0D + 1D Norm']

    data = np.array([
        [0.495, 0.706, 0.513, 0.732, 0.780, 0.665, 0.613, 0.780],
        [0.466, 0.527, 0.459, 0.712, 0.711, 0.554, 0.558, 0.711],
        [0.490, 0.514, 0.490, 0.577, 0.576, 0.522, 0.523, 0.578],
        [0.488, 0.520, 0.478, 0.633, 0.639, 0.488, 0.489, 0.640],
        [0.847, 0.500, 0.510, 0.559, 0.561, 0.540, 0.521, 0.562]
    ])

    fig, ax = plt.subplots(figsize=(9.5, 4.1), dpi=300)
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Audited ROC-AUC'},
                xticklabels=features, yticklabels=models, ax=ax, linewidths=1.0, linecolor='white', annot_kws={'size': 9, 'weight': 'bold'})
    
    ax.set_xticklabels(features, rotation=25, ha='right', fontweight='bold', fontsize=8.5)
    ax.set_yticklabels(models, fontweight='bold', fontsize=8.5)
    ax.set_title('Comprehensive Audited ROC-AUC Performance Matrix Across Architectures & Feature Banks', fontweight='bold', pad=12, fontsize=10.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, 'fig4_cross_model_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fig4_cross_model_heatmap.png")

def create_bootstrap_plot():
    raise RuntimeError(
        "SUPERSEDED: this function plotted np.random.normal(...) placeholder "
        "draws that were never connected to any real bootstrap output -- the "
        "panel titles ('Fail to Reject', 'Exact Equivalence') were fabricated "
        "text, not computed statistics. Use generate_fig_bootstrap_v2.py, which "
        "plots the real 10,000-resample arrays saved by master_pipeline.py to "
        "master_results/bootstrap_diffs_*.npy."
    )

if __name__ == '__main__':
    create_fig1_teaser()
    create_fig2_pipeline()
    create_fig4_heatmap()
    # create_bootstrap_plot()  # superseded; see generate_fig_bootstrap_v2.py
