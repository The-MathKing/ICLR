"""
compute_mutual_information.py
==============================
Estimates conditional mutual information I(Y; H1* | H0, L) across all datasets
using the non-parametric Kraskov / k-NN mutual information estimator.
Outputs mutual_information_audit.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression

BENCHMARKS = [
    ("HaluEval (Qwen2.5-3B)", "phase3_results/train_features.csv"),
    ("HaluEval (Qwen2.5-1.5B)", "phase3_results/train_features_qwen1_5b.csv"),
    ("TruthfulQA (Qwen2.5-3B)", "phase10_results/qwen3b_truthfulqa_4stat.csv"),
    ("TruthfulQA (Phi-3-mini-3.8B)", "phase10_results/phi3_truthfulqa_4stat.csv"),
]

results = []

for name, path in BENCHMARKS:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if 'label' in df.columns:
        y = (df['label'] == 'hallucinated').astype(int).values
    elif 'target' in df.columns:
        y = df['target'].values
    else:
        continue

    L = df['seq_len'].values if 'seq_len' in df.columns else np.ones(len(df))

    h0_cols = [c for c in df.columns if 'h0' in c]
    h1_cols = [c for c in df.columns if 'h1' in c]

    if not h0_cols or not h1_cols:
        continue

    X_h0 = df[h0_cols].values
    X_h1 = df[h1_cols].values
    X_h1_norm = df[h1_cols].div(np.maximum(L, 1), axis=0).values

    mi_h0 = mutual_info_classif(X_h0, y, random_state=42)
    mi_h1_raw = mutual_info_classif(X_h1, y, random_state=42)
    mi_h1_norm = mutual_info_classif(X_h1_norm, y, random_state=42)

    # Residualize H1_norm on (H0, L)
    resids = []
    cov = np.column_stack([X_h0, L])
    for j in range(X_h1_norm.shape[1]):
        reg = LinearRegression().fit(cov, X_h1_norm[:, j])
        resids.append(X_h1_norm[:, j] - reg.predict(cov))
    X_h1_resid = np.column_stack(resids)

    mi_cond = mutual_info_classif(X_h1_resid, y, random_state=42)

    results.append({
        "Benchmark": name,
        "N": len(df),
        "Mean MI I(Y; H0) [nats]": round(float(np.mean(mi_h0)), 5),
        "Mean MI I(Y; H1_raw) [nats]": round(float(np.mean(mi_h1_raw)), 5),
        "Mean MI I(Y; H1_norm) [nats]": round(float(np.mean(mi_h1_norm)), 5),
        "Cond MI I(Y; H1* | H0, L) [nats]": round(float(np.mean(mi_cond)), 5),
        "Max Cond MI [nats]": round(float(np.max(mi_cond)), 5),
    })

res_df = pd.DataFrame(results)
res_df.to_csv("mutual_information_audit.csv", index=False)
print("✓ Saved mutual_information_audit.csv:")
print(res_df.to_string(index=False))
