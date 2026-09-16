"""
compute_equivalence_tests.py
============================
Computes formal statistical equivalence tests (Two One-Sided Tests - TOST)
and 95% cluster-bootstrap confidence intervals for ΔAUC across all audited benchmarks.

Formally tests H0: |ΔAUC| >= ε versus H1: |ΔAUC| < ε (Practical Equivalence)
at margin ε = 0.015 (1.5% AUC difference).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json

def compute_cluster_bootstrap_ci_and_tost(y_true, preds_A, preds_B, groups, epsilon=0.015, n_bootstraps=2000, random_state=42):
    """
    Computes 95% cluster-bootstrap confidence intervals for ΔAUC = AUC(B) - AUC(A)
    and evaluates TOST (Two One-Sided Tests) for practical equivalence within [-ε, +ε].
    """
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    auc_A_obs = roc_auc_score(y_true, preds_A)
    auc_B_obs = roc_auc_score(y_true, preds_B)
    delta_obs = auc_B_obs - auc_A_obs
    
    diffs = []
    for _ in range(n_bootstraps):
        boot_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_idx = []
        for g in boot_groups:
            boot_idx.extend(np.where(groups == g)[0])
            
        y_boot = y_true[boot_idx]
        if len(np.unique(y_boot)) < 2:
            continue
            
        auc_A_boot = roc_auc_score(y_boot, preds_A[boot_idx])
        auc_B_boot = roc_auc_score(y_boot, preds_B[boot_idx])
        diffs.append(auc_B_boot - auc_A_boot)
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    # Two One-Sided Tests (TOST):
    # Test 1: H01: Δ <= -ε vs H11: Δ > -ε
    # p1 = fraction of bootstrap samples with diff <= -ε
    p_lower = np.mean(diffs <= -epsilon)
    
    # Test 2: H02: Δ >= +ε vs H12: Δ < +ε
    # p2 = fraction of bootstrap samples with diff >= +ε
    p_upper = np.mean(diffs >= epsilon)
    
    # TOST p-value is max(p1, p2)
    tost_p = max(p_lower, p_upper)
    is_equivalent = bool(ci_lower > -epsilon and ci_upper < epsilon)
    
    return {
        "auc_A": float(auc_A_obs),
        "auc_B": float(auc_B_obs),
        "delta_auc": float(delta_obs),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "tost_p": float(tost_p),
        "is_equivalent": is_equivalent,
        "bootstrap_std": float(np.std(diffs))
    }

def get_oof_preds(X, y, groups):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[test_idx] = lr.predict_proba(X.iloc[test_idx])[:, 1]
    return oof_preds

def run_all_equivalence_tests():
    benchmarks = [
        {
            "name": "HaluEval (Qwen2.5-3B)",
            "file": "phase3_results/train_features.csv",
            "model": "Qwen2.5-3B"
        },
        {
            "name": "TruthfulQA (Qwen2.5-3B)",
            "file": "phase10_results/qwen3b_truthfulqa_4stat.csv",
            "model": "Qwen2.5-3B"
        },
        {
            "name": "TruthfulQA (SmolLM-1.7B)",
            "file": "phase7_results/truthfulqa_smollm_features.csv",
            "model": "SmolLM-1.7B"
        },
        {
            "name": "TruthfulQA (Phi-3-mini-3.8B)",
            "file": "phase10_results/phi3_truthfulqa_4stat.csv",
            "model": "Phi-3-mini-3.8B"
        }
    ]
    
    results_list = []
    print("=" * 95)
    print("FORMAL STATISTICAL EQUIVALENCE TESTING (TOST: Margin ε = ±0.015)")
    print("Testing Incremental Information of 1D Cycles: 0D vs (0D + 1D Norm)")
    print("=" * 95)
    
    for bm in benchmarks:
        df = pd.read_csv(bm["file"])
        y = (df['label'] == 'hallucinated').astype(int)
        groups = df['example_id'].values
        
        # Feature extraction
        X_0d = df[[c for c in df.columns if 'h0' in c]]
        X_1d = df[[c for c in df.columns if 'h1' in c]]
        seq_len = df['seq_len'] if 'seq_len' in df.columns else df['n_tokens']
        X_1d_norm = X_1d.div(seq_len, axis=0)
        X_comb_norm = pd.concat([X_0d, X_1d_norm], axis=1)
        
        preds_0d = get_oof_preds(X_0d, y, groups)
        preds_comb = get_oof_preds(X_comb_norm, y, groups)
        
        stats = compute_cluster_bootstrap_ci_and_tost(y.values, preds_0d, preds_comb, groups, epsilon=0.015, n_bootstraps=2000)
        
        row = {
            "Benchmark": bm["name"],
            "AUC (0D)": f"{stats['auc_A']:.3f}",
            "AUC (0D+1D Norm)": f"{stats['auc_B']:.3f}",
            "ΔAUC": f"{stats['delta_auc']:+.4f}",
            "95% CI for ΔAUC": f"[{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]",
            "TOST p-value": f"{stats['tost_p']:.4f}",
            "Equivalence (ε=0.015)": "CONFIRMED" if stats['is_equivalent'] else "REJECTED"
        }
        results_list.append(row)
        
        print(f"\n{bm['name']}:")
        print(f"  AUC(0D): {stats['auc_A']:.4f} | AUC(0D+1D Norm): {stats['auc_B']:.4f}")
        print(f"  ΔAUC: {stats['delta_auc']:+.4f} | 95% Cluster-Bootstrap CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]")
        print(f"  TOST Equivalence (ε=±0.015): p = {stats['tost_p']:.4f} -> {row['Equivalence (ε=0.015)']}")
        
    out_df = pd.DataFrame(results_list)
    out_df.to_csv("equivalence_tests_master.csv", index=False)
    print("\nMaster statistical equivalence table written to equivalence_tests_master.csv")
    return out_df

if __name__ == "__main__":
    run_all_equivalence_tests()
