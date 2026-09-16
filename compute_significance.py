import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def get_oof_preds(X, y, groups, n_splits=10, random_state=42):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
    return oof_preds

def compute_fast_cluster_bootstrap(y, preds_A, preds_B, groups, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    group_to_rows = [np.where(group_indices == g)[0] for g in range(n_groups)]
    
    auc_A = roc_auc_score(y, preds_A)
    auc_B = roc_auc_score(y, preds_B)
    obs_diff = auc_A - auc_B
    
    diffs = []
    extreme = 0
    valid = 0
    
    for _ in range(n_boot):
        sample_g = rng.choice(n_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_rows[g] for g in sample_g])
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        dA = roc_auc_score(y_b, preds_A[idx])
        dB = roc_auc_score(y_b, preds_B[idx])
        diff = dA - dB
        diffs.append(diff)
        if abs(diff - obs_diff) >= abs(obs_diff):
            extreme += 1
        valid += 1
        
    p_val = (extreme + 1) / (valid + 1)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return obs_diff, ci_low, ci_high, p_val

def main():
    # HaluEval Qwen3B
    df_h = pd.read_csv("phase3_results/train_features.csv")
    y_h = (df_h['label'] == 'hallucinated').astype(int).values
    g_h = df_h['example_id'].values
    
    X_0d_h = df_h[[c for c in df_h.columns if 'h0' in c]]
    X_1d_raw_h = df_h[[c for c in df_h.columns if 'h1' in c]]
    X_1d_norm_h = X_1d_raw_h.div(df_h['seq_len'], axis=0)
    X_comb_h = pd.concat([X_0d_h, X_1d_norm_h], axis=1)
    h0_tot_h = [c for c in df_h.columns if 'h0_total_persistence' in c]
    X_toha_h = df_h[h0_tot_h].sum(axis=1).to_frame(name='toha')
    X_mst_h = df_h[h0_tot_h]
    
    p_0d_h = get_oof_preds(X_0d_h, pd.Series(y_h), pd.Series(g_h))
    p_comb_h = get_oof_preds(X_comb_h, pd.Series(y_h), pd.Series(g_h))
    p_toha_h = get_oof_preds(X_toha_h, pd.Series(y_h), pd.Series(g_h))
    p_mst_h = get_oof_preds(X_mst_h, pd.Series(y_h), pd.Series(g_h))
    
    # SmolLM TruthfulQA
    df_sm = pd.read_csv("phase7_results/truthfulqa_smollm_features.csv")
    y_sm = (df_sm['label'] == 'hallucinated').astype(int).values
    g_sm = df_sm['example_id'].values
    
    X_0d_sm = df_sm[[c for c in df_sm.columns if 'h0' in c]]
    X_1d_raw_sm = df_sm[[c for c in df_sm.columns if 'h1' in c]]
    X_1d_norm_sm = X_1d_raw_sm.div(df_sm['seq_len'], axis=0)
    X_comb_sm = pd.concat([X_0d_sm, X_1d_norm_sm], axis=1)
    h0_tot_sm = [c for c in df_sm.columns if 'h0_total_persistence' in c]
    X_toha_sm = df_sm[h0_tot_sm].sum(axis=1).to_frame(name='toha')
    X_mst_sm = df_sm[h0_tot_sm]
    X_msp_sm = df_sm[['msp_score']].fillna(0.5)
    
    p_0d_sm = get_oof_preds(X_0d_sm, pd.Series(y_sm), pd.Series(g_sm))
    p_comb_sm = get_oof_preds(X_comb_sm, pd.Series(y_sm), pd.Series(g_sm))
    p_toha_sm = get_oof_preds(X_toha_sm, pd.Series(y_sm), pd.Series(g_sm))
    p_mst_sm = get_oof_preds(X_mst_sm, pd.Series(y_sm), pd.Series(g_sm))
    p_msp_sm = get_oof_preds(X_msp_sm, pd.Series(y_sm), pd.Series(g_sm))
    
    comparisons = [
        ("HaluEval: 0D Only vs TOHA (MTop-Div)", y_h, p_0d_h, p_toha_h, g_h),
        ("HaluEval: 0D Only vs (0D + 1D Norm)", y_h, p_0d_h, p_comb_h, g_h),
        ("HaluEval: 0D Only vs Per-Layer MST", y_h, p_0d_h, p_mst_h, g_h),
        ("TruthfulQA (SmolLM): 0D Only vs TOHA", y_sm, p_0d_sm, p_toha_sm, g_sm),
        ("TruthfulQA (SmolLM): 0D Only vs (0D + 1D Norm)", y_sm, p_0d_sm, p_comb_sm, g_sm),
        ("TruthfulQA (SmolLM): TOHA vs MSP Baseline", y_sm, p_toha_sm, p_msp_sm, g_sm),
        ("TruthfulQA (SmolLM): 0D Only vs Per-Layer MST", y_sm, p_0d_sm, p_mst_sm, g_sm),
    ]
    
    results = []
    raw_ps = []
    for label, y_vec, pA, pB, g_vec in comparisons:
        diff, c_l, c_h, p = compute_fast_cluster_bootstrap(y_vec, pA, pB, g_vec, n_boot=2000, seed=42)
        raw_ps.append(p)
        results.append({
            "Comparison": label,
            "AUC Diff (A - B)": diff,
            "95% CI": f"[{c_l:+.4f}, {c_h:+.4f}]",
            "Raw p": p
        })
        
    # Holm-Bonferroni correction
    m = len(raw_ps)
    sort_idx = np.argsort(raw_ps)
    holm_p = np.zeros(m)
    for rank, idx in enumerate(sort_idx):
        adj = raw_ps[idx] * (m - rank)
        holm_p[idx] = min(max(adj, 0.0), 1.0)
    for i in range(1, m):
        if holm_p[sort_idx[i]] < holm_p[sort_idx[i-1]]:
            holm_p[sort_idx[i]] = holm_p[sort_idx[i-1]]
            
    for i, r in enumerate(results):
        r["Holm p"] = holm_p[i]
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv("significance_tests_results.csv", index=False)

if __name__ == "__main__":
    main()
