import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def bootstrap_auc_test(y_true, preds_A, preds_B, groups, n_bootstraps=10000, random_state=42):
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    auc_diff_obs = roc_auc_score(y_true, preds_A) - roc_auc_score(y_true, preds_B)
    
    count_extreme = 0
    valid_boots = 0
    for _ in range(n_bootstraps):
        boot_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_idx = []
        for g in boot_groups:
            boot_idx.extend(np.where(groups == g)[0])
        
        y_boot = y_true[boot_idx]
        
        if len(np.unique(y_boot)) < 2:
            continue
            
        preds_A_boot = preds_A[boot_idx]
        preds_B_boot = preds_B[boot_idx]
        
        auc_A_boot = roc_auc_score(y_boot, preds_A_boot)
        auc_B_boot = roc_auc_score(y_boot, preds_B_boot)
        diff_boot = auc_A_boot - auc_B_boot
        
        if abs(diff_boot - auc_diff_obs) >= abs(auc_diff_obs):
            count_extreme += 1
        valid_boots += 1
            
    p_value = (count_extreme + 1) / (valid_boots + 1)
    return p_value

def get_oof_preds(X, y, groups):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[test_idx] = lr.predict_proba(X.iloc[test_idx])[:, 1]
    return oof_preds

def run_tests():
    print("=== HaluEval (Qwen2.5-3B) ===")
    df_h = pd.read_csv("phase3_results/train_features.csv")
    y_h = (df_h['label'] == 'hallucinated').astype(int)
    groups_h = df_h['example_id'].values
    y_h_np = y_h.values
    
    X_0d_h = df_h[[c for c in df_h.columns if 'h0' in c]]
    X_1d_h = df_h[[c for c in df_h.columns if 'h1' in c]]
    X_1d_norm_h = X_1d_h.div(df_h['seq_len'], axis=0)
    X_comb_norm_h = pd.concat([X_0d_h, X_1d_norm_h], axis=1)
    
    h0_cols_h = [c for c in df_h.columns if 'h0_total_persistence' in c]
    X_toha_h = df_h[h0_cols_h].sum(axis=1).to_frame()
    
    preds_0d_h = get_oof_preds(X_0d_h, y_h, groups_h)
    preds_comb_h = get_oof_preds(X_comb_norm_h, y_h, groups_h)
    preds_toha_h = get_oof_preds(X_toha_h, y_h, groups_h)
    
    p_0d_vs_toha_h = bootstrap_auc_test(y_h_np, preds_0d_h, preds_toha_h, groups_h)
    p_0d_vs_comb_h = bootstrap_auc_test(y_h_np, preds_0d_h, preds_comb_h, groups_h)
    
    print(f"HaluEval 0D vs TOHA: p = {p_0d_vs_toha_h:.5f}")
    print(f"HaluEval 0D vs (0D+1DNorm): p = {p_0d_vs_comb_h:.5f}")

    print("\n=== TruthfulQA (SmolLM-1.7B) ===")
    df_t = pd.read_csv("phase7_results/truthfulqa_smollm_features.csv")
    y_t = (df_t['label'] == 'hallucinated').astype(int)
    groups_t = df_t['example_id'].values
    y_t_np = y_t.values
    
    X_0d_t = df_t[[c for c in df_t.columns if 'h0' in c]]
    X_1d_t = df_t[[c for c in df_t.columns if 'h1' in c]]
    X_1d_norm_t = X_1d_t.div(df_t['seq_len'], axis=0)
    X_comb_norm_t = pd.concat([X_0d_t, X_1d_norm_t], axis=1)
    
    h0_cols_t = [c for c in df_t.columns if 'h0_total_persistence' in c]
    X_toha_t = df_t[h0_cols_t].sum(axis=1).to_frame()
    X_msp_t = df_t[['msp_score']]
    
    preds_0d_t = get_oof_preds(X_0d_t, y_t, groups_t)
    preds_comb_t = get_oof_preds(X_comb_norm_t, y_t, groups_t)
    preds_toha_t = get_oof_preds(X_toha_t, y_t, groups_t)
    preds_msp_t = get_oof_preds(X_msp_t, y_t, groups_t)
    
    p_0d_vs_toha_t = bootstrap_auc_test(y_t_np, preds_0d_t, preds_toha_t, groups_t)
    p_0d_vs_comb_t = bootstrap_auc_test(y_t_np, preds_0d_t, preds_comb_t, groups_t)
    p_toha_vs_msp_t = bootstrap_auc_test(y_t_np, preds_toha_t, preds_msp_t, groups_t)
    p_0d_vs_msp_t = bootstrap_auc_test(y_t_np, preds_0d_t, preds_msp_t, groups_t)
    
    print(f"TruthfulQA 0D vs TOHA: p = {p_0d_vs_toha_t:.5f}")
    print(f"TruthfulQA 0D vs (0D+1DNorm): p = {p_0d_vs_comb_t:.5f}")
    print(f"TruthfulQA TOHA vs MSP: p = {p_toha_vs_msp_t:.5f}")
    print(f"TruthfulQA 0D vs MSP: p = {p_0d_vs_msp_t:.5f}")

if __name__ == "__main__":
    run_tests()
