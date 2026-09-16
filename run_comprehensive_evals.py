import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def get_length_matched_indices(df, length_col='seq_len', label_col='label', bin_size=10, random_state=42):
    rng = np.random.RandomState(random_state)
    df_temp = df.copy().reset_index(drop=True)
    df_temp['length_bin'] = (df_temp[length_col] // bin_size) * bin_size
    
    matched_indices = []
    for bin_val in sorted(df_temp['length_bin'].unique()):
        bin_df = df_temp[df_temp['length_bin'] == bin_val]
        grounded_idx = bin_df[bin_df[label_col].isin(['grounded', 0, '0', 'right_answer'])].index.tolist()
        hallu_idx = bin_df[bin_df[label_col].isin(['hallucinated', 1, '1', 'hallucinated_answer'])].index.tolist()
        
        min_count = min(len(grounded_idx), len(hallu_idx))
        if min_count > 0:
            rng.shuffle(grounded_idx)
            rng.shuffle(hallu_idx)
            matched_indices.extend(grounded_idx[:min_count])
            matched_indices.extend(hallu_idx[:min_count])
            
    return np.array(matched_indices)

def evaluate_cv(X, y, groups, n_splits=10, model_type="lr", random_state=42):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    oof_preds = np.zeros(len(y))
    
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        if model_type == "lr":
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == "xgb":
            clf = XGBClassifier(eval_metric='logloss', random_state=random_state, n_estimators=50, max_depth=3, n_jobs=1)
        
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        oof_preds[test_idx] = probs
        aucs.append(roc_auc_score(y_te, probs))
        
    return np.mean(aucs), np.std(aucs), oof_preds

def fast_bootstrap_diff(y_true, preds_A, preds_B, groups, n_bootstraps=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    
    # Pre-map group to row indices
    group_to_rows = [np.where(group_indices == g)[0] for g in range(n_groups)]
    
    auc_A_obs = roc_auc_score(y_true, preds_A)
    auc_B_obs = roc_auc_score(y_true, preds_B)
    obs_diff = auc_A_obs - auc_B_obs
    
    boot_diffs = []
    count_extreme = 0
    valid_boots = 0
    
    for _ in range(n_bootstraps):
        sampled_g = rng.choice(n_groups, size=n_groups, replace=True)
        b_idx = np.concatenate([group_to_rows[g] for g in sampled_g])
        
        y_b = y_true[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
            
        p_A_b = preds_A[b_idx]
        p_B_b = preds_B[b_idx]
        
        diff_b = roc_auc_score(y_b, p_A_b) - roc_auc_score(y_b, p_B_b)
        boot_diffs.append(diff_b)
        
        if abs(diff_b - obs_diff) >= abs(obs_diff):
            count_extreme += 1
        valid_boots += 1
        
    p_val = (count_extreme + 1) / (valid_boots + 1)
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)
    
    return obs_diff, ci_low, ci_high, p_val

def run_all():
    print("=========================================================================", flush=True)
    print("COMPREHENSIVE MULTI-MODEL MULTI-DATASET EVALUATION PIPELINE", flush=True)
    print("=========================================================================\n", flush=True)
    
    # 1. HaluEval QA (Qwen2.5-3B)
    print(">>> 1. Loading HaluEval (Qwen2.5-3B, N=4000 total / 2000 pairs)", flush=True)
    df_h = pd.read_csv("phase3_results/train_features.csv")
    df_msp_h = pd.read_csv("phase4_msp_features.csv")
    df_h = pd.merge(df_h, df_msp_h[['example_id', 'label', 'mean_prob', 'min_prob']], on=['example_id', 'label'], how='left')
    df_h['log_seq_len'] = np.log(df_h['seq_len'] + 1)
    
    y_h = (df_h['label'] == 'hallucinated').astype(int)
    groups_h = df_h['example_id']
    
    matched_idx_h = get_length_matched_indices(df_h, length_col='seq_len', label_col='label', bin_size=10, random_state=42)
    df_h_matched = df_h.iloc[matched_idx_h].reset_index(drop=True)
    y_h_m = (df_h_matched['label'] == 'hallucinated').astype(int)
    groups_h_m = df_h_matched['example_id']
    print(f"HaluEval Full: N={len(df_h)}, Length-Matched: N={len(df_h_matched)}", flush=True)
    
    # 2. TruthfulQA (Qwen2.5-3B)
    print("\n>>> 2. Loading TruthfulQA (Qwen2.5-3B, N=300)", flush=True)
    df_q3 = pd.read_csv("phase10_results/qwen3b_truthfulqa_4stat.csv")
    df_q3['log_seq_len'] = np.log(df_q3['seq_len'] + 1)
    y_q3 = (df_q3['label'] == 'hallucinated').astype(int)
    groups_q3 = df_q3['example_id']
    
    matched_idx_q3 = get_length_matched_indices(df_q3, length_col='seq_len', label_col='label', bin_size=5, random_state=42)
    df_q3_matched = df_q3.iloc[matched_idx_q3].reset_index(drop=True)
    y_q3_m = (df_q3_matched['label'] == 'hallucinated').astype(int)
    groups_q3_m = df_q3_matched['example_id']
    print(f"TruthfulQA Qwen3B Full: N={len(df_q3)}, Length-Matched: N={len(df_q3_matched)}", flush=True)
    
    # 3. TruthfulQA (SmolLM-1.7B)
    print("\n>>> 3. Loading TruthfulQA (SmolLM-1.7B, N=1000)", flush=True)
    df_sm = pd.read_csv("phase7_results/truthfulqa_smollm_features.csv")
    df_sm['log_seq_len'] = np.log(df_sm['seq_len'] + 1)
    y_sm = (df_sm['label'] == 'hallucinated').astype(int)
    groups_sm = df_sm['example_id']
    
    matched_idx_sm = get_length_matched_indices(df_sm, length_col='seq_len', label_col='label', bin_size=5, random_state=42)
    df_sm_matched = df_sm.iloc[matched_idx_sm].reset_index(drop=True)
    y_sm_m = (df_sm_matched['label'] == 'hallucinated').astype(int)
    groups_sm_m = df_sm_matched['example_id']
    # 5. TruthfulQA (Phi-3-3.8B)
    print("\n>>> 5. Loading TruthfulQA (Phi-3-3.8B, N=100)", flush=True)
    df_phi3 = pd.read_csv("phase10_results/phi3_truthfulqa_4stat.csv")
    df_phi3['log_seq_len'] = np.log(df_phi3['seq_len'] + 1)
    y_phi3 = (df_phi3['label'] == 'hallucinated').astype(int)
    groups_phi3 = df_phi3['example_id']
    
    matched_idx_phi3 = get_length_matched_indices(df_phi3, length_col='seq_len', label_col='label', bin_size=5, random_state=42)
    df_phi3_matched = df_phi3.iloc[matched_idx_phi3].reset_index(drop=True)
    y_phi3_m = (df_phi3_matched['label'] == 'hallucinated').astype(int)
    groups_phi3_m = df_phi3_matched['example_id']
    print(f"TruthfulQA Phi-3-3.8B Full: N={len(df_phi3)}, Length-Matched: N={len(df_phi3_matched)}", flush=True)

    
    datasets = [
        ("HaluEval (Qwen2.5-3B)", df_h, y_h, groups_h, df_h_matched, y_h_m, groups_h_m),
        ("TruthfulQA (Qwen2.5-3B)", df_q3, y_q3, groups_q3, df_q3_matched, y_q3_m, groups_q3_m),
        ("TruthfulQA (SmolLM-1.7B)", df_sm, y_sm, groups_sm, df_sm_matched, y_sm_m, groups_sm_m),
        ("TruthfulQA (Phi-3-3.8B)", df_phi3, y_phi3, groups_phi3, df_phi3_matched, y_phi3_m, groups_phi3_m),
    ]
    
    results_summary = []
    oof_predictions_dict = {}
    
    for name, df_full, y_full, g_full, df_match, y_match, g_match in datasets:
        print(f"\n=======================================================", flush=True)
        print(f"EVALUATING DATASET: {name}", flush=True)
        print(f"=======================================================", flush=True)
        
        def get_feature_dict(df):
            X_len = df[['seq_len', 'log_seq_len']]
            X_0d = df[[c for c in df.columns if 'h0' in c]]
            X_1d_raw = df[[c for c in df.columns if 'h1' in c]]
            X_1d_norm = X_1d_raw.div(df['seq_len'], axis=0)
            X_comb_norm = pd.concat([X_0d, X_1d_norm], axis=1)
            
            if 'mtop_div' in df.columns or any('mtop_div' in c for c in df.columns):
                # TOHA/MTop-Div is defined as the multi-layer topological divergence summed
                # across transformer depth (Section 2.2), not a raw per-layer feature vector.
                # Summing here (rather than keeping the per-layer columns) matches the single
                # summed-scalar definition used for HaluEval and TruthfulQA/SmolLM.
                mtop_cols = [c for c in df.columns if 'mtop_div' in c]
                X_toha = df[mtop_cols].sum(axis=1).to_frame(name='toha_mtop')
            else:
                h0_cols = [c for c in df.columns if 'h0_total_persistence' in c]
                X_toha = df[h0_cols].sum(axis=1).to_frame(name='toha_mtop')
                
            if 'msp_score' in df.columns:
                X_msp = df[['msp_score']].fillna(0.5)
            elif 'mean_prob' in df.columns:
                X_msp = df[['mean_prob', 'min_prob']].fillna(0.5)
            else:
                X_msp = pd.DataFrame(np.zeros((len(df), 1)))
                
            h0_tot = [c for c in df.columns if 'h0_total_persistence' in c]
            X_mst_per_layer = df[h0_tot]
            
            return {
                "Length Baseline": (X_len, X_len.shape[1]),
                "MSP Baseline": (X_msp, X_msp.shape[1]),
                "TOHA (MTop-Div)": (X_toha, X_toha.shape[1]),
                "MST Per-Layer Proxy": (X_mst_per_layer, X_mst_per_layer.shape[1]),
                "0D Only": (X_0d, X_0d.shape[1]),
                "1D Only (Raw)": (X_1d_raw, X_1d_raw.shape[1]),
                "1D Only (Norm)": (X_1d_norm, X_1d_norm.shape[1]),
                "0D + 1D (Norm)": (X_comb_norm, X_comb_norm.shape[1]),
            }
            
        feats_full = get_feature_dict(df_full)
        feats_match = get_feature_dict(df_match)
        
        for feat_name in feats_full:
            X_f, dim_f = feats_full[feat_name]
            X_m, dim_m = feats_match[feat_name]
            
            m_f_lr, s_f_lr, oof_f_lr = evaluate_cv(X_f, y_full, g_full, n_splits=10, model_type="lr")
            oof_predictions_dict[f"{name} | {feat_name} | Full"] = (y_full.values, oof_f_lr, g_full.values)
            
            n_spl = 10 if len(df_match) >= 500 else 5
            m_m_lr, s_m_lr, oof_m_lr = evaluate_cv(X_m, y_match, g_match, n_splits=n_spl, model_type="lr")
            oof_predictions_dict[f"{name} | {feat_name} | Matched"] = (y_match.values, oof_m_lr, g_match.values)
            
            m_f_xgb, s_f_xgb = None, None
            m_m_xgb, s_m_xgb = None, None
            if feat_name in ["0D Only", "1D Only (Raw)", "1D Only (Norm)", "0D + 1D (Norm)"]:
                m_f_xgb, s_f_xgb, _ = evaluate_cv(X_f, y_full, g_full, n_splits=10, model_type="xgb")
                m_m_xgb, s_m_xgb, _ = evaluate_cv(X_m, y_match, g_match, n_splits=n_spl, model_type="xgb")
                
            results_summary.append({
                "Dataset": name,
                "Feature Set": feat_name,
                "Dim (d)": dim_f,
                "Full LR AUC": f"{m_f_lr:.3f} ± {s_f_lr:.3f}",
                "Full XGB AUC": f"{m_f_xgb:.3f} ± {s_f_xgb:.3f}" if m_f_xgb is not None else "-",
                "Matched LR AUC": f"{m_m_lr:.3f} ± {s_m_lr:.3f}",
                "Matched XGB AUC": f"{m_m_xgb:.3f} ± {s_m_xgb:.3f}" if m_m_xgb is not None else "-"
            })
            
            print(f"{feat_name:22s} (d={dim_f:3d}) | Full LR: {m_f_lr:.3f}±{s_f_lr:.3f} | Matched LR: {m_m_lr:.3f}±{s_m_lr:.3f}", flush=True)
            
    summary_df = pd.DataFrame(results_summary)
    print("\n" + "="*80, flush=True)
    print("CONSOLIDATED RESULTS TABLE:", flush=True)
    print("="*80, flush=True)
    print(summary_df.to_string(index=False), flush=True)
    summary_df.to_csv("all_results_consolidated.csv", index=False)
    
    # 5. Significance Tests
    print("\n=======================================================", flush=True)
    print("STATISTICAL SIGNIFICANCE TESTS (FAST CLUSTER BOOTSTRAP, 2000 RESAMPLES)", flush=True)
    print("=======================================================", flush=True)
    
    tests_to_run = [
        ("HaluEval 0D vs TOHA (Full)", "HaluEval (Qwen2.5-3B) | 0D Only | Full", "HaluEval (Qwen2.5-3B) | TOHA (MTop-Div) | Full"),
        ("HaluEval 0D vs (0D+1D) (Full)", "HaluEval (Qwen2.5-3B) | 0D Only | Full", "HaluEval (Qwen2.5-3B) | 0D + 1D (Norm) | Full"),
        ("HaluEval 0D vs Length Baseline (Full)", "HaluEval (Qwen2.5-3B) | 0D Only | Full", "HaluEval (Qwen2.5-3B) | Length Baseline | Full"),
        ("HaluEval 0D vs Length Baseline (Matched)", "HaluEval (Qwen2.5-3B) | 0D Only | Matched", "HaluEval (Qwen2.5-3B) | Length Baseline | Matched"),
        ("HaluEval 0D vs MST Proxy (Matched)", "HaluEval (Qwen2.5-3B) | 0D Only | Matched", "HaluEval (Qwen2.5-3B) | MST Per-Layer Proxy | Matched"),
        ("Qwen3B-TruthfulQA 0D vs TOHA (Full)", "TruthfulQA (Qwen2.5-3B) | 0D Only | Full", "TruthfulQA (Qwen2.5-3B) | TOHA (MTop-Div) | Full"),
        ("Qwen3B-TruthfulQA 0D vs (0D+1D) (Full)", "TruthfulQA (Qwen2.5-3B) | 0D Only | Full", "TruthfulQA (Qwen2.5-3B) | 0D + 1D (Norm) | Full"),
        ("Qwen3B-TruthfulQA 0D vs Length Baseline (Matched)", "TruthfulQA (Qwen2.5-3B) | 0D Only | Matched", "TruthfulQA (Qwen2.5-3B) | Length Baseline | Matched"),
        ("Qwen3B-TruthfulQA 0D vs MST Proxy (Matched)", "TruthfulQA (Qwen2.5-3B) | 0D Only | Matched", "TruthfulQA (Qwen2.5-3B) | MST Per-Layer Proxy | Matched"),
        ("SmolLM 0D vs TOHA (Full)", "TruthfulQA (SmolLM-1.7B) | 0D Only | Full", "TruthfulQA (SmolLM-1.7B) | TOHA (MTop-Div) | Full"),
        ("SmolLM 0D vs (0D+1D) (Full)", "TruthfulQA (SmolLM-1.7B) | 0D Only | Full", "TruthfulQA (SmolLM-1.7B) | 0D + 1D (Norm) | Full"),
        ("SmolLM TOHA vs MSP (Full)", "TruthfulQA (SmolLM-1.7B) | TOHA (MTop-Div) | Full", "TruthfulQA (SmolLM-1.7B) | MSP Baseline | Full"),
        ("SmolLM 0D vs Length Baseline (Matched)", "TruthfulQA (SmolLM-1.7B) | 0D Only | Matched", "TruthfulQA (SmolLM-1.7B) | Length Baseline | Matched"),
        ("SmolLM 0D vs MST Proxy (Matched)", "TruthfulQA (SmolLM-1.7B) | 0D Only | Matched", "TruthfulQA (SmolLM-1.7B) | MST Per-Layer Proxy | Matched"),
        ("Phi-3-3.8B 0D vs TOHA (Full)", "TruthfulQA (Phi-3-3.8B) | 0D Only | Full", "TruthfulQA (Phi-3-3.8B) | TOHA (MTop-Div) | Full"),
        ("Phi-3-3.8B 0D vs (0D+1D) (Full)", "TruthfulQA (Phi-3-3.8B) | 0D Only | Full", "TruthfulQA (Phi-3-3.8B) | 0D + 1D (Norm) | Full"),
        ("Phi-3-3.8B 0D vs Length Baseline (Matched)", "TruthfulQA (Phi-3-3.8B) | 0D Only | Matched", "TruthfulQA (Phi-3-3.8B) | Length Baseline | Matched"),
        ("Phi-3-3.8B 0D vs MST Proxy (Matched)", "TruthfulQA (Phi-3-3.8B) | 0D Only | Matched", "TruthfulQA (Phi-3-3.8B) | MST Per-Layer Proxy | Matched"),
    ]
    
    p_values = []
    test_results = []
    
    for label, keyA, keyB in tests_to_run:
        y_A, p_A, g_A = oof_predictions_dict[keyA]
        y_B, p_B, g_B = oof_predictions_dict[keyB]
        diff, ci_l, ci_h, p_val = fast_bootstrap_diff(y_A, p_A, p_B, g_A, n_bootstraps=2000, random_state=42)
        p_values.append(p_val)
        test_results.append({
            "Comparison": label,
            "AUC Diff (A - B)": f"{diff:+.4f}",
            "95% Bootstrap CI": f"[{ci_l:+.4f}, {ci_h:+.4f}]",
            "Raw p-value": p_val
        })
        
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    holm_adj_p = np.zeros(m)
    for rank, idx in enumerate(sorted_indices):
        adj_p = p_values[idx] * (m - rank)
        holm_adj_p[idx] = min(max(adj_p, 0.0), 1.0)
    for i in range(1, m):
        idx_curr = sorted_indices[i]
        idx_prev = sorted_indices[i-1]
        if holm_adj_p[idx_curr] < holm_adj_p[idx_prev]:
            holm_adj_p[idx_curr] = holm_adj_p[idx_prev]
            
    for i, res in enumerate(test_results):
        res["Holm-Adj p-value"] = f"{holm_adj_p[i]:.5f}"
        res["Raw p-value"] = f"{res['Raw p-value']:.5f}"
        
    df_sig = pd.DataFrame(test_results)
    print(df_sig.to_string(index=False), flush=True)
    df_sig.to_csv("significance_tests_results.csv", index=False)

if __name__ == "__main__":
    run_all()
