import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def evaluate_features(X, y, groups, cv_type="grouped", n_splits=10, model_type="lr"):
    if cv_type == "grouped":
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = skf.split(X, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = skf.split(X, y)
        
    aucs = []
    for train_idx, test_idx in splits:
        # Check no group leakage in test for grouped
        if cv_type == "grouped":
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            assert len(train_groups.intersection(test_groups)) == 0, "Group leakage detected!"
            
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        if model_type == "lr":
            clf = LogisticRegression(max_iter=1000)
        elif model_type == "xgb":
            clf = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=100, max_depth=4)
            
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, probs))
        
    return np.mean(aucs), np.std(aucs)

def run_full_cv_comparison():
    print("=========================================================================")
    print("AUDIT: Ungrouped (StratifiedKFold) vs Grouped (StratifiedGroupKFold)")
    print("=========================================================================")
    
    # 1. HaluEval QA (Qwen2.5-3B)
    print("\n--- 1. HaluEval QA (Qwen2.5-3B, N=2000) ---")
    df_h = pd.read_csv("phase3_results/train_features.csv")
    df_msp = pd.read_csv("phase4_msp_features.csv")
    df_h = pd.merge(df_h, df_msp[['example_id', 'label', 'mean_prob', 'min_prob']], on=['example_id', 'label'], how='left')
    
    y_h = (df_h['label'] == 'hallucinated').astype(int)
    groups_h = df_h['example_id']
    
    X_0d_h = df_h[[c for c in df_h.columns if 'h0' in c]]
    X_1d_raw_h = df_h[[c for c in df_h.columns if 'h1' in c]]
    X_1d_norm_h = X_1d_raw_h.div(df_h['seq_len'], axis=0)
    X_comb_h = pd.concat([X_0d_h, X_1d_norm_h], axis=1)
    
    h0_cols_h = [c for c in df_h.columns if 'h0_total_persistence' in c]
    X_toha_h = df_h[h0_cols_h].sum(axis=1).to_frame(name='toha_mtop')
    X_msp_h = df_h[['mean_prob', 'min_prob']]
    
    experiments_h = [
        ("MSP Baseline", X_msp_h),
        ("TOHA (MTop-Div)", X_toha_h),
        ("0D Only", X_0d_h),
        ("1D Only (Raw)", X_1d_raw_h),
        ("1D Only (Normalized)", X_1d_norm_h),
        ("Combined (0D + 1D Norm)", X_comb_h)
    ]
    
    for name, feat in experiments_h:
        m_un, s_un = evaluate_features(feat, y_h, groups_h, cv_type="ungrouped", model_type="lr")
        m_gr, s_gr = evaluate_features(feat, y_h, groups_h, cv_type="grouped", model_type="lr")
        print(f"LR  | {name:25s} | Ungrouped: {m_un:.4f} +/- {s_un:.4f} | Grouped: {m_gr:.4f} +/- {s_gr:.4f}")
        
    for name, feat in [("0D Only", X_0d_h), ("1D Only (Raw)", X_1d_raw_h), ("1D Only (Norm)", X_1d_norm_h)]:
        m_un, s_un = evaluate_features(feat, y_h, groups_h, cv_type="ungrouped", model_type="xgb")
        m_gr, s_gr = evaluate_features(feat, y_h, groups_h, cv_type="grouped", model_type="xgb")
        print(f"XGB | {name:25s} | Ungrouped: {m_un:.4f} +/- {s_un:.4f} | Grouped: {m_gr:.4f} +/- {s_gr:.4f}")

    # 2. TruthfulQA (Qwen2.5-3B)
    print("\n--- 2. TruthfulQA (Qwen2.5-3B, N=300) ---")
    df_q3 = pd.read_csv("phase10_results/qwen3b_truthfulqa.csv")
    y_q3 = (df_q3['label'] == 'hallucinated').astype(int)
    groups_q3 = df_q3['example_id']
    
    X_0d_q3 = df_q3[[c for c in df_q3.columns if 'h0' in c]]
    X_1d_raw_q3 = df_q3[[c for c in df_q3.columns if 'h1' in c]]
    X_1d_norm_q3 = X_1d_raw_q3.div(df_q3['seq_len'], axis=0)
    X_toha_q3 = df_q3[[c for c in df_q3.columns if 'mtop_div' in c]]
    X_msp_q3 = df_q3[['msp_score']]
    
    experiments_q3 = [
        ("MSP Baseline", X_msp_q3),
        ("TOHA (MTop-Div)", X_toha_q3),
        ("0D Only", X_0d_q3),
        ("1D Only (Raw)", X_1d_raw_q3),
        ("1D Only (Normalized)", X_1d_norm_q3),
    ]
    
    for name, feat in experiments_q3:
        m_un, s_un = evaluate_features(feat, y_q3, groups_q3, cv_type="ungrouped", model_type="lr")
        m_gr, s_gr = evaluate_features(feat, y_q3, groups_q3, cv_type="grouped", model_type="lr")
        print(f"LR  | {name:25s} | Ungrouped: {m_un:.4f} +/- {s_un:.4f} | Grouped: {m_gr:.4f} +/- {s_gr:.4f}")

    # 3. TruthfulQA (SmolLM-1.7B)
    print("\n--- 3. TruthfulQA (SmolLM-1.7B, N=1000) ---")
    df_sm = pd.read_csv("phase7_results/truthfulqa_smollm_features.csv")
    y_sm = (df_sm['label'] == 'hallucinated').astype(int)
    groups_sm = df_sm['example_id']
    
    X_0d_sm = df_sm[[c for c in df_sm.columns if 'h0' in c]]
    X_1d_raw_sm = df_sm[[c for c in df_sm.columns if 'h1' in c]]
    X_1d_norm_sm = X_1d_raw_sm.div(df_sm['seq_len'], axis=0)
    X_comb_sm = pd.concat([X_0d_sm, X_1d_norm_sm], axis=1)
    
    h0_cols_sm = [c for c in df_sm.columns if 'h0_total_persistence' in c]
    X_toha_sm = df_sm[h0_cols_sm].sum(axis=1).to_frame(name='toha_mtop')
    X_msp_sm = df_sm[['msp_score']]
    
    experiments_sm = [
        ("MSP Baseline", X_msp_sm),
        ("TOHA (MTop-Div)", X_toha_sm),
        ("0D Only", X_0d_sm),
        ("1D Only (Raw)", X_1d_raw_sm),
        ("1D Only (Normalized)", X_1d_norm_sm),
        ("Combined (0D + 1D Norm)", X_comb_sm)
    ]
    
    for name, feat in experiments_sm:
        m_un, s_un = evaluate_features(feat, y_sm, groups_sm, cv_type="ungrouped", model_type="lr")
        m_gr, s_gr = evaluate_features(feat, y_sm, groups_sm, cv_type="grouped", model_type="lr")
        print(f"LR  | {name:25s} | Ungrouped: {m_un:.4f} +/- {s_un:.4f} | Grouped: {m_gr:.4f} +/- {s_gr:.4f}")
        
    for name, feat in [("0D Only", X_0d_sm), ("1D Only (Raw)", X_1d_raw_sm), ("1D Only (Norm)", X_1d_norm_sm)]:
        m_un, s_un = evaluate_features(feat, y_sm, groups_sm, cv_type="ungrouped", model_type="xgb")
        m_gr, s_gr = evaluate_features(feat, y_sm, groups_sm, cv_type="grouped", model_type="xgb")
        print(f"XGB | {name:25s} | Ungrouped: {m_un:.4f} +/- {s_un:.4f} | Grouped: {m_gr:.4f} +/- {s_gr:.4f}")

if __name__ == '__main__':
    run_full_cv_comparison()
