import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats

def run_tests():
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("=== TruthfulQA (Qwen2.5-3B) ===")
    df = pd.read_csv("phase10_results/qwen3b_truthfulqa.csv")
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    y = df['target']
    
    X_0d = df[[c for c in df.columns if 'h0' in c]]
    X_1d_raw = df[[c for c in df.columns if 'h1' in c]]
    X_1d_norm = X_1d_raw.div(df['seq_len'], axis=0)
    X_mtop = df[[c for c in df.columns if 'mtop_div' in c]]
    msp = df['msp_score'].values
    
    auc_0d, auc_1d_raw, auc_1d_norm, auc_mtop, auc_msp = [], [], [], [], []
    f1_0d, f1_1d_raw, f1_1d_norm, f1_mtop, f1_msp = [], [], [], [], []
    
    for train_idx, test_idx in skf.split(X_0d, y):
        # 0D
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_0d.iloc[train_idx], y.iloc[train_idx])
        preds = lr.predict(X_0d.iloc[test_idx])
        probs = lr.predict_proba(X_0d.iloc[test_idx])[:, 1]
        auc_0d.append(roc_auc_score(y.iloc[test_idx], probs))
        f1_0d.append(f1_score(y.iloc[test_idx], preds))
        
        # 1D Raw
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_1d_raw.iloc[train_idx], y.iloc[train_idx])
        preds = lr.predict(X_1d_raw.iloc[test_idx])
        probs = lr.predict_proba(X_1d_raw.iloc[test_idx])[:, 1]
        auc_1d_raw.append(roc_auc_score(y.iloc[test_idx], probs))
        f1_1d_raw.append(f1_score(y.iloc[test_idx], preds))
        
        # 1D Normalized
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_1d_norm.iloc[train_idx], y.iloc[train_idx])
        preds = lr.predict(X_1d_norm.iloc[test_idx])
        probs = lr.predict_proba(X_1d_norm.iloc[test_idx])[:, 1]
        auc_1d_norm.append(roc_auc_score(y.iloc[test_idx], probs))
        f1_1d_norm.append(f1_score(y.iloc[test_idx], preds))
        
        # TOHA
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_mtop.iloc[train_idx], y.iloc[train_idx])
        preds = lr.predict(X_mtop.iloc[test_idx])
        probs = lr.predict_proba(X_mtop.iloc[test_idx])[:, 1]
        auc_mtop.append(roc_auc_score(y.iloc[test_idx], probs))
        f1_mtop.append(f1_score(y.iloc[test_idx], preds))
        
        # MSP
        lr = LogisticRegression()
        lr.fit(msp[train_idx].reshape(-1, 1), y.iloc[train_idx])
        preds = lr.predict(msp[test_idx].reshape(-1, 1))
        probs = lr.predict_proba(msp[test_idx].reshape(-1, 1))[:, 1]
        auc_msp.append(roc_auc_score(y.iloc[test_idx], probs))
        f1_msp.append(f1_score(y.iloc[test_idx], preds))

    _, p_0d_vs_toha = stats.ttest_rel(auc_0d, auc_mtop)
    
    print(f"MSP Baseline (LR): AUC = {np.mean(auc_msp):.3f} +/- {np.std(auc_msp):.3f}, F1 = {np.mean(f1_msp):.3f} +/- {np.std(f1_msp):.3f}")
    print(f"TOHA MTop-Div (LR): AUC = {np.mean(auc_mtop):.3f} +/- {np.std(auc_mtop):.3f}, F1 = {np.mean(f1_mtop):.3f} +/- {np.std(f1_mtop):.3f}")
    print(f"0D Only (LR):      AUC = {np.mean(auc_0d):.3f} +/- {np.std(auc_0d):.3f}, F1 = {np.mean(f1_0d):.3f} +/- {np.std(f1_0d):.3f}")
    print(f"1D Raw (LR):       AUC = {np.mean(auc_1d_raw):.3f} +/- {np.std(auc_1d_raw):.3f}, F1 = {np.mean(f1_1d_raw):.3f} +/- {np.std(f1_1d_raw):.3f}")
    print(f"1D Norm (LR):      AUC = {np.mean(auc_1d_norm):.3f} +/- {np.std(auc_1d_norm):.3f}, F1 = {np.mean(f1_1d_norm):.3f} +/- {np.std(f1_1d_norm):.3f}")
    
    print(f"p-value 0D vs TOHA: {p_0d_vs_toha:.5f}")

if __name__ == '__main__':
    run_tests()
