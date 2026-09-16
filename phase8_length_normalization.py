import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def evaluate_dataset(name, filepath):
    print(f"\n--- Evaluating {name} ---")
    df = pd.read_csv(filepath)
    y = (df['label'] == 'hallucinated').astype(int)
    
    cols_0d = [c for c in df.columns if 'h0' in c]
    cols_1d = [c for c in df.columns if 'h1' in c]
    
    X_0d = df[cols_0d]
    X_1d_raw = df[cols_1d]
    X_1d_norm = X_1d_raw.div(df['seq_len'], axis=0)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    auc_1d_lr_raw, auc_1d_lr_norm = [], []
    auc_1d_xgb_raw, auc_1d_xgb_norm = [], []
    auc_comb_lr_norm, auc_comb_xgb_norm = [], []
    
    for train_idx, test_idx in skf.split(X_1d_raw, y):
        # 1D Raw LR
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_1d_raw.iloc[train_idx], y.iloc[train_idx])
        auc_1d_lr_raw.append(roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_1d_raw.iloc[test_idx])[:, 1]))
        
        # 1D Normalized LR
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_1d_norm.iloc[train_idx], y.iloc[train_idx])
        auc_1d_lr_norm.append(roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_1d_norm.iloc[test_idx])[:, 1]))
        
        # 1D Raw XGB
        clf = XGBClassifier(eval_metric='logloss')
        clf.fit(X_1d_raw.iloc[train_idx], y.iloc[train_idx])
        auc_1d_xgb_raw.append(roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_1d_raw.iloc[test_idx])[:, 1]))
        
        # 1D Normalized XGB
        clf = XGBClassifier(eval_metric='logloss')
        clf.fit(X_1d_norm.iloc[train_idx], y.iloc[train_idx])
        auc_1d_xgb_norm.append(roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_1d_norm.iloc[test_idx])[:, 1]))

        # Combined (0D + 1D Normalized)
        X_comb_norm = pd.concat([X_0d, X_1d_norm], axis=1)
        
        # Combined LR
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_comb_norm.iloc[train_idx], y.iloc[train_idx])
        auc_comb_lr_norm.append(roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_comb_norm.iloc[test_idx])[:, 1]))
        
        # Combined XGB
        clf = XGBClassifier(eval_metric='logloss')
        clf.fit(X_comb_norm.iloc[train_idx], y.iloc[train_idx])
        auc_comb_xgb_norm.append(roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_comb_norm.iloc[test_idx])[:, 1]))


    print(f"1D Raw (LR):         {np.mean(auc_1d_lr_raw):.3f} +/- {np.std(auc_1d_lr_raw):.3f}")
    print(f"1D Normalized (LR):  {np.mean(auc_1d_lr_norm):.3f} +/- {np.std(auc_1d_lr_norm):.3f}")
    print(f"1D Raw (XGB):        {np.mean(auc_1d_xgb_raw):.3f} +/- {np.std(auc_1d_xgb_raw):.3f}")
    print(f"1D Normalized (XGB): {np.mean(auc_1d_xgb_norm):.3f} +/- {np.std(auc_1d_xgb_norm):.3f}")
    print(f"Combined Norm (LR):  {np.mean(auc_comb_lr_norm):.3f} +/- {np.std(auc_comb_lr_norm):.3f}")
    print(f"Combined Norm (XGB): {np.mean(auc_comb_xgb_norm):.3f} +/- {np.std(auc_comb_xgb_norm):.3f}")


if __name__ == "__main__":
    evaluate_dataset("Qwen2.5-3B (HaluEval)", "phase3_results/train_features.csv")
    evaluate_dataset("SmolLM-1.7B (TruthfulQA)", "phase7_results/truthfulqa_smollm_features.csv")
