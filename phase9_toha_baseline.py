import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

def evaluate_toha(filepath, name):
    print(f"\n--- Evaluating TOHA MTop-Div Baseline on {name} ---")
    df = pd.read_csv(filepath)
    y = (df['label'] == 'hallucinated').astype(int)
    
    # MTop-Div is the sum of H0 interval lengths. We sum it across all layers.
    h0_cols = [c for c in df.columns if 'h0_total_persistence' in c]
    df['toha_mtop_div'] = df[h0_cols].sum(axis=1)
    
    X_toha = df[['toha_mtop_div']]
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_toha = []
    
    for train_idx, test_idx in skf.split(X_toha, y):
        lr = LogisticRegression()
        lr.fit(X_toha.iloc[train_idx], y.iloc[train_idx])
        preds = lr.predict_proba(X_toha.iloc[test_idx])[:, 1]
        auc_toha.append(roc_auc_score(y.iloc[test_idx], preds))
        
    print(f"TOHA MTop-Div (LR) ROC-AUC: {np.mean(auc_toha):.4f} +/- {np.std(auc_toha):.4f}")

if __name__ == "__main__":
    evaluate_toha("phase3_results/train_features.csv", "Qwen2.5-3B (HaluEval)")
    evaluate_toha("phase7_results/truthfulqa_smollm_features.csv", "SmolLM-1.7B (TruthfulQA)")
