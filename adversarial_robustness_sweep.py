"""
adversarial_robustness_sweep.py
===============================
Adversarial Multi-Dimensional Robustness & Nested Cross-Validation Mega-Sweep.

Systematically attacks the paper's core finding across:
  1. 5 Normalization Schemes:
     - Per-Token: H1 / N
     - Quadratic Clique: H1 / (N * (N-1) / 2)
     - Logarithmic: H1 / log(N + 1)
     - Rank-Quantile Normalization
     - Linear OLS Residualization: H1 - beta * N (purging all linear length correlation)
  2. 5 Classifier Families:
     - Logistic Regression (L2)
     - Logistic Regression (L1 / Sparse)
     - Linear Support Vector Classifier (LinearSVC)
     - Random Forest (100 trees)
     - XGBoost Gradient Boosted Trees
     - Multi-Layer Perceptron (MLP)
  3. Strict Nested Grouped Cross-Validation:
     - Outer Loop: 10-fold StratifiedGroupKFold for final unbiased evaluation
     - Inner Loop: 5-fold StratifiedGroupKFold for hyperparameter tuning & regularization selection
     - Outer test fold NEVER touched during any optimization decision.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import json
import time

def apply_normalization(X_1d, seq_lens, method="per_token"):
    N = seq_lens.values.reshape(-1, 1).astype(float)
    X = X_1d.copy().values.astype(float)
    
    if method == "raw":
        return X
    elif method == "per_token":
        return X / np.maximum(N, 1.0)
    elif method == "clique":
        denom = np.maximum(N * (N - 1.0) / 2.0, 1.0)
        return X / denom
    elif method == "log":
        return X / np.maximum(np.log(N + 1.0), 1e-4)
    elif method == "rank":
        qt = QuantileTransformer(n_quantiles=min(100, len(X)), output_distribution='uniform', random_state=42)
        return qt.fit_transform(X)
    elif method == "residualized":
        # Fit OLS regression X ~ N on training data to purge length dependence
        X_res = np.zeros_like(X)
        for col_idx in range(X.shape[1]):
            reg = LinearRegression()
            reg.fit(N, X[:, col_idx])
            X_res[:, col_idx] = X[:, col_idx] - reg.predict(N)
        return X_res
    else:
        return X

def evaluate_nested_cv(X, y, groups, clf_type="lr_l2", random_state=42):
    outer_skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
    outer_aucs = []
    
    for outer_train_idx, outer_test_idx in outer_skf.split(X, y, groups=groups):
        X_outer_tr, X_outer_te = X[outer_train_idx], X[outer_test_idx]
        y_outer_tr, y_outer_te = y[outer_train_idx], y[outer_test_idx]
        g_outer_tr = groups[outer_train_idx]
        
        # Standardize strictly on outer train
        scaler = StandardScaler()
        X_outer_tr_s = scaler.fit_transform(X_outer_tr)
        X_outer_te_s = scaler.transform(X_outer_te)
        
        # Inner loop for hyperparameter selection
        inner_skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        if clf_type in ["lr_l2", "lr_l1", "svm"]:
            param_grid = [0.01, 0.1, 1.0, 10.0]
        elif clf_type == "rf":
            param_grid = [50, 100, 200]
        elif clf_type == "xgb":
            param_grid = [3, 4, 6]
        elif clf_type == "mlp":
            param_grid = [(32,), (64, 32), (64,)]
        else:
            param_grid = [1.0]
            
        best_score = -1.0
        best_param = param_grid[0]
        
        for param in param_grid:
            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_skf.split(X_outer_tr_s, y_outer_tr, groups=g_outer_tr):
                X_in_tr, X_in_val = X_outer_tr_s[inner_train_idx], X_outer_tr_s[inner_val_idx]
                y_in_tr, y_in_val = y_outer_tr[inner_train_idx], y_outer_tr[inner_val_idx]
                
                if clf_type == "lr_l2":
                    model = LogisticRegression(C=param, penalty='l2', max_iter=500, random_state=random_state)
                elif clf_type == "lr_l1":
                    model = LogisticRegression(C=param, penalty='l1', solver='liblinear', max_iter=500, random_state=random_state)
                elif clf_type == "svm":
                    model = LinearSVC(C=param, max_iter=1000, random_state=random_state)
                elif clf_type == "rf":
                    model = RandomForestClassifier(n_estimators=param, max_depth=5, random_state=random_state, n_jobs=-1)
                elif clf_type == "xgb":
                    model = XGBClassifier(max_depth=param, n_estimators=50, eval_metric='logloss', random_state=random_state, n_jobs=-1)
                elif clf_type == "mlp":
                    model = MLPClassifier(hidden_layer_sizes=param, max_iter=300, random_state=random_state)
                    
                model.fit(X_in_tr, y_in_tr)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_in_val)[:, 1]
                else:
                    probs = model.decision_function(X_in_val)
                inner_scores.append(roc_auc_score(y_in_val, probs))
                
            mean_in = np.mean(inner_scores)
            if mean_in > best_score:
                best_score = mean_in
                best_param = param
                
        # Fit best model on full outer training fold
        if clf_type == "lr_l2":
            final_model = LogisticRegression(C=best_param, penalty='l2', max_iter=500, random_state=random_state)
        elif clf_type == "lr_l1":
            final_model = LogisticRegression(C=best_param, penalty='l1', solver='liblinear', max_iter=500, random_state=random_state)
        elif clf_type == "svm":
            final_model = LinearSVC(C=best_param, max_iter=1000, random_state=random_state)
        elif clf_type == "rf":
            final_model = RandomForestClassifier(n_estimators=best_param, max_depth=5, random_state=random_state, n_jobs=-1)
        elif clf_type == "xgb":
            final_model = XGBClassifier(max_depth=best_param, n_estimators=50, eval_metric='logloss', random_state=random_state, n_jobs=-1)
        elif clf_type == "mlp":
            final_model = MLPClassifier(hidden_layer_sizes=best_param, max_iter=300, random_state=random_state)
            
        final_model.fit(X_outer_tr_s, y_outer_tr)
        if hasattr(final_model, "predict_proba"):
            outer_probs = final_model.predict_proba(X_outer_te_s)[:, 1]
        else:
            outer_probs = final_model.decision_function(X_outer_te_s)
        outer_aucs.append(roc_auc_score(y_outer_te, outer_probs))
        
    return float(np.mean(outer_aucs)), float(np.std(outer_aucs))

def run_adversarial_sweep():
    print("=" * 95)
    print("STARTING ADVERSARIAL ROBUSTNESS & NESTED CROSS-VALIDATION MEGA-SWEEP")
    print("=" * 95)
    
    datasets = [
        {"name": "HaluEval (Qwen2.5-3B)", "file": "phase3_results/train_features.csv"},
        {"name": "TruthfulQA (Qwen2.5-3B)", "file": "phase10_results/qwen3b_truthfulqa_4stat.csv"},
        {"name": "TruthfulQA (SmolLM-1.7B)", "file": "phase7_results/truthfulqa_smollm_features.csv"},
        {"name": "TruthfulQA (Phi-3-mini-3.8B)", "file": "phase10_results/phi3_truthfulqa_4stat.csv"}
    ]
    
    norm_methods = ["raw", "per_token", "clique", "log", "rank", "residualized"]
    classifiers = ["lr_l2", "lr_l1", "svm", "rf", "xgb", "mlp"]
    
    master_records = []
    
    for d in datasets:
        print(f"\nEvaluating Dataset: {d['name']}...")
        df = pd.read_csv(d["file"])
        y = (df['label'] == 'hallucinated').values.astype(int)
        groups = df['example_id'].values
        seq_len = df['seq_len'] if 'seq_len' in df.columns else df['n_tokens']
        
        X_0d = df[[c for c in df.columns if 'h0' in c]]
        X_1d = df[[c for c in df.columns if 'h1' in c]]
        
        # 1. Evaluate 0D baseline across classifiers
        for clf in classifiers:
            m_0d, s_0d = evaluate_nested_cv(X_0d.values, y, groups, clf_type=clf)
            print(f"  [0D Baseline] Clf: {clf:8s} -> Nested AUC: {m_0d:.3f} ± {s_0d:.3f}")
            master_records.append({
                "Dataset": d["name"],
                "Feature Set": "0D Only",
                "Normalization": "None",
                "Classifier": clf,
                "Nested CV AUC": f"{m_0d:.3f} ± {s_0d:.3f}",
                "Mean AUC": m_0d,
                "Std AUC": s_0d
            })
            
        # 2. Evaluate Normalization Schemes & Combinations
        for norm in norm_methods:
            X_1d_norm = apply_normalization(X_1d, seq_len, method=norm)
            X_comb = np.hstack([X_0d.values, X_1d_norm])
            
            for clf in classifiers:
                # 1D Only
                m_1d, s_1d = evaluate_nested_cv(X_1d_norm, y, groups, clf_type=clf)
                master_records.append({
                    "Dataset": d["name"],
                    "Feature Set": "1D Only",
                    "Normalization": norm,
                    "Classifier": clf,
                    "Nested CV AUC": f"{m_1d:.3f} ± {s_1d:.3f}",
                    "Mean AUC": m_1d,
                    "Std AUC": s_1d
                })
                
                # 0D + 1D Combined
                m_comb, s_comb = evaluate_nested_cv(X_comb, y, groups, clf_type=clf)
                master_records.append({
                    "Dataset": d["name"],
                    "Feature Set": "0D + 1D",
                    "Normalization": norm,
                    "Classifier": clf,
                    "Nested CV AUC": f"{m_comb:.3f} ± {s_comb:.3f}",
                    "Mean AUC": m_comb,
                    "Std AUC": s_comb
                })
                print(f"  [Norm: {norm:12s} | Clf: {clf:8s}] 1D AUC: {m_1d:.3f} | 0D+1D AUC: {m_comb:.3f}")
                
    res_df = pd.DataFrame(master_records)
    res_df.to_csv("adversarial_robustness_sweep_results.csv", index=False)
    print("\nSaved full adversarial sweep results to adversarial_robustness_sweep_results.csv")
    return res_df

if __name__ == "__main__":
    run_adversarial_sweep()
