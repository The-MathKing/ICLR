"""
phase10_mistral_eval.py
=======================
Evaluation of Mistral-7B-Instruct on TruthfulQA under the standardized
4-statistic topological feature bank and grouped cross-validation protocol.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import scipy.sparse.csgraph as csgraph
import ripser
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def compute_mtop_div(D, prompt_len):
    N = D.shape[0]
    R_len = N - prompt_len
    if R_len <= 0: return 0.0
    
    D_msf = np.zeros((R_len + 1, R_len + 1))
    D_msf[1:, 1:] = D[prompt_len:, prompt_len:]
    min_dist_to_P = np.min(D[:prompt_len, prompt_len:], axis=0)
    D_msf[0, 1:] = min_dist_to_P
    D_msf[1:, 0] = min_dist_to_P
    mst = csgraph.minimum_spanning_tree(D_msf)
    return float(mst.sum())

def extract_standard_descriptors(diagrams):
    features = {}
    
    # H0 descriptors (2 per layer)
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf] if len(h0) > 0 else np.array([])
    if len(h0_finite) > 0:
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
        features['h0_max_lifetime'] = float(np.max(h0_lifetimes))
        features['h0_total_persistence'] = float(np.sum(h0_lifetimes))
    else:
        features['h0_max_lifetime'] = 0.0
        features['h0_total_persistence'] = 0.0
        
    # H1 descriptors (4 per layer)
    if len(diagrams) > 1:
        h1 = diagrams[1]
        h1_finite = h1[h1[:, 1] != np.inf] if len(h1) > 0 else np.array([])
        if len(h1_finite) > 0:
            h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
            max_idx = np.argmax(h1_lifetimes)
            features['h1_max_lifetime'] = float(np.max(h1_lifetimes))
            features['h1_total_persistence'] = float(np.sum(h1_lifetimes))
            features['h1_max_birth'] = float(h1_finite[max_idx, 0])
            features['h1_max_death'] = float(h1_finite[max_idx, 1])
        else:
            features['h1_max_lifetime'] = 0.0
            features['h1_total_persistence'] = 0.0
            features['h1_max_birth'] = 0.0
            features['h1_max_death'] = 0.0
    else:
        features['h1_max_lifetime'] = 0.0
        features['h1_total_persistence'] = 0.0
        features['h1_max_birth'] = 0.0
        features['h1_max_death'] = 0.0
        
    return features

def run_extraction(num_samples=817):
    out_csv = "phase10_results/mistral7b_truthfulqa_4stat_full.csv"
    if os.path.exists(out_csv):
        print(f"Loading cached Mistral-7B features from {out_csv}")
        return pd.read_csv(out_csv)
        
    os.makedirs("phase10_results", exist_ok=True)
    # 16GB-RAM host cannot hold a 14GB fp16 7B model reliably (OOM-kill / segfault
    # observed). Use int8 weight-only quantization (~7GB) on CPU instead.
    from transformers import QuantoConfig
    device = "cpu"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print(f"Loading {model_name} onto {device} (int8 quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True, device_map={"": device},
        quantization_config=QuantoConfig(weights="int8"),
    )
    
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(dataset).head(num_samples)
    
    results = []
    print(f"Extracting attention graphs and persistent homology across {len(df)} questions ({len(df)*2} instances)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        best_ans = row['best_answer']
        bad_ans = row['incorrect_answers'][0] if len(row['incorrect_answers']) > 0 else "I don't know."
        
        for ans_type, ans_text in [('grounded', best_ans), ('hallucinated', bad_ans)]:
            prompt = f"[INST] {question} [/INST] {ans_text}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            
            prompt_only = f"[INST] {question} [/INST] "
            prompt_len = tokenizer(prompt_only, return_tensors="pt").input_ids.shape[1]
            
            if seq_len > 1024:
                continue
                
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                
            attentions = outputs.attentions
            num_layers = len(attentions)
            
            layer_features = {}
            for layer_idx in range(num_layers):
                attn = attentions[layer_idx].float().cpu().numpy()[0]
                attn = np.nan_to_num(attn, nan=0.0)
                
                # Head average
                avg_attn = np.mean(attn, axis=0)
                W = np.maximum(avg_attn, avg_attn.T)
                D = 1.0 - W
                np.fill_diagonal(D, 0.0)
                
                # MST weight proxy
                mst = csgraph.minimum_spanning_tree(D)
                layer_features[f"layer_{layer_idx}_mst_weight"] = float(mst.sum())
                
                # TOHA MTop-Div
                mtop = compute_mtop_div(D, prompt_len)
                layer_features[f"layer_{layer_idx}_mtop_div"] = mtop
                
                # Ripser persistent homology (maxdim=1)
                res = ripser.ripser(D, distance_matrix=True, maxdim=1)
                feats = extract_standard_descriptors(res['dgms'])
                for k, v in feats.items():
                    layer_features[f"layer_{layer_idx}_{k}"] = v
                    
            logits = outputs.logits[0, :-1, :]
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            msp_score = float(max_probs.mean().item())
            
            row_data = {
                "example_id": idx,
                "label": ans_type,
                "seq_len": seq_len,
                "msp_score": msp_score
            }
            row_data.update(layer_features)
            results.append(row_data)
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved {len(results_df)} instances to {out_csv}")
    return results_df

def evaluate_mistral(df):
    y = (df['label'] == 'hallucinated').values.astype(int)
    groups = df['example_id'].values
    seq_len = df['seq_len']
    
    # Features
    X_0d = df[[c for c in df.columns if 'h0' in c]]
    X_1d_raw = df[[c for c in df.columns if 'h1' in c]]
    X_1d_norm = X_1d_raw.div(seq_len, axis=0)
    X_comb_norm = pd.concat([X_0d, X_1d_norm], axis=1)
    X_mst = df[[c for c in df.columns if 'mst_weight' in c]]
    
    # TOHA: single scalar summed across layers
    mtop_cols = [c for c in df.columns if 'mtop_div' in c]
    X_toha = df[mtop_cols].sum(axis=1).to_frame()
    X_msp = df[['msp_score']]
    X_len = df[['seq_len']]
    
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    def eval_feature(X, name):
        aucs_lr, aucs_xgb = [], []
        for train_idx, test_idx in skf.split(X, y, groups=groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # LR
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_tr, y_tr)
            aucs_lr.append(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]))
            
            # XGB
            xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=50, max_depth=3, n_jobs=1)
            xgb.fit(X_tr, y_tr)
            aucs_xgb.append(roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1]))
            
        print(f"{name:32s} | LR: {np.mean(aucs_lr):.3f} ± {np.std(aucs_lr):.3f} | XGB: {np.mean(aucs_xgb):.3f} ± {np.std(aucs_xgb):.3f}")
        return np.mean(aucs_lr), np.mean(aucs_xgb)
        
    print("=" * 80)
    print("MISTRAL-7B TRUTHFULQA EVALUATION UNDER GROUPED CROSS-VALIDATION")
    print("=" * 80)
    eval_feature(X_len, "Length Baseline (d=1)")
    eval_feature(X_msp, "MSP Baseline (d=1)")
    eval_feature(X_toha, "TOHA MTop-Div (d=1)")
    eval_feature(X_mst, "MST Per-Layer Proxy (d=32)")
    eval_feature(X_0d, "0D Only (d=64)")
    eval_feature(X_1d_raw, "1D Only Raw (d=128)")
    eval_feature(X_1d_norm, "1D Only Norm (d=128)")
    eval_feature(X_comb_norm, "0D + 1D Norm (d=192)")

if __name__ == "__main__":
    df = run_extraction(num_samples=817)
    evaluate_mistral(df)
