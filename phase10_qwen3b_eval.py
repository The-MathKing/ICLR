import os
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "/Volumes/2TB/hf_cache", an external drive on the
#  authors' Mac, which does not exist on other machines.)
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

# Roadmap item 7: TruthfulQA has 817 questions; earlier runs capped this at
# 150/500, leaving those settings at 14-54% power and confounding cross-model
# comparisons with sample size. Set FULL_TRUTHFULQA=150 to reproduce the old run.
FULL_TRUTHFULQA = 817


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
    return mst.sum()

def extract_features(diagrams):
    features = {}
    
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf] if len(h0) > 0 else np.array([])
    if len(h0_finite) > 0:
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
        features['h0_max_lifetime'] = np.max(h0_lifetimes)
        features['h0_total_persistence'] = np.sum(h0_lifetimes)
    else:
        features['h0_max_lifetime'] = 0.0
        features['h0_total_persistence'] = 0.0
        
    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf] if len(h1) > 0 else np.array([])
    if len(h1_finite) > 0:
        h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
        max_idx = np.argmax(h1_lifetimes)
        features['h1_max_lifetime'] = np.max(h1_lifetimes)
        features['h1_total_persistence'] = np.sum(h1_lifetimes)
        features['h1_max_birth'] = h1_finite[max_idx, 0]
        features['h1_max_death'] = h1_finite[max_idx, 1]
    else:
        features['h1_max_lifetime'] = 0.0
        features['h1_total_persistence'] = 0.0
        features['h1_max_birth'] = 0.0
        features['h1_max_death'] = 0.0

    return features

def run_extraction():
    # Written to a new filename (not the original qwen3b_truthfulqa.csv) so the
    # original reduced-descriptor-bank cache is preserved rather than overwritten.
    out_csv = "phase10_results/qwen3b_truthfulqa_4stat.csv"
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
        
    os.makedirs("phase10_results", exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device)
    
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(dataset)
    df = df.head(FULL_TRUTHFULQA)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        best_ans = row['best_answer']
        bad_ans = row['incorrect_answers'][0] if len(row['incorrect_answers']) > 0 else "I don't know."
        
        for ans_type, ans_text in [('grounded', best_ans), ('hallucinated', bad_ans)]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": ans_text}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            
            prompt_only_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
            prompt_only = tokenizer.apply_chat_template(prompt_only_messages, tokenize=False, add_generation_prompt=True)
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
                
                avg_attn = np.mean(attn, axis=0)
                W = np.maximum(avg_attn, avg_attn.T)
                D = 1.0 - W
                np.fill_diagonal(D, 0.0)
                
                mtop = compute_mtop_div(D, prompt_len)
                layer_features[f"layer_{layer_idx}_mtop_div"] = mtop
                
                res = ripser.ripser(D, distance_matrix=True, maxdim=1)
                feats = extract_features(res['dgms'])
                for k, v in feats.items():
                    layer_features[f"layer_{layer_idx}_{k}"] = v
                    
            logits = outputs.logits[0, :-1, :]
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            msp_score = max_probs.mean().item()
            
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
    return results_df

def run_evaluation(df):
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    y = df['target']
    
    X_0d = df[[c for c in df.columns if 'h0' in c]]
    X_1d_raw = df[[c for c in df.columns if 'h1' in c]]
    X_1d_norm = X_1d_raw.div(df['seq_len'], axis=0)
    X_mtop = df[[c for c in df.columns if 'mtop_div' in c]]
    msp = df['msp_score'].values
    
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    auc_msp = []
    auc_0d, auc_1d_raw, auc_1d_norm, auc_mtop = [], [], [], []
    
    groups = df['example_id']
    for train_idx, test_idx in skf.split(X_0d, y, groups=groups):
        # MSP
        lr = LogisticRegression()
        lr.fit(msp[train_idx].reshape(-1, 1), y.iloc[train_idx])
        auc_msp.append(roc_auc_score(y.iloc[test_idx], lr.predict_proba(msp[test_idx].reshape(-1, 1))[:, 1]))
        
        # 0D
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_0d.iloc[train_idx], y.iloc[train_idx])
        auc_0d.append(roc_auc_score(y.iloc[test_idx], lr.predict_proba(X_0d.iloc[test_idx])[:, 1]))
        
        # 1D Raw
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_1d_raw.iloc[train_idx], y.iloc[train_idx])
        auc_1d_raw.append(roc_auc_score(y.iloc[test_idx], lr.predict_proba(X_1d_raw.iloc[test_idx])[:, 1]))
        
        # 1D Normalized
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_1d_norm.iloc[train_idx], y.iloc[train_idx])
        auc_1d_norm.append(roc_auc_score(y.iloc[test_idx], lr.predict_proba(X_1d_norm.iloc[test_idx])[:, 1]))
        
        # MTop-Div (TOHA)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_mtop.iloc[train_idx], y.iloc[train_idx])
        auc_mtop.append(roc_auc_score(y.iloc[test_idx], lr.predict_proba(X_mtop.iloc[test_idx])[:, 1]))
        
    print(f"MSP Baseline (LR): {np.mean(auc_msp):.3f} +/- {np.std(auc_msp):.3f}")
    print(f"0D Only (LR):      {np.mean(auc_0d):.3f} +/- {np.std(auc_0d):.3f}")
    print(f"1D Raw (LR):       {np.mean(auc_1d_raw):.3f} +/- {np.std(auc_1d_raw):.3f}")
    print(f"1D Norm (LR):      {np.mean(auc_1d_norm):.3f} +/- {np.std(auc_1d_norm):.3f}")
    print(f"TOHA MTop-Div (LR):{np.mean(auc_mtop):.3f} +/- {np.std(auc_mtop):.3f}")

if __name__ == "__main__":
    df = run_extraction()
    run_evaluation(df)
