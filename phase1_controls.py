import os
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "<an external drive>", an external drive on the
#  authors' Mac, which does not exist on other machines.)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import scipy.sparse.csgraph as csgraph
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def extract_control_features():
    out_csv = "phase1_controls_halueval.csv"
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
        
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device)
    
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:500]")
    df = pd.DataFrame(dataset)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for ans_type in ['right_answer', 'hallucinated_answer']:
            prompt_only_str = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer:"
            full_prompt_str = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer: {row[ans_type]}"
            
            p_inputs = tokenizer(prompt_only_str, return_tensors="pt")
            f_inputs = tokenizer(full_prompt_str, return_tensors="pt").to(device)
            
            n_prompt_tokens = p_inputs.input_ids.shape[1]
            n_seq_tokens = f_inputs.input_ids.shape[1]
            n_answer_tokens = n_seq_tokens - n_prompt_tokens
            
            if n_seq_tokens > 1024:
                continue
                
            with torch.no_grad():
                outputs = model(**f_inputs, output_attentions=True, output_hidden_states=True)
                
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states[-1] # last layer hidden states (1, seq_len, dim)
            
            # Mean pool answer span hidden states
            ans_hidden = hidden_states[0, n_prompt_tokens:, :]
            pooled_hidden = ans_hidden.mean(dim=0).float().cpu().numpy()
            
            layer_features = {}
            for layer_idx in range(len(attentions)):
                attn = attentions[layer_idx].float().cpu().numpy()[0]
                attn = np.nan_to_num(attn, nan=0.0)
                
                # Mean over heads
                avg_attn = np.mean(attn, axis=0)
                
                # Non-topological stats
                layer_features[f"layer_{layer_idx}_mean_attn"] = np.mean(avg_attn)
                layer_features[f"layer_{layer_idx}_max_attn"] = np.max(avg_attn)
                layer_features[f"layer_{layer_idx}_sink_mass"] = np.sum(avg_attn[:, 0]) / n_seq_tokens
                
                # MST total weight
                W = np.maximum(avg_attn, avg_attn.T)
                D = 1.0 - W
                np.fill_diagonal(D, 0.0)
                mst = csgraph.minimum_spanning_tree(D)
                layer_features[f"layer_{layer_idx}_mst_weight"] = mst.sum()
                layer_features[f"layer_{layer_idx}_mst_max_edge"] = mst.max()
                
            row_data = {
                "example_id": idx,
                "label": "grounded" if ans_type == 'right_answer' else "hallucinated",
                "n_prompt_tokens": n_prompt_tokens,
                "n_answer_tokens": n_answer_tokens,
                "log_n_answer_tokens": np.log(n_answer_tokens + 1)
            }
            row_data.update(layer_features)
            for i, val in enumerate(pooled_hidden):
                row_data[f"hidden_{i}"] = val
                
            results.append(row_data)
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)
    return res_df

def run_evaluations(df):
    y = (df['label'] == 'hallucinated').astype(int)
    groups = df['example_id']
    
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    def eval_features(X, name):
        aucs = []
        for train_idx, test_idx in skf.split(X, y, groups=groups):
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = lr.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[test_idx], preds))
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"{name:35s}: {mean_auc:.3f} +/- {std_auc:.3f}")
        return mean_auc
        
    print("\n=== Phase 1 Control Evaluations ===")
    
    # 1.1 Length-Only Baseline
    X_len = df[['n_prompt_tokens', 'n_answer_tokens', 'log_n_answer_tokens']]
    eval_features(X_len, "1.1 Length-Only Baseline")
    
    # 1.3 Non-Topological Stats (Combined)
    X_non_top = df[[c for c in df.columns if any(x in c for x in ['mean_attn', 'max_attn', 'sink_mass'])]]
    eval_features(X_non_top, "1.3 Non-Topological Stats")
    
    # MST Total Weight (Equivalent to 0D)
    X_mst = df[[c for c in df.columns if 'mst_weight' in c]]
    eval_features(X_mst, "1.3 MST Total Weight (0D proxy)")
    
    # 1.4 Hidden-State Linear Probe
    X_hidden = df[[c for c in df.columns if 'hidden_' in c]]
    eval_features(X_hidden, "1.4 Hidden-State Probe")

if __name__ == "__main__":
    df = extract_control_features()
    run_evaluations(df)
