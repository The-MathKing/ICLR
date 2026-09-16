"""
phase10_tinyllama_eval.py
==========================
Adds a genuine Llama-architecture model (TinyLlama-1.1B-Chat-v1.0, ungated,
ships LlamaForCausalLM) to the audit grid, addressing the review point that
"Qwen2.5-1.5B/3B" are two scales of one family, not two families -- this is
a fifth family, not just a sixth scale. Extraction pattern matches
phase10_phi3_eval.py exactly (same TruthfulQA subset construction, same
4-statistic descriptor bank) so it plugs directly into master_pipeline.py.
"""
import os
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "/Volumes/2TB/hf_cache", an external drive on the
#  authors' Mac, which does not exist on other machines.)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import scipy.sparse.csgraph as csgraph
import ripser
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Roadmap item 7: TruthfulQA has 817 questions; earlier runs capped this at
# 150/500, leaving those settings at 14-54% power and confounding cross-model
# comparisons with sample size. Set FULL_TRUTHFULQA=150 to reproduce the old run.
FULL_TRUTHFULQA = 817


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_CSV = "phase10_results/tinyllama_truthfulqa_4stat.csv"


def compute_mtop_div(D, prompt_len):
    N = D.shape[0]
    R_len = N - prompt_len
    if R_len <= 0:
        return 0.0
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
    if os.path.exists(OUT_CSV):
        print(f"Found cached: {OUT_CSV}")
        return pd.read_csv(OUT_CSV)

    os.makedirs("phase10_results", exist_ok=True)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    print(f"Model class: {config.architectures}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device)
    model.eval()

    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(dataset)
    df = df.head(FULL_TRUTHFULQA)

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        best_ans = row['best_answer']
        bad_ans = row['incorrect_answers'][0] if len(row['incorrect_answers']) > 0 else "I don't know."

        for ans_type, ans_text in [('grounded', best_ans), ('hallucinated', bad_ans)]:
            prompt = f"<|user|>\n{question}</s>\n<|assistant|>\n{ans_text}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]

            prompt_only = f"<|user|>\n{question}</s>\n<|assistant|>\n"
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

            row_data = {"example_id": idx, "label": ans_type, "seq_len": seq_len, "msp_score": msp_score}
            row_data.update(layer_features)
            results.append(row_data)

        if (idx + 1) % 30 == 0:
            print(f"  {idx+1}/150")
            pd.DataFrame(results).to_csv(OUT_CSV + ".tmp", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(results_df)} rows -> {OUT_CSV}")
    return results_df


if __name__ == "__main__":
    run_extraction()
