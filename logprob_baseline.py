"""
logprob_baseline.py
====================
Adds a mean-token-log-probability / perplexity baseline (review item B8:
"missing baselines" -- perplexity / mean token log-prob was not audited
anywhere in the paper). Computed on the SAME 200-question HaluEval QA
subset (Qwen2.5-3B-Instruct) used by per_head_tda_ablation.py, so it is
directly comparable to the head-averaged/per-head AUCs already reported
there (0.677 / 0.892).

Feature: mean log P(answer_token_i | prompt, answer_<i>) over the answer
span only (not the knowledge/question prefix), i.e. exactly what "mean
token log-probability" baselines in the hallucination-detection literature
use (e.g. as a component of MSP-style baselines).
"""
import os, gc
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "<an external drive>", an external drive on the
#  authors' Mac, which does not exist on other machines.)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_CSV = "phase_perhead/logprob_baseline_features.csv"
SUBSET_SIZE = 200
MAX_SEQ_LEN = 256
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def extract():
    if os.path.exists(OUT_CSV):
        print(f"Found cached: {OUT_CSV}")
        return pd.read_csv(OUT_CSV)

    from datasets import load_dataset
    dataset = load_dataset("pminervini/HaluEval", "qa", split=f"data[:{SUBSET_SIZE}]")
    halueval_df = pd.DataFrame(dataset)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, local_files_only=True,
    ).to(device)
    model.eval()

    rows = []
    for idx, halu_row in halueval_df.iterrows():
        for ans_type, label in [("right_answer", "grounded"), ("hallucinated_answer", "hallucinated")]:
            prefix = f"Knowledge: {halu_row['knowledge']}\nQuestion: {halu_row['question']}\nAnswer: "
            answer = str(halu_row[ans_type])
            prefix_ids = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN - 32).input_ids
            full_ids = tokenizer(prefix + answer, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids
            n_prefix = prefix_ids.shape[1]
            if full_ids.shape[1] <= n_prefix + 1:
                continue
            full_ids = full_ids.to(device)
            with torch.no_grad():
                logits = model(full_ids).logits[0].float()  # [seq, vocab]
            logprobs = F.log_softmax(logits, dim=-1)
            targets = full_ids[0, 1:]
            tok_logprobs = logprobs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)  # [seq-1]
            ans_logprobs = tok_logprobs[n_prefix - 1:]  # log-probs of answer tokens only
            if ans_logprobs.numel() == 0:
                continue
            mean_lp = float(ans_logprobs.mean().cpu())
            min_lp = float(ans_logprobs.min().cpu())
            ppl = float(np.exp(-mean_lp))
            rows.append({"example_id": idx, "label": label, "n_answer_tokens": ans_logprobs.numel(),
                         "mean_logprob": mean_lp, "min_logprob": min_lp, "perplexity": ppl})
            del logits, logprobs, tok_logprobs, ans_logprobs
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{SUBSET_SIZE}")

    del model
    gc.collect()
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df_out)} rows -> {OUT_CSV}")
    return df_out


def get_oof(X, y, groups, seed=42):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    Xv = X.values if hasattr(X, "values") else X
    for tr, te in skf.split(Xv, y, groups=groups):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        pipe.fit(Xv[tr], y[tr])
        oof[te] = pipe.predict_proba(Xv[te])[:, 1]
    return oof


def main():
    df = extract()
    y = (df["label"] == "hallucinated").astype(int).values
    groups = df["example_id"].values

    results = {}
    for name, cols in [("Mean log-prob only", ["mean_logprob"]),
                        ("Perplexity only", ["perplexity"]),
                        ("Mean log-prob + min log-prob", ["mean_logprob", "min_logprob"])]:
        oof = get_oof(df[cols], y, groups)
        auc = roc_auc_score(y, oof)
        results[name] = auc
        print(f"  {name:35s} AUC={auc:.4f}")

    pd.DataFrame([{"Baseline": k, "AUC": round(v, 4)} for k, v in results.items()]).to_csv(
        "phase_perhead/logprob_baseline_results.csv", index=False)
    print("Saved -> phase_perhead/logprob_baseline_results.csv")


if __name__ == "__main__":
    main()
