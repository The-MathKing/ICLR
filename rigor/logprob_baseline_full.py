#!/usr/bin/env python3
"""
logprob_baseline_full.py -- roadmap item 1.

The paper's single strongest baseline (mean + min token log-probability,
AUC 0.984, Table 17) is extracted on a 200-question / N=329-row HaluEval subset
for one model. Nothing blocks running it everywhere: it needs one teacher-forced
forward pass per row, which is the same pass the attention extraction already
does. This script runs it on all seven settings at full N.

Rows are aligned to each setting's existing feature CSV on (example_id, label),
so the resulting AUCs are directly comparable to the 0D/1D numbers in Table 2
rather than being computed over a different row set.

    python rigor/logprob_baseline_full.py                  # all settings
    python rigor/logprob_baseline_full.py --only "TruthfulQA (Mistral-7B)"

Outputs:
    rigor/results/logprob_<setting>.csv       per-row features
    rigor/results/logprob_summary.csv         AUC per setting per feature set
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rigor.device import assert_gpu_usable, empty_cache, get_device, get_dtype, hf_cache_note, report
from rigor import settings as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# NOTE: trust_remote_code is deliberately NOT set. Phi-3's hub-hosted
# modeling_phi3.py reads config.rope_scaling["type"], a key transformers 5.x
# renamed to "rope_type", so remote code raises KeyError: 'type'. All models
# used here have native transformers implementations.
MAX_SEQ_LEN = 1024
SEED = 42


def answer_logprobs(model, tokenizer, prefix, answer, device):
    """Mean/min log P(answer_token_i | prefix, answer_<i>) over the answer span."""
    prefix_ids = tokenizer(prefix, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN - 32).input_ids
    full_ids = tokenizer(prefix + answer, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN).input_ids
    n_prefix = prefix_ids.shape[1]
    if full_ids.shape[1] <= n_prefix + 1:
        return None
    full_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(full_ids).logits[0].float()
    logprobs = F.log_softmax(logits, dim=-1)
    targets = full_ids[0, 1:]
    tok_lp = logprobs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    ans_lp = tok_lp[n_prefix - 1:]
    if ans_lp.numel() == 0:
        return None
    mean_lp = float(ans_lp.mean().cpu())
    out = {
        "mean_logprob": mean_lp,
        "min_logprob": float(ans_lp.min().cpu()),
        "perplexity": float(np.exp(-mean_lp)),
        "n_answer_tokens": int(ans_lp.numel()),
    }
    del logits, logprobs, tok_lp, ans_lp
    return out


def extract_setting(cfg, device):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_csv = os.path.join(OUT_DIR, f"logprob_{S.slug(cfg['name'])}.csv")
    if os.path.exists(out_csv):
        print(f"  cached -> {out_csv}")
        return pd.read_csv(out_csv)

    feat_path = cfg["features"]
    if not os.path.exists(feat_path):
        print(f"  SKIP: feature file not found ({feat_path}); "
              f"run that setting's extraction first.")
        return None
    feats = pd.read_csv(feat_path, usecols=["example_id", "label"])
    wanted = set(zip(feats["example_id"], feats["label"]))
    print(f"  aligning to {len(wanted)} rows from {feat_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=get_dtype(device),
    ).to(device)
    model.eval()

    if cfg["benchmark"] == "halueval":
        ds = load_dataset("pminervini/HaluEval", "qa",
                          split=f"data[:{cfg['n_questions']}]")
    else:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(ds).head(cfg["n_questions"])

    rows = []
    for idx, row in df.iterrows():
        if cfg["benchmark"] == "halueval":
            prefix, good, bad = S.halueval_pair(row)
            pairs = [("grounded", prefix, good), ("hallucinated", prefix, bad)]
        else:
            question, good, bad = S.truthfulqa_pair(row)
            prefix = S.truthfulqa_chat_prefix(tokenizer, question)
            pairs = [("grounded", prefix, good), ("hallucinated", prefix, bad)]

        for label, pfx, ans in pairs:
            if (idx, label) not in wanted:
                continue
            got = answer_logprobs(model, tokenizer, pfx, ans, device)
            if got is None:
                continue
            got.update({"example_id": idx, "label": label})
            rows.append(got)
        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(df)}")
            empty_cache(device)

    del model
    gc.collect()
    empty_cache(device)

    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  saved {len(out)} rows -> {out_csv}")
    return out


def grouped_oof_auc(X, y, groups, seed=SEED):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    oof = np.zeros(len(y))
    for tr, te in skf.split(Xv, y, groups=groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
        pipe.fit(Xv[tr], y[tr])
        oof[te] = pipe.predict_proba(Xv[te])[:, 1]
    return roc_auc_score(y, oof)


FEATURE_SETS = [
    ("Mean log-prob only", ["mean_logprob"]),
    ("Min log-prob only", ["min_logprob"]),
    ("Perplexity only", ["perplexity"]),
    ("Mean + min log-prob", ["mean_logprob", "min_logprob"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single setting by name")
    args = ap.parse_args()

    hf_cache_note()
    device = get_device()
    assert_gpu_usable(device)
    report(device)
    os.makedirs(OUT_DIR, exist_ok=True)

    todo = S.SETTINGS if args.only is None else [S.SETTINGS_BY_NAME[args.only]]
    summary = []
    for cfg in todo:
        print(f"\n=== {cfg['name']} ({cfg['model']}) ===")
        df = extract_setting(cfg, device)
        if df is None or df.empty:
            continue
        y = (df["label"] == "hallucinated").astype(int).values
        groups = df["example_id"].values
        for fs_name, cols in FEATURE_SETS:
            auc = grouped_oof_auc(df[cols], y, groups)
            print(f"  {fs_name:24s} AUC = {auc:.4f}   (N={len(df)}, "
                  f"groups={len(np.unique(groups))})")
            summary.append({"Setting": cfg["name"], "Feature Set": fs_name,
                            "N_rows": len(df), "N_groups": int(len(np.unique(groups))),
                            "AUC": round(float(auc), 4)})

    if summary:
        path = os.path.join(OUT_DIR, "logprob_summary.csv")
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
