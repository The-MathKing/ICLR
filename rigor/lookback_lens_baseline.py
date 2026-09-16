#!/usr/bin/env python3
"""
lookback_lens_baseline.py -- roadmap item 8.

Lookback Lens (Chuang et al., 2024) is the single most threatening missing
baseline for this paper, because it is also computed from attention and is
NOT topological. If it matches or beats 0D-MST, the paper's thesis gets
stronger, not weaker: attention graphs do carry hallucination signal, and
persistent homology simply is not the efficient way to extract it.

Feature, per (layer, head): the "lookback ratio" -- the share of that head's
attention mass that an answer token sends back to the prompt/context, rather
than to previously generated answer tokens:

    R[l,h] = mean over answer positions t of
                 sum_{j < prompt_len} A[l,h,t,j]
             ---------------------------------------
                 sum_{j <= t}        A[l,h,t,j]

giving an L x H feature bank, evaluated under the identical StandardScaler +
L2 logistic regression + StratifiedGroupKFold protocol as every other number
in the paper.

    python rigor/lookback_lens_baseline.py --only "TruthfulQA (Qwen2.5-3B)"

Outputs:
    rigor/results/lookback_<setting>.csv
    rigor/results/lookback_summary.csv
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
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
EPS = 1e-9


def lookback_ratios(attentions, prompt_len):
    """attentions: tuple of L tensors, each [1, H, N, N] (causal, row-stochastic).
    Returns a flat dict of L*H lookback ratios."""
    feats = {}
    for l, attn in enumerate(attentions):
        a = attn[0].float()                       # [H, N, N]
        n = a.shape[-1]
        if prompt_len >= n:                       # no answer tokens to score
            for h in range(a.shape[0]):
                feats[f"layer_{l}_head_{h}_lookback"] = 0.0
            continue
        ans = a[:, prompt_len:, :]                # [H, n_ans, N]
        to_prompt = ans[:, :, :prompt_len].sum(dim=-1)   # [H, n_ans]
        total = ans.sum(dim=-1)                          # [H, n_ans]
        ratio = (to_prompt / (total + EPS)).mean(dim=-1)  # [H]
        for h, v in enumerate(ratio.cpu().numpy()):
            feats[f"layer_{l}_head_{h}_lookback"] = float(v)
    return feats


def extract_setting(cfg, device):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_csv = os.path.join(OUT_DIR, f"lookback_{S.slug(cfg['name'])}.csv")
    if os.path.exists(out_csv):
        print(f"  cached -> {out_csv}")
        return pd.read_csv(out_csv)

    if not os.path.exists(cfg["features"]):
        print(f"  SKIP: {cfg['features']} not found.")
        return None
    feats_ref = pd.read_csv(cfg["features"], usecols=["example_id", "label"])
    wanted = set(zip(feats_ref["example_id"], feats_ref["label"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=get_dtype(device),
        attn_implementation="eager",   # REQUIRED: SDPA/flash return no attentions
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
            builds = [("grounded", prefix, prefix + good),
                      ("hallucinated", prefix, prefix + bad)]
        else:
            question, good, bad = S.truthfulqa_pair(row)
            pfx = S.truthfulqa_chat_prefix(tokenizer, question)
            builds = [
                ("grounded", pfx, S.truthfulqa_chat_full(tokenizer, question, good)),
                ("hallucinated", pfx, S.truthfulqa_chat_full(tokenizer, question, bad)),
            ]

        for label, prefix_text, full_text in builds:
            if (idx, label) not in wanted:
                continue
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            if seq_len > MAX_SEQ_LEN:
                continue
            prompt_len = tokenizer(prefix_text, return_tensors="pt").input_ids.shape[1]
            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            feats = lookback_ratios(out.attentions, prompt_len)
            feats.update({"example_id": idx, "label": label, "seq_len": seq_len,
                          "prompt_len": prompt_len})
            rows.append(feats)
            del out
        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(df)}")
            empty_cache(device)

    del model
    gc.collect()
    empty_cache(device)

    out_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  saved {len(out_df)} rows x {out_df.shape[1]} cols -> {out_csv}")
    return out_df


def grouped_oof_auc(X, y, groups, seed=SEED):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    Xv = np.nan_to_num(X.values.astype(float))
    oof = np.zeros(len(y))
    for tr, te in skf.split(Xv, y, groups=groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
        pipe.fit(Xv[tr], y[tr])
        oof[te] = pipe.predict_proba(Xv[te])[:, 1]
    return roc_auc_score(y, oof), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
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
        cols = [c for c in df.columns if c.endswith("_lookback")]
        y = (df["label"] == "hallucinated").astype(int).values
        groups = df["example_id"].values
        auc, _ = grouped_oof_auc(df[cols], y, groups)
        print(f"  Lookback Lens ({len(cols)} dims)  AUC = {auc:.4f}")
        summary.append({"Setting": cfg["name"], "Feature Set": "Lookback Lens",
                        "Dim": len(cols), "N_rows": len(df),
                        "N_groups": int(len(np.unique(groups))),
                        "AUC": round(float(auc), 4)})

    if summary:
        path = os.path.join(OUT_DIR, "lookback_summary.csv")
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
