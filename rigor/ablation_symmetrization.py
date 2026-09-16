#!/usr/bin/env python3
"""
ablation_symmetrization.py -- roadmap item A2.

Closes a limitation the paper names explicitly:

    "We also did not ablate the symmetrization rule (max vs. mean vs.
     lower-triangular) or the distance transform (1-W vs. -log W) beyond the
     single convention used throughout."

These are the only two free parameters of the entire filtration, and Section 3.2
argues the max-symmetrization "is a bookkeeping step, not a filtering choice with
free parameters." That is a claim an ablation should establish rather than assert
-- and it is the most obvious "did you just pick the lucky convention?" question
a reviewer can ask.

Sweeps 3 symmetrizations x 2 distance transforms = 6 cells, reporting 0D AUC
under the canonical StandardScaler + L2 logistic regression + StratifiedGroupKFold
protocol. Attention is extracted once per model and reused across all six cells,
so the cost is one extraction pass, not six.

    python rigor/ablation_symmetrization.py --only "TruthfulQA (Qwen2.5-3B)"
    python rigor/ablation_symmetrization.py                  # all TruthfulQA settings

Output: rigor/results/ablation_symmetrization.csv
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rigor.device import assert_gpu_usable, empty_cache, get_device, get_dtype, hf_cache_note, report
from rigor import settings as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(OUT_DIR, "ablation_symmetrization.csv")
# NOTE: trust_remote_code is deliberately NOT set. Phi-3's hub-hosted
# modeling_phi3.py reads config.rope_scaling["type"], a key transformers 5.x
# renamed to "rope_type", so remote code raises KeyError: 'type'. All models
# used here have native transformers implementations.
MAX_SEQ_LEN = 1024
SEED = 42
EPS = 1e-8

SYMMETRIZATIONS = {
    "max (paper)": lambda A: np.maximum(A, A.T),
    "mean":        lambda A: 0.5 * (A + A.T),
    "lower-tri":   lambda A: np.tril(A) + np.tril(A, -1).T,
}
TRANSFORMS = {
    "1-W (paper)": lambda W: 1.0 - W,
    "-log W":      lambda W: -np.log(np.clip(W, EPS, 1.0)),
}


def mst_weight(D):
    """0D total persistence == MST edge-weight sum (Theorem 1)."""
    D = np.array(D, dtype=float, copy=True)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    return float(minimum_spanning_tree(D).sum())


def extract(cfg, device):
    """One forward pass per row; store the head-averaged attention per layer.

    We keep the raw head-averaged matrices (not the derived features) so all six
    ablation cells are computed from identical attention, isolating the
    convention as the only varying factor.
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = os.path.join(OUT_DIR, f"attn_{S.slug(cfg['name'])}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z["attn"]), pd.DataFrame(z["meta"], columns=["example_id", "label"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=get_dtype(device),
        attn_implementation="eager",
    ).to(device)
    model.eval()

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(ds).head(cfg["n_questions"])

    attn_all, meta = [], []
    for idx, row in df.iterrows():
        question, good, bad = S.truthfulqa_pair(row)
        for label, ans in [("grounded", good), ("hallucinated", bad)]:
            if "Mistral" in cfg["model"]:
                full = S.mistral_inst_full(question, ans)
            else:
                full = S.truthfulqa_chat_full(tokenizer, question, ans)
            inputs = tokenizer(full, return_tensors="pt").to(device)
            if inputs.input_ids.shape[1] > MAX_SEQ_LEN:
                continue
            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            # head-average per layer -> [L, N, N], float32 to keep the cache small
            per_layer = np.stack([
                np.nan_to_num(a.float().cpu().numpy()[0], nan=0.0).mean(axis=0)
                for a in out.attentions
            ]).astype(np.float32)
            attn_all.append(per_layer)
            meta.append((idx, label))
            del out
        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(df)}")
            empty_cache(device)

    del model
    gc.collect()
    empty_cache(device)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(cache, attn=np.array(attn_all, dtype=object),
                        meta=np.array(meta, dtype=object))
    return attn_all, pd.DataFrame(meta, columns=["example_id", "label"])


def bank_for(attn_all, sym_fn, tr_fn):
    """0D feature bank: MST weight per layer."""
    rows = []
    for per_layer in attn_all:
        rows.append([mst_weight(tr_fn(sym_fn(A))) for A in per_layer])
    return np.asarray(rows, dtype=float)


def grouped_auc(X, y, groups, seed=SEED):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    X = np.nan_to_num(X)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups=groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    hf_cache_note()
    device = get_device()
    assert_gpu_usable(device)
    report(device)
    os.makedirs(OUT_DIR, exist_ok=True)

    todo = [c for c in S.SETTINGS if c["benchmark"] == "truthfulqa"]
    if args.only:
        todo = [S.SETTINGS_BY_NAME[args.only]]

    results = []
    for cfg in todo:
        print(f"\n=== {cfg['name']} ===")
        attn_all, meta = extract(cfg, device)
        y = (meta["label"] == "hallucinated").astype(int).values
        groups = meta["example_id"].values
        print(f"  {len(attn_all)} rows, {len(np.unique(groups))} groups")
        for sname, sfn in SYMMETRIZATIONS.items():
            for tname, tfn in TRANSFORMS.items():
                auc = grouped_auc(bank_for(attn_all, sfn, tfn), y, groups)
                tag = "  <- paper convention" if "paper" in sname and "paper" in tname else ""
                print(f"    {sname:12s} x {tname:12s}  0D AUC = {auc:.4f}{tag}")
                results.append({"Setting": cfg["name"], "Symmetrization": sname,
                                "Transform": tname, "AUC_0D": round(float(auc), 4)})
        del attn_all
        gc.collect()

    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
