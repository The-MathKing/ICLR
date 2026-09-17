"""
per_head_tda_ablation.py
========================
Tests whether per-head (non-averaged) attention topology carries signal missed
by head-averaged TDA. This directly addresses the "head-averaging discards
per-head topology" reviewer objection.

Strategy:
  - Load Qwen2.5-3B-Instruct
  - For a balanced subset of HaluEval QA (N=400 examples = 200 questions × 2 labels)
    that already have head-averaged TDA features, re-extract attention with per-head storage
  - Compute 0D + 1D persistence for EACH attention head individually at layers 16-27
    (mid-late layers where head-averaged 0D peaked: AUC ~0.785)
  - Evaluate grouped 10-fold CV:
      a) head-averaged 0D (baseline, should match published results on subset)
      b) per-head 0D concatenation (all heads × layers)
      c) best single head (oracle upper bound)
  - Report TOST equivalence between head-averaged and per-head

Outputs:
  phase_perhead/per_head_tda_results.csv
"""

import os, gc, time
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "<an external drive>", an external drive on the
#  authors' Mac, which does not exist on other machines.)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ripser
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
import os
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "<an external drive>", an external drive on the
#  authors' Mac, which does not exist on other machines.)

MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"

OUT_DIR      = "phase_perhead"
SUBSET_SIZE  = 200        # number of questions (→ 400 rows: grounded + hallucinated)
MAX_SEQ_LEN  = 256        # keep short for speed; per-head TDA is expensive
LAYER_START  = 18         # mid-late layers (peak performance zone from layerwise analysis)
LAYER_END    = 27         # inclusive (Qwen has 36 layers total)
N_BOOTSTRAP  = 5000
EPS          = 0.015

os.makedirs(OUT_DIR, exist_ok=True)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


def extract_features_from_attn(attn_matrix):
    """Compute 0D and 1D persistence features from a single [N,N] attention matrix."""
    N = attn_matrix.shape[0]
    W = np.maximum(attn_matrix, attn_matrix.T)
    D = 1.0 - W
    np.fill_diagonal(D, 0.0)
    D = D.astype(np.float32)

    try:
        diagrams = ripser.ripser(D, maxdim=1, distance_matrix=True)["dgms"]
    except Exception:
        return {"h0_total": 0.0, "h1_total": 0.0, "h1_count": 0}

    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf]
    h0_total = float(np.sum(h0_finite[:, 1] - h0_finite[:, 0])) if len(h0_finite) > 0 else 0.0

    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf] if len(h1) > 0 else np.array([])
    h1_total = float(np.sum(h1_finite[:, 1] - h1_finite[:, 0])) if len(h1_finite) > 0 else 0.0
    h1_count = len(h1_finite)

    return {"h0_total": h0_total, "h1_total": h1_total, "h1_count": h1_count}


def run_extraction():
    out_csv = f"{OUT_DIR}/per_head_features.csv"
    if os.path.exists(out_csv):
        print(f"  Found cached per-head features: {out_csv}")
        return pd.read_csv(out_csv)

    from datasets import load_dataset
    dataset = load_dataset("pminervini/HaluEval", "qa", split=f"data[:{SUBSET_SIZE}]")
    halueval_df = pd.DataFrame(dataset)

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()

    n_heads = model.config.num_attention_heads  # 16 for Qwen2.5-3B
    layers_of_interest = list(range(LAYER_START, min(LAYER_END + 1, model.config.num_hidden_layers)))
    print(f"  n_heads={n_heads}, layers_of_interest={layers_of_interest}")

    rows = []
    skipped = 0

    for idx, halu_row in halueval_df.iterrows():
        for ans_type, label in [("right_answer", "grounded"), ("hallucinated_answer", "hallucinated")]:
            prompt = (
                f"Knowledge: {halu_row['knowledge']}\n"
                f"Question: {halu_row['question']}\n"
                f"Answer: {halu_row[ans_type]}"
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LEN, padding=False
            ).to(device)
            seq_len = inputs["input_ids"].shape[1]

            if seq_len < 5:
                skipped += 1
                continue

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            row = {"example_id": idx, "label": label, "seq_len": seq_len}

            # head_avg 0D as sanity check
            head_avg_0d_total = 0.0

            for layer_idx in layers_of_interest:
                # attn_layer: [1, n_heads, seq_len, seq_len]
                attn_layer = outputs.attentions[layer_idx][0].float().cpu().numpy()

                # Head-averaged
                avg_attn = attn_layer.mean(axis=0)  # [seq_len, seq_len]
                avg_feats = extract_features_from_attn(avg_attn)
                row[f"l{layer_idx}_havg_h0"] = avg_feats["h0_total"]
                row[f"l{layer_idx}_havg_h1"] = avg_feats["h1_total"] / max(seq_len, 1)
                head_avg_0d_total += avg_feats["h0_total"]

                # Per-head
                for h in range(n_heads):
                    head_attn = attn_layer[h]  # [seq_len, seq_len]
                    feats = extract_features_from_attn(head_attn)
                    row[f"l{layer_idx}_h{h}_h0"] = feats["h0_total"]
                    row[f"l{layer_idx}_h{h}_h1"] = feats["h1_total"] / max(seq_len, 1)

            row["head_avg_0d_total"] = head_avg_0d_total
            rows.append(row)

            del outputs
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{SUBSET_SIZE} | Skipped: {skipped}")
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)

    del model
    gc.collect()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"  ✓ Saved {len(df_out)} rows → {out_csv}")
    return df_out


def cluster_bootstrap_tost(y, pA, pB, groups, eps=EPS, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    uniq, ginv = np.unique(groups, return_inverse=True)
    ng = len(uniq)
    g2r = [np.where(ginv == i)[0] for i in range(ng)]
    delta_obs = roc_auc_score(y, pB) - roc_auc_score(y, pA)
    boots = []
    for _ in range(n_boot):
        sg = rng.choice(ng, size=ng, replace=True)
        idx = np.concatenate([g2r[g] for g in sg])
        yb = y[idx]
        if len(np.unique(yb)) < 2: continue
        boots.append(roc_auc_score(yb, pB[idx]) - roc_auc_score(yb, pA[idx]))
    boots = np.array(boots)
    ci_lo = np.percentile(boots, 2.5)
    ci_hi = np.percentile(boots, 97.5)
    p_tost = max(np.mean(boots <= -eps), np.mean(boots >= eps))
    equiv  = ci_lo > -eps and ci_hi < eps
    return delta_obs, ci_lo, ci_hi, p_tost, equiv


def get_oof(X, y, groups, seed=42):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups=groups):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        Xte = X.iloc[te] if hasattr(X, "iloc") else X[te]
        pipe.fit(Xtr, y[tr])
        oof[te] = pipe.predict_proba(Xte)[:, 1]
    return oof


def run_eval(df):
    print("\n" + "="*70)
    print("PER-HEAD TDA ABLATION EVALUATION")
    print("="*70)

    y    = (df["label"] == "hallucinated").astype(int).values
    grps = df["example_id"].values

    # Feature sets
    havg_0d_cols  = [c for c in df.columns if c.endswith("_havg_h0")]
    havg_1d_cols  = [c for c in df.columns if c.endswith("_havg_h1")]
    perh_0d_cols  = [c for c in df.columns if c.endswith("_h0") and "havg" not in c]
    perh_1d_cols  = [c for c in df.columns if c.endswith("_h1") and "havg" not in c]

    print(f"  Head-avg 0D cols: {len(havg_0d_cols)} | Per-head 0D cols: {len(perh_0d_cols)}")

    X_havg  = df[havg_0d_cols]
    X_perh  = df[perh_0d_cols]
    X_both  = pd.concat([df[havg_0d_cols], df[havg_1d_cols],
                         df[perh_0d_cols], df[perh_1d_cols]], axis=1)

    oof_havg  = get_oof(X_havg,  y, grps)
    oof_perh  = get_oof(X_perh,  y, grps)
    oof_both  = get_oof(X_both,  y, grps)

    auc_havg  = roc_auc_score(y, oof_havg)
    auc_perh  = roc_auc_score(y, oof_perh)
    auc_both  = roc_auc_score(y, oof_both)

    print(f"\n  AUC (head-averaged 0D, layers {LAYER_START}-{LAYER_END}): {auc_havg:.4f}")
    print(f"  AUC (per-head 0D concatenation):                       {auc_perh:.4f}")
    print(f"  AUC (head-avg + per-head combined):                    {auc_both:.4f}")

    # TOST: head-avg vs per-head
    delta, ci_lo, ci_hi, p_tost, equiv = cluster_bootstrap_tost(y, oof_havg, oof_perh, grps)
    print(f"\n  TOST (head-avg vs per-head 0D):")
    print(f"    ΔAUC = {delta:+.4f} | 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    TOST p (ε=0.015) = {p_tost:.4f} → {'EQUIVALENT' if equiv else 'NOT EQUIVALENT'}")

    results = [
        {"Probe": f"Head-averaged 0D (layers {LAYER_START}-{LAYER_END})", "AUC": round(auc_havg, 4), "N_features": len(havg_0d_cols)},
        {"Probe": "Per-head 0D concatenation",                             "AUC": round(auc_perh, 4), "N_features": len(perh_0d_cols)},
        {"Probe": "Combined (head-avg + per-head 0D + 1D)",                "AUC": round(auc_both, 4), "N_features": X_both.shape[1]},
        {"Probe": "ΔAUC (per-head vs head-avg)", "AUC": round(delta, 4),  "N_features": "-"},
        {"Probe": "TOST p-value (ε=0.015)",      "AUC": round(p_tost, 4), "N_features": "-"},
        {"Probe": "Equivalent?",                  "AUC": str(equiv),      "N_features": "-"},
    ]
    out_df = pd.DataFrame(results)
    out_df.to_csv(f"{OUT_DIR}/per_head_tda_results.csv", index=False)
    print(f"\n  ✓ Saved → {OUT_DIR}/per_head_tda_results.csv")
    return out_df


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# PER-HEAD TDA ABLATION EXPERIMENT")
    print("#"*70)
    df = run_extraction()
    run_eval(df)
