"""
hidden_state_probe.py
=====================
Extracts mean-pooled last-layer hidden states from Qwen2.5-3B-Instruct for all
HaluEval QA examples, then evaluates a logistic regression probe under the same
strict grouped cross-validation protocol used for TDA features.

This provides the missing "what actually works" anchor baseline that reviewers
will expect alongside the negative TDA result.

Outputs:
  phase_hs_results/halueval_qwen3b_hidden_states.csv  — per-example hidden state features
  phase_hs_results/hidden_state_probe_results.csv     — AUC + TOST vs TDA baselines
"""

import os, gc, time
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "/Volumes/2TB/hf_cache", an external drive on the
#  authors' Mac, which does not exist on other machines.)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
import os
# HF_HOME intentionally not set here: honour the environment.
# (was hardcoded to "/Volumes/2TB/hf_cache", an external drive on the
#  authors' Mac, which does not exist on other machines.)

MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
HS_CSV       = "phase_hs_results/halueval_qwen3b_hidden_states.csv"
FEATURES_CSV = "phase3_results/train_features.csv"  # Qwen2.5-3B TDA features (for comparison)
OUT_DIR      = "phase_hs_results"
MAX_SEQ_LEN  = 384
PCA_DIMS     = 64

os.makedirs(OUT_DIR, exist_ok=True)

# ── Device: prefer MPS on Apple Silicon, else CPU ─────────────────────────────
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def load_model():
    print(f"Loading {MODEL_NAME} in float16...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        output_hidden_states=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, tokenizer


def extract_hidden_state(model, tokenizer, text):
    """
    Returns mean-pooled last-layer hidden state (shape: [hidden_dim]).
    Also returns last-token hidden state as an alternative probe.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    ).to(device)

    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=False)

    # Last layer hidden states: shape [1, seq_len, hidden_dim]
    last_hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]

    # Mean pool over sequence (excluding padding — no padding here since batch=1)
    mean_pooled = last_hidden.mean(dim=0).float().cpu().numpy()   # [hidden_dim]
    last_token  = last_hidden[-1].float().cpu().numpy()            # [hidden_dim]

    # Also grab middle layer (layer L//2) mean pool for richer feature set
    mid_layer_idx = len(outputs.hidden_states) // 2
    mid_hidden = outputs.hidden_states[mid_layer_idx][0].mean(dim=0).float().cpu().numpy()

    del outputs, last_hidden, inputs
    return mean_pooled, last_token, mid_hidden, seq_len


def build_prompt(row):
    """Same prompt format as the original phase3 TDA extraction pipeline."""
    return (
        f"Knowledge: {row['knowledge']}\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}"
    )


def extract_all_hidden_states():
    """
    Extracts hidden states for all examples in train_features.csv.
    Saves to HS_CSV so we don't need to re-run the model if the script
    is interrupted and restarted.
    """
    if os.path.exists(HS_CSV):
        existing = pd.read_csv(HS_CSV)
        print(f"  Found existing HS file with {len(existing)} rows. Resuming from checkpoint.")
        return existing

    # Load the HaluEval dataset text to reconstruct prompts
    # We use the original TDA features CSV to get example_id and label order
    tda_df = pd.read_csv(FEATURES_CSV)
    print(f"  TDA feature rows: {len(tda_df)}")

    from datasets import load_dataset
    print("  Loading HaluEval QA dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:2000]")
    halueval_df = pd.DataFrame(dataset)
    print(f"  HaluEval rows: {len(halueval_df)}")

    model, tokenizer = load_model()

    rows = []
    hidden_dim = None
    skipped = 0

    for idx, halu_row in halueval_df.iterrows():
        for ans_type, label in [("right_answer", "grounded"), ("hallucinated_answer", "hallucinated")]:
            halu_row_copy = halu_row.copy()
            halu_row_copy["answer"] = halu_row[ans_type]
            text = build_prompt(halu_row_copy)

            try:
                mean_pool, last_tok, mid_pool, seq_len = extract_hidden_state(model, tokenizer, text)
            except Exception as e:
                print(f"  SKIP idx={idx} label={label}: {e}")
                skipped += 1
                continue

            if hidden_dim is None:
                hidden_dim = len(mean_pool)
                print(f"  Hidden dim detected: {hidden_dim}")

            row = {
                "example_id": idx,
                "label": label,
                "seq_len": seq_len,
            }
            # Store mean-pooled and last-token as flat features
            for i, v in enumerate(mean_pool):
                row[f"mean_{i}"] = float(v)
            for i, v in enumerate(last_tok):
                row[f"last_{i}"] = float(v)
            for i, v in enumerate(mid_pool):
                row[f"mid_{i}"] = float(v)
            rows.append(row)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(halueval_df)} examples | Skipped: {skipped}")
            # Checkpoint save
            pd.DataFrame(rows).to_csv(HS_CSV + ".tmp", index=False)

        # Clear GPU/MPS cache every 50 examples
        if (idx + 1) % 50 == 0:
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(HS_CSV, index=False)
    print(f"\n  Saved {len(df_out)} rows to {HS_CSV} (skipped {skipped})")

    # Cleanup model from memory
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return df_out


def run_probe_eval(hs_df, tda_df):
    """
    Runs grouped 10-fold CV with logistic regression on hidden state features
    and compares AUC to TDA baselines from the existing feature CSV.
    """
    print("\n" + "="*70)
    print("HIDDEN STATE PROBE EVALUATION")
    print("="*70)

    y    = (hs_df["label"] == "hallucinated").astype(int).values
    grps = hs_df["example_id"].values

    # ── PCA-compressed hidden state features ────────────────────────────────
    mean_cols = [c for c in hs_df.columns if c.startswith("mean_")]
    last_cols  = [c for c in hs_df.columns if c.startswith("last_")]
    mid_cols   = [c for c in hs_df.columns if c.startswith("mid_")]

    X_mean = hs_df[mean_cols].values
    X_last = hs_df[last_cols].values
    X_mid  = hs_df[mid_cols].values
    X_all  = np.concatenate([X_mean, X_last, X_mid], axis=1)

    print(f"  Raw feature dims: mean={X_mean.shape[1]}, last={X_last.shape[1]}, mid={X_mid.shape[1]}")

    # PCA compression (fit on full dataset — no leakage since we're computing
    # a fixed rotation; fold-level scaling is done inside the LR pipeline)
    pca = PCA(n_components=min(PCA_DIMS, X_all.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_all)
    print(f"  PCA({PCA_DIMS}) explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

    # ── Grouped 10-fold CV ────────────────────────────────────────────────────
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

    def oof_auc(X):
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y, groups=grps):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, oof), oof

    print("\n  Running grouped 10-fold CV on hidden state probes...")
    auc_mean,  oof_mean  = oof_auc(X_mean)
    auc_last,  oof_last  = oof_auc(X_last)
    auc_pca,   oof_pca   = oof_auc(X_pca)
    print(f"  AUC (mean-pool last-layer LR): {auc_mean:.4f}")
    print(f"  AUC (last-token last-layer LR): {auc_last:.4f}")
    print(f"  AUC (PCA-{PCA_DIMS} all-layers LR): {auc_pca:.4f}")

    # ── TDA baseline AUCs from existing data ─────────────────────────────────
    # Re-align tda_df with hs_df by (example_id, label)
    merged = hs_df[["example_id", "label"]].copy()
    merged = merged.merge(
        tda_df[["example_id", "label"] + [c for c in tda_df.columns if "h0" in c or "h1" in c or "seq_len" == c]],
        on=["example_id", "label"], how="inner"
    )
    print(f"\n  Merged rows (HS ∩ TDA): {len(merged)}")

    y_m   = (merged["label"] == "hallucinated").astype(int).values
    grp_m = merged["example_id"].values
    sl_m  = merged["seq_len"].values

    X_0d  = merged[[c for c in merged.columns if "h0" in c]]
    X_1d  = merged[[c for c in merged.columns if "h1" in c]]
    X_1dn = X_1d.div(sl_m, axis=0)
    X_cb  = pd.concat([X_0d, X_1dn], axis=1)

    def oof_auc_pd(X, yy, gg):
        oof = np.zeros(len(yy))
        for tr, te in StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(X, yy, groups=gg):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
            Xtr = X.iloc[tr] if hasattr(X, 'iloc') else X[tr]
            Xte = X.iloc[te] if hasattr(X, 'iloc') else X[te]
            pipe.fit(Xtr, yy[tr])
            oof[te] = pipe.predict_proba(Xte)[:, 1]
        return roc_auc_score(yy, oof)

    auc_0d = oof_auc_pd(X_0d,  y_m, grp_m)
    auc_cb = oof_auc_pd(X_cb,  y_m, grp_m)
    print(f"  AUC (0D TDA):       {auc_0d:.4f}")
    print(f"  AUC (0D+1D* TDA):   {auc_cb:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = pd.DataFrame([
        {"Probe": "Hidden State (mean-pool, last layer)", "AUC": round(auc_mean, 4), "Dim": X_mean.shape[1]},
        {"Probe": "Hidden State (last-token, last layer)", "AUC": round(auc_last, 4), "Dim": X_last.shape[1]},
        {"Probe": f"Hidden State (PCA-{PCA_DIMS}, all layers)", "AUC": round(auc_pca, 4), "Dim": PCA_DIMS},
        {"Probe": "TDA 0D Only (attention MST)",             "AUC": round(auc_0d, 4), "Dim": X_0d.shape[1]},
        {"Probe": "TDA 0D+1D Normalized (combined)",         "AUC": round(auc_cb, 4), "Dim": X_cb.shape[1]},
    ])
    results.to_csv(f"{OUT_DIR}/hidden_state_probe_results.csv", index=False)
    print(f"\n  ✓ Saved → {OUT_DIR}/hidden_state_probe_results.csv")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# HIDDEN STATE PROBE EXPERIMENT")
    print("#"*70)

    hs_df  = extract_all_hidden_states()
    tda_df = pd.read_csv(FEATURES_CSV)
    run_probe_eval(hs_df, tda_df)
