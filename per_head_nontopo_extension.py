"""
per_head_nontopo_extension.py
==============================
Extends per_head_tda_ablation.py's per-head audit (review item B10) with:
  (a) a matched-dimension non-topological per-head baseline (max attention
      weight and entropy per head/layer, same 10 layers x 16 heads = 160 dims
      each, exactly matching per-head 0D's dimensionality), extracted on the
      SAME 200 HaluEval questions (same example_id ordering) used by
      per_head_tda_ablation.py, so the comparison is apples-to-apples.
  (b) proper cluster-bootstrap CIs (not just a TOST p-value) for every
      per-head vs. head-averaged and per-head-topo vs. per-head-nontopo
      comparison.
Outputs: phase_perhead/per_head_nontopo_features.csv, phase_perhead/per_head_full_comparison.csv
"""
import os, gc
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
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = "phase_perhead"
SUBSET_SIZE = 200
MAX_SEQ_LEN = 256
LAYER_START = 18
LAYER_END = 27
N_BOOT = 5000
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def extract_nontopo():
    out_csv = f"{OUT_DIR}/per_head_nontopo_features.csv"
    if os.path.exists(out_csv):
        print(f"Found cached: {out_csv}")
        return pd.read_csv(out_csv)

    from datasets import load_dataset
    dataset = load_dataset("pminervini/HaluEval", "qa", split=f"data[:{SUBSET_SIZE}]")
    halueval_df = pd.DataFrame(dataset)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, attn_implementation="eager", local_files_only=True,
    ).to(device)
    model.eval()
    n_heads = model.config.num_attention_heads
    layers_of_interest = list(range(LAYER_START, min(LAYER_END + 1, model.config.num_hidden_layers)))

    rows = []
    for idx, halu_row in halueval_df.iterrows():
        for ans_type, label in [("right_answer", "grounded"), ("hallucinated_answer", "hallucinated")]:
            prompt = (f"Knowledge: {halu_row['knowledge']}\nQuestion: {halu_row['question']}\nAnswer: {halu_row[ans_type]}")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=False).to(device)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 5:
                continue
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            row = {"example_id": idx, "label": label, "seq_len": seq_len}
            for layer_idx in layers_of_interest:
                attn_layer = outputs.attentions[layer_idx][0].float().cpu().numpy()  # [n_heads, seq, seq]
                for h in range(n_heads):
                    A = attn_layer[h]
                    row_max = A.max(axis=1)  # per-token max attention
                    row[f"l{layer_idx}_h{h}_maxattn"] = float(row_max.mean())
                    eps = 1e-12
                    ent = -(A * np.log(A + eps)).sum(axis=1)
                    row[f"l{layer_idx}_h{h}_entropy"] = float(ent.mean())
            rows.append(row)
            del outputs
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{SUBSET_SIZE}")
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)

    del model
    gc.collect()
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} rows -> {out_csv}")
    return df_out


def cluster_bootstrap_ci(y, pA, pB, groups, n_boot=N_BOOT, seed=42):
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
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, pB[idx]) - roc_auc_score(yb, pA[idx]))
    boots = np.array(boots)
    return delta_obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


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
    df_topo = pd.read_csv(f"{OUT_DIR}/per_head_features.csv")
    df_nontopo = extract_nontopo()

    # align on (example_id, label) to guarantee identical row order/labels
    df_topo = df_topo.sort_values(["example_id", "label"]).reset_index(drop=True)
    df_nontopo = df_nontopo.sort_values(["example_id", "label"]).reset_index(drop=True)
    assert (df_topo["example_id"].values == df_nontopo["example_id"].values).all()
    assert (df_topo["label"].values == df_nontopo["label"].values).all()

    y = (df_topo["label"] == "hallucinated").astype(int).values
    grps = df_topo["example_id"].values

    havg_0d_cols = [c for c in df_topo.columns if c.endswith("_havg_h0")]
    perh_0d_cols = [c for c in df_topo.columns if c.endswith("_h0") and "havg" not in c]
    maxattn_cols = [c for c in df_nontopo.columns if c.endswith("_maxattn")]
    entropy_cols = [c for c in df_nontopo.columns if c.endswith("_entropy")]

    print(f"perh_0d: {len(perh_0d_cols)} dims | maxattn: {len(maxattn_cols)} dims | entropy: {len(entropy_cols)} dims")

    oof_havg = get_oof(df_topo[havg_0d_cols], y, grps)
    oof_perh0d = get_oof(df_topo[perh_0d_cols], y, grps)
    oof_maxattn = get_oof(df_nontopo[maxattn_cols], y, grps)
    oof_entropy = get_oof(df_nontopo[entropy_cols], y, grps)
    oof_nontopo_both = get_oof(pd.concat([df_nontopo[maxattn_cols], df_nontopo[entropy_cols]], axis=1), y, grps)

    results = {}
    for name, oof in [("Head-averaged 0D (10 dims)", oof_havg),
                       ("Per-head 0D (160 dims)", oof_perh0d),
                       ("Per-head max-attn, matched-dim (160 dims)", oof_maxattn),
                       ("Per-head entropy, matched-dim (160 dims)", oof_entropy),
                       ("Per-head max-attn+entropy (320 dims)", oof_nontopo_both)]:
        auc = roc_auc_score(y, oof)
        results[name] = (auc, oof)
        print(f"  {name:45s} AUC={auc:.4f}")

    print("\n--- Superiority comparisons (per-head TOPO vs. matched-dim NON-topo) ---")
    rows = []
    for cmp_name, (oof_a, oof_b) in [
        ("Per-head 0D vs. per-head max-attn (matched dim)", (oof_maxattn, oof_perh0d)),
        ("Per-head 0D vs. per-head entropy (matched dim)", (oof_entropy, oof_perh0d)),
        ("Per-head 0D vs. per-head max-attn+entropy", (oof_nontopo_both, oof_perh0d)),
        ("Per-head 0D vs. head-averaged 0D", (oof_havg, oof_perh0d)),
    ]:
        delta, lo, hi = cluster_bootstrap_ci(y, oof_a, oof_b, grps)
        sig = "SIGNIFICANT (CI excludes 0)" if (lo > 0 or hi < 0) else "not significant (CI includes 0)"
        print(f"  {cmp_name}: delta={delta:+.4f}  95% CI=[{lo:+.4f},{hi:+.4f}]  {sig}")
        rows.append({"Comparison": cmp_name, "delta": delta, "ci_lo": lo, "ci_hi": hi, "verdict": sig})

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/per_head_full_comparison.csv", index=False)
    auc_summary = pd.DataFrame([{"Probe": k, "AUC": round(v[0], 4)} for k, v in results.items()])
    auc_summary.to_csv(f"{OUT_DIR}/per_head_auc_summary.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/per_head_full_comparison.csv, per_head_auc_summary.csv")


if __name__ == "__main__":
    main()
