"""
rigor/gate_b_lexical_survival.py
================================
GATE B: does the 0D (= MST) feature bank survive the answer-level lexical confound?

The paper is being repositioned around a POSITIVE claim -- that on TruthfulQA, which is
length-balanced, the 0D/MST bank is the strongest cheap signal (it beats log-probability,
MSP and length). That claim is only defensible if 0D adds something a bag-of-words model
does not already have.

This is not obviously true. master_results/lexical_confound_results.csv reports that
TF-IDF on the answer text ALONE reaches AUC 0.830 on TruthfulQA -- HIGHER than 0D's 0.756.
rigor/joint_confound_analysis.py tested the analogous question for the 1D bank but only on
HaluEval, where the answer was devastating (the +0.015 increment collapsed to -0.0002).

So the question this script answers is: after conditioning on answer-level lexical
features, does 0D still contribute? Specifically it compares

    AUC(Lex)          vs   AUC(Lex + 0D)        -> does topology add over bag-of-words?
    AUC(0D)           vs   AUC(0D + Lex)        -> how much of 0D is lexically mediated?

with a group-level cluster-bootstrap CI on the increment, read as a SUPERIORITY interval
(does the CI exclude zero?), not as a TOST equivalence interval.

Reuses master_pipeline.canonical_oof and master_pipeline.cluster_bootstrap_tost so the
protocol is identical to every other number in the paper.

Design note on the strength of the control. A weak lexical control makes 0D look good by
default, so the control here is deliberately strong: TF-IDF(1-2 grams, 2000 features)
reduced by SVD to `--svd_components` dimensions (default 50, versus the 10 used by
joint_confound_analysis.py). The lexical-alone AUC is reported so the reader can see how
much of the known ceiling the control actually captures; if it lands well below the 0.830
reported by compute_lexical_confound.py, the control is too weak and the gate is not
informative.

Usage:
    python rigor/gate_b_lexical_survival.py
    python rigor/gate_b_lexical_survival.py --svd_components 100 --n_boot 2000

Outputs: rigor/results/gate_b_lexical_survival.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from master_pipeline import canonical_oof, cluster_bootstrap_tost  # noqa: E402
from rigor.settings import SETTINGS  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# answer text, reconstructed to match each extraction protocol exactly
# ---------------------------------------------------------------------------
def truthfulqa_answer_text():
    """(example_id, label, answer_text) matching the phase10 extraction protocol.

    Mirrors compute_lexical_confound.truthfulqa_lexical_confound: example_id is the
    row index into the generation/validation split, the grounded answer is
    `best_answer`, and the hallucinated answer is `incorrect_answers[0]`.
    """
    df = pd.DataFrame(load_dataset("truthfulqa/truthful_qa", "generation", split="validation"))
    rows = []
    for idx, r in df.iterrows():
        bad = r["incorrect_answers"][0] if len(r["incorrect_answers"]) > 0 else "I don't know."
        rows.append({"example_id": int(idx), "label": "grounded", "answer_text": str(r["best_answer"])})
        rows.append({"example_id": int(idx), "label": "hallucinated", "answer_text": str(bad)})
    return pd.DataFrame(rows)


def halueval_answer_text(n_questions):
    """(example_id, label, answer_text) matching the HaluEval extraction protocol."""
    ds = load_dataset("pminervini/HaluEval", "qa", split=f"data[:{n_questions}]")
    df = pd.DataFrame(ds)
    rows = []
    for idx, r in df.iterrows():
        rows.append({"example_id": int(idx), "label": "grounded", "answer_text": str(r["right_answer"])})
        rows.append({"example_id": int(idx), "label": "hallucinated", "answer_text": str(r["hallucinated_answer"])})
    return pd.DataFrame(rows)


def lexical_features(text, n_components, seed=42):
    """Char length, word count, and SVD-reduced TF-IDF of the answer text alone."""
    char_len = text.str.len().values.astype(float)
    word_count = text.apply(lambda s: len(s.split())).values.astype(float)

    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    mat = tfidf.fit_transform(text)
    k = int(min(n_components, mat.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    reduced = svd.fit_transform(mat)

    return np.column_stack([char_len, word_count, reduced]), float(svd.explained_variance_ratio_.sum())


# ---------------------------------------------------------------------------
def main(svd_components, n_boot, seed):
    truthful_text = None
    results = []

    for s in SETTINGS:
        name, path = s["name"], s["features"]
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue

        print(f"\n=== {name} ===")
        df = pd.read_csv(path)
        df["example_id"] = df["example_id"].astype(int)
        df["label"] = df["label"].astype(str)

        if s["benchmark"] == "truthfulqa":
            if truthful_text is None:
                truthful_text = truthfulqa_answer_text()
            text_df = truthful_text
        else:
            text_df = halueval_answer_text(s["n_questions"])

        merged = df.merge(text_df, on=["example_id", "label"], how="inner")
        if len(merged) < 100:
            print(f"SKIP {name}: only {len(merged)} rows merged")
            continue

        y = (merged["label"] == "hallucinated").astype(int).values
        groups = merged["example_id"].values

        h0_cols = [c for c in merged.columns if "_h0_" in c]
        X_h0 = merged[h0_cols].values
        X_lex, evr = lexical_features(merged["answer_text"], svd_components, seed)

        print(f"rows={len(merged)} groups={len(np.unique(groups))} "
              f"0D dims={X_h0.shape[1]} lex dims={X_lex.shape[1]} (TF-IDF var explained {evr:.1%})")

        oof_lex = canonical_oof(X_lex, y, groups, seed=seed)
        oof_h0 = canonical_oof(X_h0, y, groups, seed=seed)
        oof_both = canonical_oof(np.hstack([X_lex, X_h0]), y, groups, seed=seed)

        # Does 0D add over bag-of-words? (the gate)
        add_stats, _ = cluster_bootstrap_tost(y, oof_lex, oof_both, groups, n_boot=n_boot, seed=seed)
        # How much of 0D is lexically mediated?
        med_stats, _ = cluster_bootstrap_tost(y, oof_h0, oof_both, groups, n_boot=n_boot, seed=seed)

        auc_lex, auc_h0, auc_both = add_stats["auc_A"], med_stats["auc_A"], add_stats["auc_B"]
        delta = add_stats["delta_obs"]
        lo, hi = add_stats["ci95_low"], add_stats["ci95_high"]
        survives = lo > 0

        print(f"  AUC(Lex)        = {auc_lex:.4f}")
        print(f"  AUC(0D)         = {auc_h0:.4f}")
        print(f"  AUC(Lex + 0D)   = {auc_both:.4f}")
        print(f"  0D over Lex     = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"-> {'SURVIVES' if survives else 'DOES NOT SURVIVE'}")
        print(f"  Lex over 0D     = {med_stats['delta_obs']:+.4f}  "
              f"95% CI [{med_stats['ci95_low']:+.4f}, {med_stats['ci95_high']:+.4f}]")

        results.append({
            "Setting": name,
            "Benchmark": s["benchmark"],
            "N_rows": len(merged),
            "N_groups": int(len(np.unique(groups))),
            "dim_0D": X_h0.shape[1],
            "dim_lex": X_lex.shape[1],
            "tfidf_var_explained": round(evr, 4),
            "AUC_Lex": round(auc_lex, 4),
            "AUC_0D": round(auc_h0, 4),
            "AUC_Lex_0D": round(auc_both, 4),
            "Delta_0D_over_Lex": round(delta, 4),
            "CI95_low": round(lo, 4),
            "CI95_high": round(hi, 4),
            "bootstrap_se": round(add_stats["bootstrap_se"], 5),
            "survives_lexical_control": bool(survives),
            "Delta_Lex_over_0D": round(med_stats["delta_obs"], 4),
            "Lex_over_0D_CI95_low": round(med_stats["ci95_low"], 4),
            "Lex_over_0D_CI95_high": round(med_stats["ci95_high"], 4),
        })

    if not results:
        print("\nNo settings evaluated.")
        return 1

    out = pd.DataFrame(results)
    out.insert(1, "svd_components", svd_components)
    os.makedirs("rigor/results", exist_ok=True)
    # Filename carries the control strength: the gate is only meaningful relative to how
    # strong the lexical control is, so the two must never be conflated.
    out_path = f"rigor/results/gate_b_lexical_survival_svd{svd_components}.csv"
    out.to_csv(out_path, index=False)

    print("\n=== GATE B VERDICT ===")
    print(out[["Setting", "AUC_Lex", "AUC_0D", "AUC_Lex_0D",
               "Delta_0D_over_Lex", "CI95_low", "survives_lexical_control"]].to_string(index=False))

    tqa = out[out["Benchmark"] == "truthfulqa"]
    n_survive = int(tqa["survives_lexical_control"].sum())
    print(f"\nTruthfulQA settings where 0D survives lexical control: {n_survive}/{len(tqa)}")
    if n_survive == len(tqa) and len(tqa) > 0:
        print("PASS: the positive repositioning is supported on every TruthfulQA setting.")
    elif n_survive > 0:
        print("PARTIAL: supported on some settings -- the claim must be scoped to those.")
    else:
        print("FAIL: 0D adds nothing over bag-of-words; the positive claim is unavailable.")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--svd_components", type=int, default=50)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raise SystemExit(main(args.svd_components, args.n_boot, args.seed))
