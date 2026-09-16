"""
compute_lexical_confound.py
============================
Quantifies a second, independent confound in HaluEval QA: grounded and
hallucinated answers are separable by pure surface-form features of the
ANSWER TEXT ALONE (no model internals, no attention, no question/knowledge
context) -- both on the full benchmark and on the paper's own standard
token-length-matched subset, showing that length-matching (which balances
total *tokenized prompt+answer* length) does not control for this.

This corroborates concurrent, independent benchmark-artifact critiques
(Hussain & Kantarcioglu, 2026 "PARALLAX"; Janiak et al., 2025 "The Illusion
of Progress") specifically for the attention-graph TDA literature's primary
evaluation benchmark.

Outputs: master_results/lexical_confound_results.csv
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def get_length_matched_indices(df, length_col="seq_len", label_col="label", bin_size=10, random_state=42):
    rng = np.random.RandomState(random_state)
    df_temp = df.copy().reset_index(drop=True)
    df_temp["length_bin"] = (df_temp[length_col] // bin_size) * bin_size
    matched = []
    for b in sorted(df_temp["length_bin"].unique()):
        bin_df = df_temp[df_temp["length_bin"] == b]
        g_idx = bin_df[bin_df[label_col] == "grounded"].index.tolist()
        h_idx = bin_df[bin_df[label_col] == "hallucinated"].index.tolist()
        m = min(len(g_idx), len(h_idx))
        if m > 0:
            rng.shuffle(g_idx); rng.shuffle(h_idx)
            matched.extend(g_idx[:m]); matched.extend(h_idx[:m])
    return np.array(matched)


def cv_auc(X, y, groups):
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def halueval_lexical_confound():
    tda = pd.read_csv("phase3_results/train_features.csv")[["example_id", "label", "seq_len"]]
    ds = load_dataset("pminervini/HaluEval", "qa", split="data[:2000]")
    hdf = pd.DataFrame(ds)
    rows = []
    for idx, r in hdf.iterrows():
        for col, lab in [("right_answer", "grounded"), ("hallucinated_answer", "hallucinated")]:
            rows.append({"example_id": idx, "label": lab, "answer": r[col]})
    text_df = pd.DataFrame(rows)
    merged = tda.merge(text_df, on=["example_id", "label"], how="inner")

    results = []
    for subset_name, sub in [
        ("Full", merged),
        ("Length-Matched (token seq_len, bin=10)", merged.iloc[get_length_matched_indices(merged, bin_size=10)].reset_index(drop=True)),
    ]:
        y = (sub["label"] == "hallucinated").astype(int).values
        g = sub["example_id"].values

        lens = sub["answer"].str.len().values.reshape(-1, 1).astype(float)
        auc_len = cv_auc(lens, y, g)

        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(sub["answer"])
        auc_tfidf = cv_auc(X_tfidf, y, g)

        results.append({
            "Dataset": "HaluEval QA", "Subset": subset_name, "N": len(sub),
            "AUC (answer char-length only)": round(auc_len, 4),
            "AUC (TF-IDF answer text only)": round(auc_tfidf, 4),
        })
        print(f"[HaluEval QA / {subset_name}] N={len(sub)}  char-length AUC={auc_len:.4f}  TF-IDF AUC={auc_tfidf:.4f}")

    return results


def truthfulqa_lexical_confound():
    """Sanity check: does the same artifact appear in TruthfulQA? (best_answer vs
    one sampled incorrect_answer, matching the phase10 extraction protocol)."""
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(ds)
    rows = []
    for idx, r in df.iterrows():
        bad = r["incorrect_answers"][0] if len(r["incorrect_answers"]) > 0 else "I don't know."
        rows.append({"example_id": idx, "label": "grounded", "answer": r["best_answer"]})
        rows.append({"example_id": idx, "label": "hallucinated", "answer": bad})
    sub = pd.DataFrame(rows)
    y = (sub["label"] == "hallucinated").astype(int).values
    g = sub["example_id"].values

    lens = sub["answer"].str.len().values.reshape(-1, 1).astype(float)
    auc_len = cv_auc(lens, y, g)
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(sub["answer"])
    auc_tfidf = cv_auc(X_tfidf, y, g)
    print(f"[TruthfulQA / Full] N={len(sub)}  char-length AUC={auc_len:.4f}  TF-IDF AUC={auc_tfidf:.4f}")
    return [{
        "Dataset": "TruthfulQA", "Subset": "Full", "N": len(sub),
        "AUC (answer char-length only)": round(auc_len, 4),
        "AUC (TF-IDF answer text only)": round(auc_tfidf, 4),
    }]


if __name__ == "__main__":
    results = halueval_lexical_confound() + truthfulqa_lexical_confound()
    pd.DataFrame(results).to_csv("master_results/lexical_confound_results.csv", index=False)
    print("\n✓ Saved master_results/lexical_confound_results.csv")
