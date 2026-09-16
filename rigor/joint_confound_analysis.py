"""
rigor/joint_confound_analysis.py
================================
Evaluates Directive E4: Jointly conditions the 1D incremental AUC and predictive
information gain on BOTH the sequence length confound and the answer-level lexical /
surface-form confound (character length, word count, and TF-IDF features).

Tests whether the small, non-zero 1D increment on HaluEval QA (+0.010 to +0.016 AUC)
survives when controlling for surface form artifacts.

Outputs: rigor/results/joint_confound_results.csv
"""
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def run_joint_confound_analysis():
    os.makedirs("rigor/results", exist_ok=True)
    print("Loading HaluEval QA dataset for answer text...")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data[:2000]")
    hdf = pd.DataFrame(ds)
    
    rows = []
    for idx, r in hdf.iterrows():
        for col, lab in [("right_answer", "grounded"), ("hallucinated_answer", "hallucinated")]:
            rows.append({"example_id": int(idx), "label": str(lab), "answer_text": str(r[col])})
    text_df = pd.DataFrame(rows)
    text_df["example_id"] = text_df["example_id"].astype(int)
    text_df["label"] = text_df["label"].astype(str)
    
    # Lexical features: char length, word count, TF-IDF
    text_df["char_len"] = text_df["answer_text"].str.len().astype(float)
    text_df["word_count"] = text_df["answer_text"].apply(lambda s: len(s.split())).astype(float)
    
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    tfidf_mat = tfidf.fit_transform(text_df["answer_text"])
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)
    for k in range(10):
        text_df[f"tfidf_svd_{k}"] = svd_feats[:, k]
        
    lexical_cols = ["char_len", "word_count"] + [f"tfidf_svd_{k}" for k in range(10)]
    
    settings = [
        ("HaluEval QA (Qwen2.5-3B)", "phase3_results/train_features.csv"),
        ("HaluEval QA (Qwen2.5-1.5B)", "phase3_results/train_features_qwen1_5b.csv")
    ]
    
    results = []
    
    for name, path in settings:
        print(f"\nProcessing {name}...")
        df = pd.read_csv(path)
        df["example_id"] = df["example_id"].astype(int)
        df["label"] = df["label"].astype(str)
        
        merged = df.merge(text_df, on=["example_id", "label"], how="inner")
        print(f"Merged samples: {len(merged)}")
        
        y = (merged["label"] == "hallucinated").astype(int).values
        groups = merged["example_id"].values
        seq_len = merged["seq_len"].values.reshape(-1, 1).astype(float)
        
        h0_cols = [c for c in df.columns if "_h0_" in c or c.startswith("h0_")]
        h1_cols = [c for c in df.columns if "_h1_" in c or c.startswith("h1_")]
        
        X_h0 = merged[h0_cols].values
        X_h1 = merged[h1_cols].values
        
        # Fit alpha_hat on sequence length for H1
        log_len = np.log(np.maximum(merged["seq_len"].values, 1))
        h1_tot = merged[[c for c in h1_cols if "tot" in c or "total" in c]].values.mean(axis=1) if any("tot" in c for c in h1_cols) else X_h1.mean(axis=1)
        valid = (h1_tot > 0) & (merged["seq_len"].values > 1)
        alpha_hat = np.polyfit(log_len[valid], np.log(h1_tot[valid] + 1e-12), 1)[0] if valid.sum() > 10 else 1.0
        print(f"Calibrated alpha_hat: {alpha_hat:.3f}")
        
        X_h1_norm = X_h1 / (np.maximum(merged["seq_len"].values.reshape(-1, 1), 1) ** alpha_hat)
        X_lexical = merged[lexical_cols].values
        
        # Cross-validation helper
        def get_oof_auc(feat_blocks):
            X = np.hstack(feat_blocks)
            skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
            oof = np.zeros(len(y))
            for tr, te in skf.split(X, y, groups=groups):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr])
                X_te = scaler.transform(X[te])
                clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
                clf.fit(X_tr, y[tr])
                oof[te] = clf.predict_proba(X_te)[:, 1]
            return roc_auc_score(y, oof)
        
        auc_h0_len = get_oof_auc([X_h0, seq_len])
        auc_h0_len_h1 = get_oof_auc([X_h0, seq_len, X_h1_norm])
        delta_without_lexical = auc_h0_len_h1 - auc_h0_len
        
        auc_h0_len_lex = get_oof_auc([X_h0, seq_len, X_lexical])
        auc_h0_len_lex_h1 = get_oof_auc([X_h0, seq_len, X_lexical, X_h1_norm])
        delta_with_lexical = auc_h0_len_lex_h1 - auc_h0_len_lex
        
        auc_lex_alone = get_oof_auc([X_lexical])
        
        print(f"  AUC(H0 + Len): {auc_h0_len:.4f}")
        print(f"  AUC(H0 + Len + H1*): {auc_h0_len_h1:.4f} -> Delta: {delta_without_lexical:+.4f}")
        print(f"  AUC(Lexical Alone): {auc_lex_alone:.4f}")
        print(f"  AUC(H0 + Len + Lex): {auc_h0_len_lex:.4f}")
        print(f"  AUC(H0 + Len + Lex + H1*): {auc_h0_len_lex_h1:.4f} -> Delta (joint controlled): {delta_with_lexical:+.4f}")
        
        results.append({
            "Setting": name,
            "N_rows": len(merged),
            "N_groups": len(np.unique(groups)),
            "alpha_hat": alpha_hat,
            "AUC_H0_Len": auc_h0_len,
            "AUC_H0_Len_H1": auc_h0_len_h1,
            "Delta_Uncontrolled_Lex": delta_without_lexical,
            "AUC_Lexical_Alone": auc_lex_alone,
            "AUC_H0_Len_Lex": auc_h0_len_lex,
            "AUC_H0_Len_Lex_H1": auc_h0_len_lex_h1,
            "Delta_Joint_Controlled": delta_with_lexical
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("rigor/results/joint_confound_results.csv", index=False)
    print("\nSaved joint confound results to rigor/results/joint_confound_results.csv")
    print(res_df.to_string())

if __name__ == "__main__":
    run_joint_confound_analysis()
