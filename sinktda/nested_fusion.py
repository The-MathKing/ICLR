"""
nested_fusion.py -- late fusion without the stacking leak.

late_fusion.py stacks the stored first-level out-of-fold scores. Those scores are honest
for each row, but the score of a *training* row of the second-level model came from a
first-level model that was fit on folds containing the second-level *test* questions, so
the stacker is trained on features that have seen its test labels. Here the first level is
refit inside every outer training fold:

  for each outer split (tr, te) of the usual grouped repeated CV:
    for each bank X:
      inner grouped K-fold on tr only  -> out-of-fold scores for the rows of tr
      fit on all of tr                 -> scores for te
    second level: logistic regression on logit scores, fit on tr's inner OOF scores,
    applied to te's scores

No score used to train or apply the stacker depends on a label in te. First-level models
are the evaluator's own (_fit_dense: standardize + L2 logistic regression, C tuned by inner
grouped 3-fold CV for banks wider than 64 columns), so the only change is the nesting.

Needs the raw features (layers.parquet, perhead.npz, hidden.npy, toha.npz), so it covers
the settings whose dumps are on this machine.

  python -m sinktda.nested_fusion [setting ...]
Writes sinktda_results/nested_fusion.csv (tests named NLF_<extra>_beyond_<base>).
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sinktda.evaluate import (N_JOBS, N_SPLITS, RES, ROOT, SEEDS, COMBOS, _fit_dense, _splitter,
                              banks, bootstrap_delta, combine, fast_auc, load_setting, merge_csv)
from sinktda.late_fusion import logit

K_INNER = 5
PAIRS = [("NONTOPO", "0D"), ("NONTOPO", "DEFL"), ("NONTOPO", "PH_DELTA"), ("NONTOPO", "TOHA"),
         ("MAXP", "TOHA"), ("PH_SINK", "PH_DELTA")]


def feature_banks(name):
    df, ph, hid = load_setting(name)
    B = banks(df, ph, hid)
    n = len(df)
    out = {"NONTOPO": combine(B, COMBOS["NONTOPO"]), "0D": B["0D"], "DEFL": B["DEFL"]}
    if ph:
        out["PH_SINK"] = B["PH_SINK"]
        out["PH_DELTA"] = np.nan_to_num(ph["ph_delta0"].reshape(n, -1).astype(np.float64))
    f = os.path.join(ROOT, name, "toha.npz")
    if os.path.exists(f):
        z = np.load(f)
        if (z["example_id"] != df["example_id"].values).any():
            raise RuntimeError(f"{name}: toha.npz rows do not align with layers.parquet")
        out["TOHA"] = z["toha"].reshape(n, -1).astype(np.float64)   # SUP_toha in toha.py
        out["MAXP"] = z["maxp"].reshape(n, -1).astype(np.float64)   # SUP_maxp
    return out, df["y"].values, df["example_id"].values


def _stack(S_tr, y_tr, S_te):
    """Second level, exactly as late_fusion.py: standardized L2 logistic regression, C=1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(logit(S_tr), y_tr)
    return m.predict_proba(logit(S_te))[:, 1]


def run(name):
    B, y, g = feature_banks(name)
    pairs = [(a, b) for a, b in PAIRS if a in B and b in B]
    used = sorted({k for p in pairs for k in p})
    P = {p: [] for p in pairs}
    P1 = {p: [] for p in pairs}
    for seed in SEEDS:
        pa = {p: np.zeros(len(y)) for p in pairs}
        pb = {p: np.zeros(len(y)) for p in pairs}
        for tr, te in _splitter(g, seed, N_SPLITS).split(np.zeros(len(y)), y, g):
            inner, test = {}, {}
            for k in used:
                X = B[k]
                s = np.zeros(len(tr))
                for itr, ite in _splitter(g[tr], seed, K_INNER).split(np.zeros(len(tr)), y[tr], g[tr]):
                    m = _fit_dense(X[tr], y[tr], g[tr], itr, seed)
                    s[ite] = m.predict_proba(X[tr][ite])[:, 1]
                inner[k] = s
                test[k] = _fit_dense(X, y, g, tr, seed).predict_proba(X[te])[:, 1]
            for a, b in pairs:
                pa[(a, b)][te] = _stack(inner[a][:, None], y[tr], test[a][:, None])
                pb[(a, b)][te] = _stack(np.column_stack([inner[a], inner[b]]), y[tr],
                                        np.column_stack([test[a], test[b]]))
        for p in pairs:
            P[p].append(pa[p])
            P1[p].append(pb[p])
        print(f"[nested] {name} seed {seed} done", flush=True)
    rows = []
    for a, b in pairs:
        A, Bp = np.mean(P[(a, b)], 0), np.mean(P1[(a, b)], 0)
        r = dict(setting=name, test=f"NLF_{b}_beyond_{a}", A=a, B=f"{a}(+){b}",
                 auc_A=fast_auc(y, A), auc_B=fast_auc(y, Bp))
        r.update(bootstrap_delta(y, A, Bp, g))
        rows.append(r)
        print(name, r["test"], f"{r['delta']:+.4f} [{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}]", flush=True)
    return rows


def main():
    names = sys.argv[1:] or sorted(
        d for d in os.listdir(ROOT)
        if all(os.path.exists(os.path.join(ROOT, d, f)) for f in ("layers.parquet", "hidden.npy", "perhead.npz")))
    out = Parallel(n_jobs=N_JOBS)(delayed(run)(s) for s in names)
    merge_csv(os.path.join(RES, "nested_fusion.csv"), [r for rows in out for r in rows])


if __name__ == "__main__":
    main()
