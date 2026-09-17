"""
evaluate.py -- the single evaluator for the sink-reduction study.

Protocol (applies to every number produced here):
  * StandardScaler + L2 logistic regression. C = 1.0 for banks with <= 64 dims (the
    legacy canonical setting); for wider banks C is chosen from {0.01, 0.1, 1.0} by an
    inner 3-fold grouped CV on the training fold only.
  * Outer: StratifiedGroupKFold(10), repeated over R = 3 seeds; out-of-fold
    probabilities are averaged over the repeats; the seed-to-seed SD of the pooled
    AUC is reported alongside.
  * Differences: 10,000-resample question-level cluster bootstrap on the
    seed-averaged predictions; TOST read off the 90% percentile interval
    (equivalently max one-sided tail < 0.05); TOST power at true delta = 0 from the
    bootstrap SE.
  * Paired benchmarks (one grounded + one hallucinated row per question) also report
    within-pair accuracy: P(score_hallucinated > score_grounded).

  python -m sinktda.evaluate                 # all settings found in sinktda_out/
  python -m sinktda.evaluate truthfulqa_qwen3b
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack, csr_matrix
from scipy.stats import norm, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.environ.get("SINKTDA_OUT", "sinktda_out")
RES = "sinktda_results"
SEEDS = (42, 43, 44)
N_SPLITS = 10
N_BOOT = 10000
EPS = (0.010, 0.015, 0.020, 0.025)
C_GRID = (0.01, 0.1, 1.0)
N_JOBS = int(os.environ.get("SINKTDA_JOBS", "3"))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_setting(name):
    d = os.path.join(ROOT, name)
    df = pd.read_parquet(os.path.join(d, "layers.parquet"))
    ph = dict(np.load(os.path.join(d, "perhead.npz"))) if os.path.exists(os.path.join(d, "perhead.npz")) else {}
    hid = np.load(os.path.join(d, "hidden.npy")).astype(np.float32)
    df["y"] = (df["label"] == "hallucinated").astype(int)
    return df, ph, hid


def n_layers(df):
    return 1 + max(int(c.split("_")[1]) for c in df.columns if c.startswith("layer_"))


def cols(df, keys):
    L = n_layers(df)
    return [f"layer_{l}_{k}" for l in range(L) for k in keys]


def banks(df, ph, hid):
    """name -> dense feature matrix (or 'LEX' sentinel)."""
    N = df["seq_len"].values[:, None].astype(float)
    A = np.maximum(df["answer_len"].values[:, None].astype(float), 1.0)
    h1k = ["h1_max_lifetime", "h1_total_persistence", "h1_max_birth", "h1_max_death"]
    B = {
        "LEN": df[["seq_len", "answer_len"]].values.astype(float),
        "0D": df[cols(df, ["h0_max_lifetime", "h0_total_persistence"])].values,
        "SINK": df[cols(df, ["star_max", "star_tot"])].values,
        "1D": df[cols(df, h1k)].values / N,
        "DEFL": np.hstack([df[cols(df, ["defl_h0_max_lifetime", "defl_h0_total_persistence"])].values,
                           df[cols(df, ["defl_" + k for k in h1k])].values / N]),
        "ANS": np.hstack([df[cols(df, ["ans_h0_max_lifetime", "ans_h0_total_persistence"])].values,
                          df[cols(df, ["ans_" + k for k in h1k])].values / A]),
        "ROWSTAT": df[cols(df, ["entropy_mean", "rowmax_mean"])].values,
        "LOGPROB": df[["lp_mean", "lp_min", "lp_sum", "ent_mean", "ent_max", "msp_score"]].values,
        "HIDDEN": hid[:, 0, :],
        "LEX": None,
    }
    B["LEGACY_MSTPROXY"] = df[cols(df, ["h0_total_persistence"])].values
    if ph:
        n = len(df)
        B["LLMCHECK"] = ph["ph_llmcheck"].sum(-1)
        B["LOOKBACK"] = ph["ph_lookback"].reshape(n, -1)
        B["PH_0D"] = np.hstack([ph["ph_h0_tot"].reshape(n, -1), ph["ph_h0_max"].reshape(n, -1)])
        B["PH_0DTOT"] = ph["ph_h0_tot"].reshape(n, -1)
        B["PH_SINK"] = ph["ph_sink_mass"].reshape(n, -1)
        B["PH_ENT"] = ph["ph_entropy"].reshape(n, -1)
    for k, v in list(B.items()):
        if v is not None:
            B[k] = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return B


def combine(B, names):
    return np.hstack([B[n] for n in names])


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _splitter(groups, seed, k):
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)


def _fit_dense(X, y, g, tr, seed):
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    C = 1.0
    if X.shape[1] > 64:
        best = -1
        for c in C_GRID:
            aucs = []
            for itr, ite in _splitter(gtr, seed, 3).split(Xtr, ytr, gtr):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                m.fit(Xtr[itr], ytr[itr])
                aucs.append(fast_auc(ytr[ite], m.predict_proba(Xtr[ite])[:, 1]))
            if np.mean(aucs) > best:
                best, C = np.mean(aucs), c
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    m.fit(Xtr, ytr)
    return m


def _lex_matrix(texts_tr, texts_te):
    w = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=2000, sublinear_tf=True)
    Xtr = hstack([w.fit_transform(texts_tr), c.fit_transform(texts_tr)]).tocsr()
    Xte = hstack([w.transform(texts_te), c.transform(texts_te)]).tocsr()
    return Xtr, Xte


def oof_predictions(X, y, g, texts=None, use_lex=False):
    """Seed-averaged OOF probabilities + per-seed pooled AUCs."""
    preds, seed_aucs = [], []
    for seed in SEEDS:
        oof = np.zeros(len(y))
        for tr, te in _splitter(g, seed, N_SPLITS).split(np.zeros(len(y)), y, g):
            if use_lex:
                Ltr, Lte = _lex_matrix(texts[tr], texts[te])
                if X is not None:
                    sc = StandardScaler().fit(X[tr])
                    Ltr = hstack([Ltr, csr_matrix(sc.transform(X[tr]))]).tocsr()
                    Lte = hstack([Lte, csr_matrix(sc.transform(X[te]))]).tocsr()
                m = LogisticRegression(C=1.0, max_iter=3000).fit(Ltr, y[tr])
                oof[te] = m.predict_proba(Lte)[:, 1]
            else:
                m = _fit_dense(X, y, g, tr, seed)
                oof[te] = m.predict_proba(X[te])[:, 1]
        preds.append(oof)
        seed_aucs.append(fast_auc(y, oof))
    return np.mean(preds, 0), np.array(seed_aucs)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def fast_auc(y, s):
    y = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_pair_accuracy(df, s):
    t = pd.DataFrame({"g": df["example_id"].values, "y": df["y"].values, "s": s})
    piv = t.pivot_table(index="g", columns="y", values="s", aggfunc="first").dropna()
    if piv.shape[1] < 2 or len(piv) == 0:
        return np.nan
    return float(((piv[1] > piv[0]) + 0.5 * (piv[1] == piv[0])).mean())


def bootstrap_delta(y, pA, pB, g, seed=0, n_boot=N_BOOT):
    rng = np.random.RandomState(seed)
    ug, inv = np.unique(g, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(ug))]
    lens = np.array([len(r) for r in rows])
    flat = np.concatenate(rows)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        idx = np.concatenate([flat[starts[k]:starts[k] + lens[k]] for k in pick]) if lens.max() > 1 else flat[pick]
        diffs[b] = fast_auc(y[idx], pB[idx]) - fast_auc(y[idx], pA[idx])
    diffs = diffs[np.isfinite(diffs)]
    d = fast_auc(y, pB) - fast_auc(y, pA)
    se = float(diffs.std())
    out = dict(delta=d, se=se, ci95_lo=np.percentile(diffs, 2.5), ci95_hi=np.percentile(diffs, 97.5),
               ci90_lo=np.percentile(diffs, 5), ci90_hi=np.percentile(diffs, 95),
               p_superiority=float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())), n_groups=len(ug))
    for e in EPS:
        p = max((diffs <= -e).mean(), (diffs >= e).mean())
        out[f"tost_p_{e}"] = float(p)
        out[f"equiv_{e}"] = bool(p < 0.05)
        out[f"tost_power_{e}"] = float(max(0.0, 2 * norm.cdf(e / se - norm.ppf(0.95)) - 1)) if se > 0 else 1.0
    return out


# ---------------------------------------------------------------------------
# theory checks on real attention
# ---------------------------------------------------------------------------
def theory_checks(name, df):
    L = n_layers(df)
    N = df["seq_len"].values
    rows = []
    viol_h1 = viol_p0 = viol_max = 0
    for l in range(L):
        P0 = df[f"layer_{l}_h0_total_persistence"].values
        S = df[f"layer_{l}_star_tot"].values
        d0 = df[f"layer_{l}_delta0"].values
        dm = df[f"layer_{l}_delta_min"].values
        h1m = df[f"layer_{l}_h1_max_lifetime"].values
        h1t = df[f"layer_{l}_h1_total_persistence"].values
        viol_h1 += int((h1m > d0 + 1e-5).sum())
        viol_p0 += int(((P0 > S + 1e-3) | (P0 < S - (N - 1) * d0 - 1e-3)).sum())
        M0 = df[f"layer_{l}_h0_max_lifetime"].values
        Smax = df[f"layer_{l}_star_max"].values
        viol_max += int(((M0 > Smax + 1e-5) | (M0 < Smax - d0 - 1e-5)).sum())
        # length-normalized: P0/(N-1) vs S0/(N-1) = 1 - mean sink attention
        P0n, Sn = P0 / np.maximum(N - 1, 1), S / np.maximum(N - 1, 1)
        rows.append(dict(setting=name, layer=l, depth=l / max(L - 1, 1),
                         frac_coned=float((d0 <= 1e-7).mean()),
                         median_delta0=float(np.median(d0)), median_delta_min=float(np.median(dm)),
                         frac_apex_bos=float((df[f"layer_{l}_delta_argmin"].values == 0).mean()),
                         mean_sink_mass=float(df[f"layer_{l}_sink_mass"].mean()),
                         frac_h1_zero=float((h1t == 0).mean()),
                         rel_gap=float(np.median((S - P0) / np.maximum(S, 1e-9))),
                         pearson_P0_star=float(np.corrcoef(P0, S)[0, 1]) if P0.std() > 0 and S.std() > 0 else np.nan,
                         spearman_P0_star=float(spearmanr(P0, S)[0]) if P0.std() > 0 and S.std() > 0 else np.nan,
                         spearman_P0_star_norm=float(spearmanr(P0n, Sn)[0]) if P0n.std() > 0 and Sn.std() > 0 else np.nan,
                         spearman_star_N=float(spearmanr(S, N)[0]) if S.std() > 0 and N.std() > 0 else np.nan))
    lay = pd.DataFrame(rows)
    d0_all = df[cols(df, ["delta0"])].values
    dm_all = df[cols(df, ["delta_min"])].values
    h1_all = df[cols(df, ["h1_total_persistence"])].values
    summ = dict(setting=name, n_rows=len(df), layers=L, mean_N=float(N.mean()),
                cells=int(d0_all.size), bound_violations_h1=viol_h1, bound_violations_p0=viol_p0,
                bound_violations_maxdeath=viol_max,
                frac_cells_coned=float((d0_all <= 1e-7).mean()),
                frac_cells_coned_any_apex=float((dm_all <= 1e-7).mean()),
                mean_sink_mass=float(df[cols(df, ["sink_mass"])].values.mean()),
                frac_cells_h1_zero=float((h1_all == 0).mean()),
                frac_h1zero_given_coned=float((h1_all[d0_all <= 1e-7] == 0).mean()) if (d0_all <= 1e-7).any() else np.nan,
                median_delta0=float(np.median(d0_all)),
                median_layer_spearman_P0_star=float(lay["spearman_P0_star"].median()),
                median_layer_spearman_P0_star_norm=float(lay["spearman_P0_star_norm"].median()),
                min_layer_spearman_P0_star_norm=float(lay["spearman_P0_star_norm"].min()),
                median_layer_rel_gap=float(lay["rel_gap"].median()),
                cell_spearman_delta0_h1=float(spearmanr(d0_all.ravel(), h1_all.ravel())[0]))
    return summ, lay


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
BANK_LIST = ["LEN", "0D", "SINK", "1D", "DEFL", "ANS", "ROWSTAT", "LOGPROB", "HIDDEN", "LEX",
             "LLMCHECK", "LOOKBACK", "PH_0D", "PH_0DTOT", "PH_SINK", "PH_ENT"]
COMBOS = {
    "0D+SINK": ["0D", "SINK"], "0D+1D": ["0D", "1D"], "SINK+DEFL": ["SINK", "DEFL"],
    "SINK+ANS": ["SINK", "ANS"], "SINK+1D": ["SINK", "1D"],
    "PH_SINK+PH_0DTOT": ["PH_SINK", "PH_0DTOT"],
    "NONTOPO": ["LOGPROB", "HIDDEN", "LOOKBACK", "LLMCHECK", "ROWSTAT"],
    "NONTOPO+0D": ["LOGPROB", "HIDDEN", "LOOKBACK", "LLMCHECK", "ROWSTAT", "0D"],
    "NONTOPO+DEFL": ["LOGPROB", "HIDDEN", "LOOKBACK", "LLMCHECK", "ROWSTAT", "DEFL"],
    "LOGPROB+0D": ["LOGPROB", "0D"], "HIDDEN+0D": ["HIDDEN", "0D"],
}
LEX_COMBOS = {"LEX+0D": "0D", "LEX+SINK": "SINK", "LEX+DEFL": "DEFL"}
COMPARISONS = [
    ("T1_reduction", "0D", "SINK"), ("T1_legacy_proxy", "0D", "LEGACY_MSTPROXY"),
    ("T2_topo_beyond_sink", "SINK", "0D+SINK"),
    ("T3_deflated_beyond_sink", "SINK", "SINK+DEFL"),
    ("T4_1D_beyond_0D", "0D", "0D+1D"), ("T4b_1D_beyond_sink", "SINK", "SINK+1D"),
    ("T6_answer_only_beyond_sink", "SINK", "SINK+ANS"),
    ("PH_reduction", "PH_0DTOT", "PH_SINK"), ("PH_topo_beyond_sink", "PH_SINK", "PH_SINK+PH_0DTOT"),
    ("PH_0D_vs_entropy", "PH_0D", "PH_ENT"),
    ("B_logprob", "0D", "LOGPROB"), ("B_hidden", "0D", "HIDDEN"), ("B_lookback", "0D", "LOOKBACK"),
    ("B_llmcheck", "0D", "LLMCHECK"), ("B_lex", "0D", "LEX"), ("B_rowstat", "0D", "ROWSTAT"),
    ("B_len", "0D", "LEN"),
    ("I_0D_beyond_nontopo", "NONTOPO", "NONTOPO+0D"), ("I_defl_beyond_nontopo", "NONTOPO", "NONTOPO+DEFL"),
    ("I_0D_beyond_lex", "LEX", "LEX+0D"), ("I_sink_beyond_lex", "LEX", "LEX+SINK"),
    ("I_defl_beyond_lex", "LEX", "LEX+DEFL"),
    ("I_0D_beyond_logprob", "LOGPROB", "LOGPROB+0D"), ("I_0D_beyond_hidden", "HIDDEN", "HIDDEN+0D"),
]


def _run_bank(key, X, y, g, texts, lex_extra=None):
    if key == "LEX":
        return key, oof_predictions(None, y, g, texts, use_lex=True)
    if key in LEX_COMBOS:
        return key, oof_predictions(lex_extra, y, g, texts, use_lex=True)
    return key, oof_predictions(X, y, g)


def evaluate_setting(name):
    df, ph, hid = load_setting(name)
    y = df["y"].values
    g = df["example_id"].values
    texts = df["answer"].astype(str).values
    B = banks(df, ph, hid)
    for k, v in COMBOS.items():
        if all(n in B for n in v):
            B[k] = combine(B, v)
    jobs = []
    for k, v in B.items():
        if k in ("LEX",) or v is not None:
            jobs.append((k, v, None))
    for k, base in LEX_COMBOS.items():
        jobs.append((k, None, B[base]))
    out = Parallel(n_jobs=N_JOBS)(delayed(_run_bank)(k, X, y, g, texts, extra) for k, X, extra in jobs)
    preds = {k: p for k, (p, _) in out}
    seed_aucs = {k: a for k, (_, a) in out}
    paired = df.groupby("example_id").size().max() == 2
    rows = []
    for k, p in preds.items():
        rows.append(dict(setting=name, bank=k, dim=(B[k].shape[1] if B.get(k) is not None else np.nan),
                         auc=fast_auc(y, p), auc_seed_sd=float(seed_aucs[k].std()),
                         within_pair_acc=within_pair_accuracy(df, p) if paired else np.nan,
                         n_rows=len(y), n_groups=len(np.unique(g)), pos_rate=float(y.mean())))
    auc_df = pd.DataFrame(rows)
    comp_rows = Parallel(n_jobs=N_JOBS)(
        delayed(_comp)(name, tag, a, b, y, preds[a], preds[b], g)
        for tag, a, b in COMPARISONS if a in preds and b in preds)
    comp_df = pd.DataFrame(comp_rows)
    np.savez_compressed(os.path.join(RES, "oof", f"{name}.npz"), y=y, g=g, **preds)
    summ, lay = theory_checks(name, df)
    return auc_df, comp_df, summ, lay


def _comp(name, tag, a, b, y, pa, pb, g):
    r = dict(setting=name, test=tag, A=a, B=b, auc_A=fast_auc(y, pa), auc_B=fast_auc(y, pb))
    r.update(bootstrap_delta(y, pa, pb, g))
    return r


def main():
    os.makedirs(os.path.join(RES, "oof"), exist_ok=True)
    if sys.argv[1:2] == ["--theory-only"]:
        for name in sys.argv[2:]:
            df = load_setting(name)[0]
            summ, lay = theory_checks(name, df)
            pd.DataFrame([summ]).to_csv(os.path.join(RES, f"theory_{name}.csv"), index=False)
            lay.to_csv(os.path.join(RES, f"theory_layers_{name}.csv"), index=False)
            print(summ)
        return
    names = sys.argv[1:] or sorted(d for d in os.listdir(ROOT)
                                   if os.path.exists(os.path.join(ROOT, d, "layers.parquet")))
    for name in names:
        print(f"[eval] {name}", flush=True)
        auc_df, comp_df, summ, lay = evaluate_setting(name)
        auc_df.to_csv(os.path.join(RES, f"auc_{name}.csv"), index=False)
        comp_df.to_csv(os.path.join(RES, f"comp_{name}.csv"), index=False)
        pd.DataFrame([summ]).to_csv(os.path.join(RES, f"theory_{name}.csv"), index=False)
        lay.to_csv(os.path.join(RES, f"theory_layers_{name}.csv"), index=False)
        print(auc_df[["bank", "dim", "auc", "auc_seed_sd", "within_pair_acc"]].to_string(index=False))
        print(comp_df[["test", "auc_A", "auc_B", "delta", "ci90_lo", "ci90_hi", "equiv_0.015"]].to_string(index=False))
        print(summ, flush=True)


if __name__ == "__main__":
    main()
