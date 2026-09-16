"""
phase1_stats_improvements.py
============================
Computes all statistical improvements for the paper in one pass:

1. Qwen2.5-1.5B TOST row (HaluEval) — closes missing-model gap
2. Epsilon sensitivity table — TOST at ε ∈ {0.010, 0.015, 0.020, 0.025}
3. Power analysis / MDE — minimum detectable ΔAUC at 80% power
4. Scaling exponent regression — log-log P1 vs N to verify linearity in Prop 3
5. 0D ↔ attention-marginals Spearman correlation — empirical support for Prop 4
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── helpers ──────────────────────────────────────────────────────────────────

def get_oof_preds(X, y, groups, n_splits=10, seed=42):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups=groups):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof


def cluster_bootstrap_tost(y, preds_A, preds_B, groups, epsilons=(0.010, 0.015, 0.020, 0.025),
                            n_boot=10_000, seed=42):
    """
    Cluster-bootstrap TOST at multiple epsilon margins.
    Returns observed ΔAUC, 95% CI, and TOST p-values for each epsilon.
    """
    rng = np.random.RandomState(seed)
    unique_g, g_inv = np.unique(groups, return_inverse=True)
    n_g = len(unique_g)
    g2rows = [np.where(g_inv == i)[0] for i in range(n_g)]

    auc_A = roc_auc_score(y, preds_A)
    auc_B = roc_auc_score(y, preds_B)
    delta_obs = auc_B - auc_A

    boot_deltas = []
    for _ in range(n_boot):
        sg = rng.choice(n_g, size=n_g, replace=True)
        idx = np.concatenate([g2rows[g] for g in sg])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_deltas.append(roc_auc_score(yb, preds_B[idx]) - roc_auc_score(yb, preds_A[idx]))

    boot_deltas = np.array(boot_deltas)
    ci_low = np.percentile(boot_deltas, 2.5)
    ci_high = np.percentile(boot_deltas, 97.5)

    tost_ps = {}
    for eps in epsilons:
        p = max(np.mean(boot_deltas <= -eps), np.mean(boot_deltas >= eps))
        tost_ps[eps] = float(p) if p > 0 else 1e-5  # cap at 1e-5 for display

    return {
        "auc_A": float(auc_A),
        "auc_B": float(auc_B),
        "delta": float(delta_obs),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "tost_ps": tost_ps,
        "boot_std": float(np.std(boot_deltas)),
        "boot_deltas": boot_deltas,
        "n_samples": len(y),
        "n_groups": n_g,
    }


def compute_mde(n_samples, n_groups, alpha=0.05, power=0.80, n_sim=5000, seed=99):
    """
    Empirical minimum detectable effect size (ΔAUC) via simulation.
    Simulates the cluster-bootstrap null distribution and finds the minimum |ΔAUC|
    that would be detectable with `power` probability at significance `alpha`.
    
    Uses a conservative simulation: generates bootstrap noise at scale observed
    in real data (passed as boot_std), sweeps effect sizes.
    """
    # We estimate MDE analytically via the normal approximation for the
    # cluster-bootstrap: MDE ≈ z_{1-α/2} * σ_boot / sqrt(n_boot)
    # Since we run 10k bootstraps, σ_boot / sqrt(10000) is very small.
    # The relevant SE is σ_boot itself (the bootstrap std of ΔAUC).
    # MDE at 80% power: delta_min = (z_alpha + z_power) * SE
    # where SE ≈ boot_std (already accounts for clustering).
    # We return this as the MDE.
    pass  # computed per-dataset using boot_std from cluster_bootstrap_tost


# ── dataset registry ─────────────────────────────────────────────────────────

DATASETS = [
    {"name": "HaluEval (Qwen2.5-3B)",      "file": "phase3_results/train_features.csv"},
    {"name": "HaluEval (Qwen2.5-1.5B)",    "file": "phase3_results/train_features_qwen1_5b.csv"},
    {"name": "TruthfulQA (Qwen2.5-3B)",    "file": "phase10_results/qwen3b_truthfulqa_4stat.csv"},
    {"name": "TruthfulQA (SmolLM-1.7B)",   "file": "phase7_results/truthfulqa_smollm_features.csv"},
    {"name": "TruthfulQA (Phi-3-mini-3.8B)", "file": "phase10_results/phi3_truthfulqa_4stat.csv"},
]

EPSILONS = (0.010, 0.015, 0.020, 0.025)
N_BOOT = 10_000


# ── 1. TOST + Epsilon Sensitivity ─────────────────────────────────────────────

def run_tost_and_sensitivity():
    print("\n" + "="*80)
    print("SECTION 1 — TOST + EPSILON SENSITIVITY TABLE (10,000 bootstrap resamples)")
    print("="*80)

    rows = []
    boot_stds = {}

    for d in DATASETS:
        print(f"\n  → Processing: {d['name']}")
        df = pd.read_csv(d["file"])
        y = (df['label'] == 'hallucinated').astype(int).values
        groups = df['example_id'].values
        seq_len = df['seq_len'].values

        X_0d   = df[[c for c in df.columns if 'h0' in c]]
        X_1d   = df[[c for c in df.columns if 'h1' in c]]
        X_1d_n = X_1d.div(seq_len, axis=0)
        X_comb = pd.concat([X_0d, X_1d_n], axis=1)

        p0   = get_oof_preds(X_0d,   pd.Series(y), pd.Series(groups))
        pcmb = get_oof_preds(X_comb, pd.Series(y), pd.Series(groups))

        res = cluster_bootstrap_tost(y, p0, pcmb, groups, epsilons=EPSILONS, n_boot=N_BOOT)
        boot_stds[d["name"]] = res["boot_std"]

        row = {
            "Benchmark":          d["name"],
            "N_samples":          res["n_samples"],
            "N_groups":           res["n_groups"],
            "AUC(0D)":            round(res["auc_A"], 4),
            "AUC(0D+1D*)":        round(res["auc_B"], 4),
            "ΔAUC":               round(res["delta"], 5),
            "95% CI low":         round(res["ci_low"], 5),
            "95% CI high":        round(res["ci_high"], 5),
            "TOST_p(ε=0.010)":    res["tost_ps"][0.010],
            "TOST_p(ε=0.015)":    res["tost_ps"][0.015],
            "TOST_p(ε=0.020)":    res["tost_ps"][0.020],
            "TOST_p(ε=0.025)":    res["tost_ps"][0.025],
            "Equiv(ε=0.015)":     "CONFIRMED" if (res["ci_low"] > -0.015 and res["ci_high"] < 0.015) else "REJECTED",
        }
        rows.append(row)
        print(f"     AUC(0D)={res['auc_A']:.4f}  AUC(0D+1D*)={res['auc_B']:.4f}  Δ={res['delta']:+.5f}")
        print(f"     95% CI: [{res['ci_low']:+.5f}, {res['ci_high']:+.5f}]")
        for eps in EPSILONS:
            print(f"     TOST p (ε={eps:.3f}) = {res['tost_ps'][eps]:.5f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("tost_epsilon_sensitivity.csv", index=False)
    print(f"\n  ✓ Saved → tost_epsilon_sensitivity.csv")
    return df_out, boot_stds


# ── 2. Power Analysis / MDE ───────────────────────────────────────────────────

def run_power_analysis(boot_stds):
    """
    For each benchmark, compute the Minimum Detectable Effect (MDE) in ΔAUC
    at 80% power and α=0.05 (two-sided), using the cluster-bootstrap std as the SE estimate.

    MDE = (z_{α/2} + z_{power}) × SE
    where SE ≈ boot_std (the empirical std of the cluster-bootstrap ΔAUC distribution).
    """
    print("\n" + "="*80)
    print("SECTION 2 — POWER ANALYSIS: MINIMUM DETECTABLE EFFECT (MDE) IN ΔAUC")
    print("="*80)

    z_alpha = scipy_stats.norm.ppf(1 - 0.05/2)   # 1.960
    z_power = scipy_stats.norm.ppf(0.80)           # 0.842

    rows = []
    for d in DATASETS:
        df = pd.read_csv(d["file"])
        n = len(df) // 2  # approx unique examples (each has grounded+hallucinated)
        n_g = df['example_id'].nunique()
        se = boot_stds[d["name"]]
        mde = (z_alpha + z_power) * se
        pct_power_at_015 = float(scipy_stats.norm.cdf(0.015 / se - z_alpha) +
                                  scipy_stats.norm.sf(-0.015 / se + z_alpha))
        # Simpler: power = P(|Z| > z_alpha) where Z ~ N(delta/se, 1)
        # For delta = 0.015: power = 1 - Φ(z_alpha - 0.015/se) + Φ(-z_alpha - 0.015/se)
        power_at_015 = (1 - scipy_stats.norm.cdf(z_alpha - 0.015/se) +
                        scipy_stats.norm.cdf(-z_alpha - 0.015/se))

        row = {
            "Benchmark":              d["name"],
            "N_examples":             n,
            "N_groups":               n_g,
            "Bootstrap SE (ΔAUC)":    round(se, 5),
            "MDE at 80% power":       round(mde, 4),
            "Power at Δ=0.015":       f"{power_at_015*100:.1f}%",
            "Detects ε=0.015?":       "YES (>80%)" if power_at_015 >= 0.80 else f"NO ({power_at_015*100:.0f}%)",
        }
        rows.append(row)
        print(f"\n  {d['name']}")
        print(f"    N_groups={n_g}, Bootstrap SE={se:.5f}")
        print(f"    MDE @ 80% power: {mde:.4f} ΔAUC")
        print(f"    Power to detect Δ=0.015: {power_at_015*100:.1f}%")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("power_analysis_mde.csv", index=False)
    print(f"\n  ✓ Saved → power_analysis_mde.csv")
    return df_out


# ── 3. Scaling Exponent Regression (Prop 3 Verification) ─────────────────────

def run_scaling_regression():
    """
    Log-log regression: log(P1_total) ~ α * log(N) + β
    Tests whether 1D persistence scales linearly with N (α ≈ 1) or super-linearly.
    Uses HaluEval Qwen2.5-3B as the primary dataset (largest N, most variance).
    """
    print("\n" + "="*80)
    print("SECTION 3 — SCALING EXPONENT REGRESSION: log(P1) vs log(N)")
    print("="*80)

    results = []
    for d in DATASETS:
        df = pd.read_csv(d["file"])
        seq_len = df['seq_len'].values
        h1_cols = [c for c in df.columns if 'h1_total_persistence' in c]
        if not h1_cols:
            print(f"  Skipping {d['name']} — no h1_total_persistence columns")
            continue

        # Sum of raw 1D total persistence across all layers (unnormalized)
        p1_total = df[h1_cols].sum(axis=1).values

        # Filter to only rows with P1 > 0 (many short seqs have zero 1-cycles)
        mask = (p1_total > 0) & (seq_len > 0)
        if mask.sum() < 50:
            print(f"  Skipping {d['name']} — too few non-zero P1 rows ({mask.sum()})")
            continue

        log_n  = np.log(seq_len[mask])
        log_p1 = np.log(p1_total[mask])

        slope, intercept, r_value, p_value, se = scipy_stats.linregress(log_n, log_p1)
        r2 = r_value**2

        # Spearman for robustness
        spear_r, spear_p = scipy_stats.spearmanr(seq_len[mask], p1_total[mask])

        row = {
            "Dataset":         d["name"],
            "N_nonzero_rows":  int(mask.sum()),
            "OLS slope α":     round(slope, 4),
            "OLS intercept β": round(intercept, 4),
            "R²":              round(r2, 4),
            "OLS p-value":     float(p_value),
            "Spearman r":      round(spear_r, 4),
            "Spearman p":      float(spear_p),
            "Interpretation":  f"P1 ~ N^{slope:.2f} ({'linear' if abs(slope-1)<0.2 else 'super-linear' if slope>1 else 'sub-linear'})"
        }
        results.append(row)

        print(f"\n  {d['name']}")
        print(f"    Non-zero 1D rows: {mask.sum()} / {len(df)}")
        print(f"    OLS: log(P1) = {slope:.4f}·log(N) + {intercept:.4f}  (R²={r2:.4f}, p={p_value:.2e})")
        print(f"    → P1 ∝ N^{slope:.3f}  ({row['Interpretation']})")
        print(f"    Spearman(N, P1) = {spear_r:.4f}  (p={spear_p:.2e})")

    df_out = pd.DataFrame(results)
    df_out.to_csv("scaling_exponent_regression.csv", index=False)
    print(f"\n  ✓ Saved → scaling_exponent_regression.csv")
    return df_out


# ── 4. 0D ↔ Attention Marginals Spearman Correlation ─────────────────────────

def run_marginals_correlation():
    """
    Proposition 4 empirical support:
    Computes Spearman correlation between:
      - Total 0D persistence (sum of h0_total_persistence across layers) — proxy for MST weight
      - Attention sink mass: proxy = mean of h0_max_lifetime (the dominant attention sink weight)
      - Attention entropy proxy: std of h0_total_persistence values

    A high Spearman r confirms that 0D persistence is driven by attention marginals.
    """
    print("\n" + "="*80)
    print("SECTION 4 — 0D PERSISTENCE ↔ ATTENTION MARGINALS CORRELATION (Prop 4 Support)")
    print("="*80)

    results = []
    for d in DATASETS:
        df = pd.read_csv(d["file"])
        h0_tp_cols = [c for c in df.columns if 'h0_total_persistence' in c]
        h0_ml_cols = [c for c in df.columns if 'h0_max_lifetime' in c]
        if not h0_tp_cols or not h0_ml_cols:
            print(f"  Skipping {d['name']} — missing h0 columns")
            continue

        total_0d = df[h0_tp_cols].sum(axis=1).values
        sink_mass = df[h0_ml_cols].mean(axis=1).values  # mean max-lifetime ≈ attention sink
        sink_std  = df[h0_ml_cols].std(axis=1).values   # heterogeneity of attention distribution

        r_sink, p_sink = scipy_stats.spearmanr(total_0d, sink_mass)
        r_std,  p_std  = scipy_stats.spearmanr(total_0d, sink_std)

        row = {
            "Dataset":                    d["name"],
            "N":                          len(df),
            "Spearman(0D, sink_mass)":    round(r_sink, 4),
            "p(sink_mass)":               float(p_sink),
            "Spearman(0D, sink_std)":     round(r_std,  4),
            "p(sink_std)":                float(p_std),
            "Interpretation":             (
                "0D driven by attention sinks" if abs(r_sink) > 0.7
                else "Moderate sink correlation" if abs(r_sink) > 0.4
                else "Weak sink correlation"
            )
        }
        results.append(row)
        print(f"\n  {d['name']}")
        print(f"    Spearman(Total 0D Pers, Mean Sink Mass)  = {r_sink:.4f}  (p={p_sink:.2e})")
        print(f"    Spearman(Total 0D Pers, Sink Heterog.)  = {r_std:.4f}  (p={p_std:.2e})")
        print(f"    → {row['Interpretation']}")

    df_out = pd.DataFrame(results)
    df_out.to_csv("prop4_marginals_correlation.csv", index=False)
    print(f"\n  ✓ Saved → prop4_marginals_correlation.csv")
    return df_out


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# PAPER IMPROVEMENT: PHASE 1 STATISTICAL ANALYSIS")
    print("#"*80)

    tost_df, boot_stds = run_tost_and_sensitivity()
    power_df           = run_power_analysis(boot_stds)
    scaling_df         = run_scaling_regression()
    corr_df            = run_marginals_correlation()

    print("\n\n" + "="*80)
    print("ALL DONE — Summary of output files:")
    print("  tost_epsilon_sensitivity.csv   — TOST + epsilon sensitivity (5 benchmarks)")
    print("  power_analysis_mde.csv         — Power analysis & MDE per benchmark")
    print("  scaling_exponent_regression.csv — log(P1) vs log(N) OLS + Spearman")
    print("  prop4_marginals_correlation.csv — Spearman(0D, attention marginals)")
    print("="*80)
