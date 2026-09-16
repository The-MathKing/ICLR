"""
stats_upgrades.py -- two drop-in corrections to master_pipeline.py's statistics.

(1) within_group_permutation_null()
    master_pipeline.predictive_info_gain() calibrates I_hat against a null built
    by `rng.permutation(y)` -- a GLOBAL label shuffle. The data are paired: each
    question group holds exactly one grounded and one hallucinated row, and CV is
    StratifiedGroupKFold over those groups. A global shuffle destroys that
    one-positive-per-group structure, so the null is generated under a different
    dependence structure than the observed statistic. This module permutes labels
    WITHIN each group, which is the design-exact null.

(2) bca_tost()
    master_pipeline.cluster_bootstrap_tost() computes the one-sided p-values as
    raw percentile tail masses, mean(diffs >= eps) and mean(diffs <= -eps). Those
    coincide with exact TOST p-values only when the bootstrap distribution of
    Delta is symmetric. At n_groups ~ 150 that is not free. This module reports a
    bias-corrected and accelerated (BCa) interval instead and derives the TOST
    decision from it: equivalence at margin eps is confirmed iff the 90% BCa
    interval lies entirely inside (-eps, +eps), which is the standard
    interval-inclusion form of TOST at alpha = 0.05.

Both functions are deliberately written against the same inputs the existing
pipeline already has (out-of-fold predictions + group ids), so they can be
swapped in without re-fitting anything.
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------
# (1) design-exact permutation null
# --------------------------------------------------------------------------
def permute_within_groups(y, groups, rng):
    """Return a copy of y with labels permuted independently inside each group.

    For the 2-rows-per-group layout used throughout this paper this is a fair
    coin-flip of the (grounded, hallucinated) assignment within each question,
    which preserves both the overall base rate and the exactly-one-positive-per
    -group structure. For groups of other sizes it is a full within-group
    permutation, which preserves each group's own label multiset.
    """
    y = np.asarray(y)
    out = y.copy()
    order = np.argsort(groups, kind="stable")
    sorted_groups = np.asarray(groups)[order]
    # boundaries of each run of equal group id
    bounds = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi]
        out[idx] = rng.permutation(y[idx])
    return out


def within_group_permutation_null(cv_logloss_fn, Xb, Xc, y, groups,
                                  n_perm=100, seed=42):
    """Permutation null for the predictive information gain.

    cv_logloss_fn(X, y, groups, seed) -> (logloss, oof)   [same signature as the
    inner helper already defined in master_pipeline.predictive_info_gain]

    Returns the same dict shape as the existing implementation so the CSV schema
    and Table 18 do not change -- only the null does.
    """
    ll_base, _ = cv_logloss_fn(Xb, y, groups, seed)
    ll_comb, _ = cv_logloss_fn(Xc, y, groups, seed)
    info_gain_obs = ll_base - ll_comb

    rng = np.random.RandomState(seed)
    null_gains = []
    for i in range(n_perm):
        y_perm = permute_within_groups(y, groups, rng)
        if len(np.unique(y_perm)) < 2:
            continue
        ll_b, _ = cv_logloss_fn(Xb, y_perm, groups, seed + i + 1)
        ll_c, _ = cv_logloss_fn(Xc, y_perm, groups, seed + i + 1)
        null_gains.append(ll_b - ll_c)
    null_gains = np.asarray(null_gains)

    return {
        "info_gain_nats": float(info_gain_obs),
        "null_mean": float(np.mean(null_gains)),
        "null_p95": float(np.percentile(null_gains, 95)),
        "perm_p_value": float(np.mean(null_gains >= info_gain_obs)),
        "null_type": "within_group_permutation",
        "n_perm_effective": int(len(null_gains)),
    }


# --------------------------------------------------------------------------
# (2) BCa bootstrap + interval-inclusion TOST
# --------------------------------------------------------------------------
def _delta_auc(y, pA, pB, idx):
    yy = y[idx]
    if len(np.unique(yy)) < 2:
        return np.nan
    return roc_auc_score(yy, pB[idx]) - roc_auc_score(yy, pA[idx])


def bca_tost(y, predsA, predsB, groups, epsilons=(0.010, 0.015, 0.020, 0.025),
             n_boot=10000, seed=42):
    """Cluster bootstrap for Delta AUC with BCa intervals and TOST verdicts.

    Returns a dict with the same core keys as
    master_pipeline.cluster_bootstrap_tost, plus BCa 90%/95% limits and an
    `equivalent_bca_eps{e}` verdict per margin.
    """
    y = np.asarray(y)
    predsA = np.asarray(predsA)
    predsB = np.asarray(predsB)
    groups = np.asarray(groups)

    unique_groups, inv = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    rows_by_group = [np.flatnonzero(inv == g) for g in range(n_groups)]

    all_idx = np.arange(len(y))
    theta_hat = _delta_auc(y, predsA, predsB, all_idx)

    # --- bootstrap over groups
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.randint(0, n_groups, size=n_groups)
        idx = np.concatenate([rows_by_group[g] for g in sampled])
        d = _delta_auc(y, predsA, predsB, idx)
        if not np.isnan(d):
            boot.append(d)
    boot = np.asarray(boot)
    B = len(boot)

    # --- bias correction z0
    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (B + 1)), 1.0 - 1.0 / (B + 1))
    z0 = norm.ppf(prop_less)

    # --- acceleration a, from a leave-one-group-out jackknife
    jack = []
    for g in range(n_groups):
        keep = np.concatenate([rows_by_group[h] for h in range(n_groups) if h != g])
        d = _delta_auc(y, predsA, predsB, keep)
        if not np.isnan(d):
            jack.append(d)
    jack = np.asarray(jack)
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = float(num / den) if den != 0 else 0.0

    def bca_limits(conf):
        """Two-sided BCa interval at the given confidence level."""
        alpha = (1.0 - conf) / 2.0
        out = []
        for q in (alpha, 1.0 - alpha):
            zq = norm.ppf(q)
            adj = z0 + (z0 + zq) / (1.0 - a * (z0 + zq))
            pct = 100.0 * norm.cdf(adj)
            out.append(float(np.percentile(boot, np.clip(pct, 0.0, 100.0))))
        return out[0], out[1]

    lo90, hi90 = bca_limits(0.90)
    lo95, hi95 = bca_limits(0.95)

    res = {
        "auc_A": float(roc_auc_score(y, predsA)),
        "auc_B": float(roc_auc_score(y, predsB)),
        "delta_obs": float(theta_hat),
        "bca_ci90_low": lo90, "bca_ci90_high": hi90,
        "bca_ci95_low": lo95, "bca_ci95_high": hi95,
        "percentile_ci95_low": float(np.percentile(boot, 2.5)),
        "percentile_ci95_high": float(np.percentile(boot, 97.5)),
        "bootstrap_se": float(np.std(boot)),
        "bca_z0": float(z0), "bca_a": a,
        "n_groups": int(n_groups), "n_boot_effective": int(B),
    }
    for eps in epsilons:
        # TOST at alpha=0.05 <=> the 90% interval lies inside (-eps, +eps)
        confirmed = bool(lo90 > -eps and hi90 < eps)
        res[f"equivalent_bca_eps{eps}"] = "CONFIRMED" if confirmed else "REJECTED"
        # percentile-tail p-value, for comparison with the published table
        res[f"tost_p_percentile_eps{eps}"] = float(
            max(np.mean(boot <= -eps), np.mean(boot >= eps))
        )
    return res, boot


def skew_diagnostic(boot):
    """How far from symmetric is the bootstrap distribution? If |skew| is small
    and z0 ~ 0, the published percentile TOST and BCa will agree, and reporting
    that agreement is itself a useful robustness statement."""
    boot = np.asarray(boot)
    m, s = boot.mean(), boot.std()
    skew = float(np.mean(((boot - m) / s) ** 3)) if s > 0 else 0.0
    return {"mean": float(m), "std": float(s), "skew": skew,
            "median_minus_mean": float(np.median(boot) - m)}
