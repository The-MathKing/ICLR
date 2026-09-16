"""
rigor/gate_c_mst_vs_logprob.py
==============================
GATE C: is the 0D (= MST) advantage over the log-probability baseline statistically real?

The manuscript currently asserts the opposite of what the data show. Section 7 says the
mean-token-log-probability probe "outperforms every topological, hidden-state, or
attention-marginal feature bank we report anywhere in this paper" -- but that was measured
only on HaluEval QA (N=329, one model), the benchmark now known to be lexically
contaminated (answer text alone separates the classes at AUC 0.96-0.98).

On TruthfulQA at full N, comparing master_results/master_auc_table.csv against
rigor/results/logprob_summary.csv suggests the reverse: 0D reaches 0.58-0.76 while the best
log-probability variant is near chance (0.48-0.61). That comparison is currently an
eyeball across two CSVs produced by different scripts on slightly different row sets (the
log-prob files have 1613-1630 rows against the feature files' 1634), so it needs to be
redone properly: inner-joined on (example_id, label) and tested with a paired, group-level
cluster bootstrap.

The interval here is a SUPERIORITY interval -- the question is whether the CI on
AUC(0D) - AUC(log-prob) excludes zero -- not a TOST equivalence interval. It reuses
cluster_bootstrap_tost only for its bootstrap machinery.

The log-prob bank is [mean_logprob, min_logprob], which logprob_summary.csv shows is the
strongest variant in almost every setting; using the strongest comparator is the
conservative choice for the claim being made.

Usage:
    python rigor/gate_c_mst_vs_logprob.py
    python rigor/gate_c_mst_vs_logprob.py --n_boot 10000

Outputs: rigor/results/gate_c_mst_vs_logprob.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from master_pipeline import canonical_oof, cluster_bootstrap_tost  # noqa: E402
from rigor.settings import SETTINGS, slug  # noqa: E402

warnings.filterwarnings("ignore")

LOGPROB_COLS = ["mean_logprob", "min_logprob"]


def main(n_boot, seed):
    results = []

    for s in SETTINGS:
        name, path = s["name"], s["features"]
        lp_path = f"rigor/results/logprob_{slug(name)}.csv"

        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue
        if not os.path.exists(lp_path):
            print(f"SKIP {name}: {lp_path} not found")
            continue

        print(f"\n=== {name} ===")
        df = pd.read_csv(path)
        lp = pd.read_csv(lp_path)
        for frame in (df, lp):
            frame["example_id"] = frame["example_id"].astype(int)
            frame["label"] = frame["label"].astype(str)

        merged = df.merge(lp[["example_id", "label"] + LOGPROB_COLS],
                          on=["example_id", "label"], how="inner")
        dropped = len(df) - len(merged)
        if len(merged) < 100:
            print(f"SKIP {name}: only {len(merged)} rows after join")
            continue

        y = (merged["label"] == "hallucinated").astype(int).values
        groups = merged["example_id"].values

        h0_cols = [c for c in merged.columns if "_h0_" in c]
        X_h0 = merged[h0_cols].values
        X_lp = merged[LOGPROB_COLS].values

        print(f"rows={len(merged)} (dropped {dropped} unjoined) "
              f"groups={len(np.unique(groups))} 0D dims={X_h0.shape[1]}")

        oof_lp = canonical_oof(X_lp, y, groups, seed=seed)
        oof_h0 = canonical_oof(X_h0, y, groups, seed=seed)
        oof_both = canonical_oof(np.hstack([X_lp, X_h0]), y, groups, seed=seed)

        # 0D versus log-prob (the gate)
        sup, _ = cluster_bootstrap_tost(y, oof_lp, oof_h0, groups, n_boot=n_boot, seed=seed)
        # are they complementary, or does 0D subsume log-prob?
        comp, _ = cluster_bootstrap_tost(y, oof_h0, oof_both, groups, n_boot=n_boot, seed=seed)

        auc_lp, auc_h0, auc_both = sup["auc_A"], sup["auc_B"], comp["auc_B"]
        delta, lo, hi = sup["delta_obs"], sup["ci95_low"], sup["ci95_high"]

        if lo > 0:
            verdict = "0D WINS"
        elif hi < 0:
            verdict = "LOGPROB WINS"
        else:
            verdict = "TIE"

        print(f"  AUC(log-prob)   = {auc_lp:.4f}")
        print(f"  AUC(0D)         = {auc_h0:.4f}")
        print(f"  AUC(both)       = {auc_both:.4f}")
        print(f"  0D - log-prob   = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
        print(f"  both - 0D       = {comp['delta_obs']:+.4f}  "
              f"95% CI [{comp['ci95_low']:+.4f}, {comp['ci95_high']:+.4f}]"
              f"{'  (complementary)' if comp['ci95_low'] > 0 else ''}")

        results.append({
            "Setting": name,
            "Benchmark": s["benchmark"],
            "N_rows": len(merged),
            "N_rows_dropped_in_join": int(dropped),
            "N_groups": int(len(np.unique(groups))),
            "dim_0D": X_h0.shape[1],
            "AUC_logprob": round(auc_lp, 4),
            "AUC_0D": round(auc_h0, 4),
            "AUC_0D_plus_logprob": round(auc_both, 4),
            "Delta_0D_minus_logprob": round(delta, 4),
            "CI95_low": round(lo, 4),
            "CI95_high": round(hi, 4),
            "bootstrap_se": round(sup["bootstrap_se"], 5),
            "verdict": verdict,
            "Delta_both_minus_0D": round(comp["delta_obs"], 4),
            "complementary": bool(comp["ci95_low"] > 0),
        })

    if not results:
        print("\nNo settings evaluated.")
        return 1

    out = pd.DataFrame(results)
    os.makedirs("rigor/results", exist_ok=True)
    out.to_csv("rigor/results/gate_c_mst_vs_logprob.csv", index=False)

    print("\n=== GATE C VERDICT ===")
    print(out[["Setting", "AUC_logprob", "AUC_0D", "Delta_0D_minus_logprob",
               "CI95_low", "CI95_high", "verdict"]].to_string(index=False))

    tqa = out[out["Benchmark"] == "truthfulqa"]
    wins = int((tqa["verdict"] == "0D WINS").sum())
    print(f"\nTruthfulQA settings where 0D significantly beats log-prob: {wins}/{len(tqa)}")
    hal = out[out["Benchmark"] == "halueval"]
    if len(hal):
        hal_wins = int((hal["verdict"] == "LOGPROB WINS").sum())
        print(f"HaluEval settings where log-prob significantly beats 0D: {hal_wins}/{len(hal)}"
              "  (expected -- this is the contaminated benchmark)")
    print("\nSaved rigor/results/gate_c_mst_vs_logprob.csv")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raise SystemExit(main(args.n_boot, args.seed))
