"""
ragtruth_checks.py -- does the TOHA reduction survive long retrieved contexts?

Proposition 1 bounds TOHA's divergence d between the pi-bar score and the pi-bar score
minus a prompt-level coning defect delta_P, with equality when delta_P vanishes. The
paper verifies this on short-form QA, where prompts are tens of tokens. TOHA was designed
for retrieval-augmented generation, where the prompt is a retrieved passage: attention over
it can be diffuse, delta_P has more room to grow, and the exact case can become rare.

This reads the RAGTruth run produced by

  python -m sinktda.toha extract --bench ragtruth --model <m> --n <n> --max-len <L>

and writes one row per length bucket, plus an "all" row, to

  archive/ragtruth/sinktda_results/ragtruth_checks.csv

Run:  python -m sinktda.ragtruth_checks [--model qwen1.5b]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.environ.get("SINKTDA_OUT", "sinktda_out")
RES = os.environ.get("SINKTDA_RES", "archive/ragtruth/sinktda_results")
TOL = 1e-5
BUCKETS = [(0, 512), (512, 768), (768, 1024), (1024, 1536), (1536, 1 << 30)]


def median_head_rho(x, y):
    """Median over (layer, head) of the Spearman correlation across examples."""
    E, L, H = x.shape
    r = [spearmanr(x[:, l, h], y[:, l, h]).statistic for l in range(L) for h in range(H)]
    r = [v for v in r if np.isfinite(v)]
    return float(np.median(r)) if r else float("nan")


def stats(d, mp, dP, sr, tag, n_ex):
    coned = dP <= TOL
    return dict(
        bucket=tag, examples=n_ex, head_graphs=int(d.size),
        # a violation would mean the proposition is false or the boundary is wrong
        viol_upper=int(((d - mp) > TOL).sum()),
        viol_lower=int((((mp - dP) - d) > TOL).sum()),
        frac_coned=float(coned.mean()),
        max_gap_coned=float(np.abs(d - mp)[coned].max()) if coned.any() else float("nan"),
        median_dP=float(np.median(dP)), p90_dP=float(np.percentile(dP, 90)),
        rho_maxp=median_head_rho(d, mp), rho_sinkr=median_head_rho(d, sr),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b")
    a = ap.parse_args()

    f = f"{OUT}/ragtruth_{a.model}/toha.npz"
    if not os.path.exists(f):
        raise SystemExit(f"no RAGTruth run at {f}; run `toha extract --bench ragtruth` first")
    z = np.load(f, allow_pickle=True)
    d, mp, dP, sr = (z[k].astype(np.float64) for k in ("toha", "maxp", "dP", "sinkr"))
    sl = z["seq_len"]

    rows = [stats(d, mp, dP, sr, "all", len(sl))]
    for lo, hi in BUCKETS:
        m = (sl >= lo) & (sl < hi)
        if m.sum() < 20:                      # too few to report a median correlation
            continue
        rows.append(stats(d[m], mp[m], dP[m], sr[m],
                          f"{lo}-{hi if hi < (1 << 29) else 'inf'}", int(m.sum())))

    os.makedirs(RES, exist_ok=True)
    out = pd.DataFrame(rows)
    out.insert(0, "model", a.model)
    out.to_csv(f"{RES}/ragtruth_checks.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {RES}/ragtruth_checks.csv")


if __name__ == "__main__":
    main()
