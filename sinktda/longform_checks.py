"""
longform_checks.py -- does Theorem 1 still bite when the sequence is long?

Every benchmark in the body is short-form: tens of tokens, and in the sink-dominated models
most layer graphs are exactly coned. The reviewerly question this answers is whether that
survives on long, heavily context-grounded input, where attention is spread over a retrieved
passage rather than concentrated on a handful of tokens.

This is the mechanistic half of that question and needs no hallucination labels: Theorem 1 is
a statement about the attention graph, so the quantities to watch are the coned share, the
coning defect, and how tightly length-normalized 0D total persistence still tracks sink
attention. Reads a whole-graph run produced by

  python -m sinktda.extract --bench ragtruth --model <m> --n <n> --tag long \\
      --no-perhead --max-len <L> --maxdim 0

and writes one row per length bucket, plus an "all" row, to

  archive/longform/sinktda_results/longform_checks.csv

Run:  python -m sinktda.longform_checks [--setting ragtruth_qwen1.5b_long]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.environ.get("SINKTDA_OUT", "sinktda_out")
RES = os.environ.get("SINKTDA_RES", "archive/longform/sinktda_results")
TOL = 1e-9
BUCKETS = [(0, 512), (512, 768), (768, 1024), (1024, 1536), (1536, 1 << 30)]


def layer_stack(df, suffix):
    """(layers, examples) for a per-layer column, in layer order."""
    cols = [c for c in df.columns if re.fullmatch(rf"layer_\d+_{suffix}", c)]
    cols.sort(key=lambda c: int(c.split("_")[1]))
    return np.stack([df[c].values.astype(np.float64) for c in cols]), cols


def stats(df, tag):
    d0, _ = layer_stack(df, "delta0")
    sm, _ = layer_stack(df, "sink_mass")
    p0, _ = layer_stack(df, "h0_total_persistence")
    n = df["seq_len"].values.astype(np.float64)

    coned = d0 <= TOL
    # Theorem 1(c): on a coned graph P_0 equals the star weight (N-1)(1-mbar) exactly, so
    # the length-normalized pair is what the correlation should be computed on -- raw P_0
    # and raw S_0 both scale with N and would correlate through length alone.
    per_layer_rho = []
    for l in range(d0.shape[0]):
        x, y = p0[l] / np.maximum(n - 1, 1), 1.0 - sm[l]
        if np.std(x) > 0 and np.std(y) > 0:
            r = spearmanr(x, y).statistic
            if np.isfinite(r):
                per_layer_rho.append(r)

    return dict(
        bucket=tag, examples=len(df), layer_graphs=int(d0.size),
        frac_coned=float(coned.mean()),
        median_delta0=float(np.median(d0)), p90_delta0=float(np.percentile(d0, 90)),
        mean_sink_mass=float(sm.mean()),
        median_layer_rho_norm=float(np.median(per_layer_rho)) if per_layer_rho else np.nan,
        min_layer_rho_norm=float(np.min(per_layer_rho)) if per_layer_rho else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="ragtruth_qwen1.5b_long")
    a = ap.parse_args()

    f = f"{OUT}/{a.setting}/layers.parquet"
    if not os.path.exists(f):
        raise SystemExit(f"no run at {f}")
    df = pd.read_parquet(f)

    rows = [stats(df, "all")]
    for lo, hi in BUCKETS:
        m = (df["seq_len"] >= lo) & (df["seq_len"] < hi)
        if m.sum() < 20:                       # too few for a stable per-layer median
            continue
        rows.append(stats(df[m], f"{lo}-{hi if hi < (1 << 29) else 'inf'}"))

    os.makedirs(RES, exist_ok=True)
    out = pd.DataFrame(rows)
    out.insert(0, "setting", a.setting)
    out.to_csv(f"{RES}/longform_checks.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {RES}/longform_checks.csv")


if __name__ == "__main__":
    main()
