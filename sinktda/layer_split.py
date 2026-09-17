"""
layer_split.py -- where does 0D add information beyond the sink?

Layers are split (label-free) by their exact-coning rate: "coned" layers have
frac_coned >= 0.9, "open" layers have frac_coned < 0.5. Theorem 1 predicts that
0D features of coned layers carry nothing beyond SINK, so any 0D|SINK gain must
come from open layers.

  python -m sinktda.layer_split [setting ...]
Writes sinktda_results/layer_split.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from sinktda.evaluate import (RES, ROOT, banks, bootstrap_delta, cols, fast_auc,
                              load_setting, oof_predictions)


def run(name):
    df, ph, hid = load_setting(name)
    lay = pd.read_csv(f"{RES}/theory_layers_{name}.csv")
    y, g = df["y"].values, df["example_id"].values
    B = banks(df, ph, hid)
    coned = lay.loc[lay["frac_coned"] >= 0.9, "layer"].tolist()
    open_ = lay.loc[lay["frac_coned"] < 0.5, "layer"].tolist()
    base, _ = oof_predictions(B["SINK"], y, g)
    out = []
    for tag, ls in (("coned", coned), ("open", open_)):
        if not ls:
            out.append(dict(setting=name, subset=tag, n_layers=0))
            continue
        X0 = df[[f"layer_{l}_{k}" for l in ls for k in ("h0_max_lifetime", "h0_total_persistence")]].values
        p, _ = oof_predictions(np.hstack([B["SINK"], X0]), y, g)
        r = dict(setting=name, subset=tag, n_layers=len(ls), layers=" ".join(map(str, ls)),
                 auc_sink=fast_auc(y, base), auc_sink_plus_0d=fast_auc(y, p))
        r.update(bootstrap_delta(y, base, p, g))
        out.append(r)
        print({k: r[k] for k in ("setting", "subset", "n_layers", "delta", "ci90_lo", "ci90_hi") if k in r}, flush=True)
    return out


def main():
    names = sys.argv[1:] or sorted(f[len("theory_layers_"):-4] for f in os.listdir(RES)
                                   if f.startswith("theory_layers_"))
    rows = []
    for n in names:
        rows += run(n)
    path = f"{RES}/layer_split.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old["setting"].isin(new["setting"])]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
