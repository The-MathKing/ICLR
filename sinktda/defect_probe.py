"""
defect_probe.py -- is the coning defect itself a hallucination signal?

Banks (same evaluator as every other number):
  DELTA     per-layer delta_0 and min_s delta_s (head-averaged graph)
  PH_DELTA  per-head delta_0
  PH_SINK+PH_DELTA
Incremental tests: early fusion over PH_SINK, late fusion (stacked OOF logits) over
SINK, PH_SINK and NONTOPO, with the stored first-level scores in sinktda_results/oof/.

  python -m sinktda.defect_probe [setting ...]
Writes sinktda_results/defect_probe.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sinktda.evaluate import (N_JOBS, RES, ROOT, bootstrap_delta, cols, fast_auc, load_setting,
                              oof_predictions)
from sinktda.late_fusion import logit


def _one(name):
    df, ph, hid = load_setting(name)
    y, g = df["y"].values, df["example_id"].values
    n = len(df)
    B = {"DELTA": df[cols(df, ["delta0", "delta_min"])].values}
    if ph:
        B["PH_DELTA"] = ph["ph_delta0"].reshape(n, -1)
        B["PH_SINK+PH_DELTA"] = np.hstack([ph["ph_sink_mass"].reshape(n, -1), B["PH_DELTA"]])
    B = {k: np.nan_to_num(v.astype(np.float64)) for k, v in B.items()}
    preds = {k: oof_predictions(v, y, g)[0] for k, v in B.items()}
    z = np.load(os.path.join(RES, "oof", f"{name}.npz"))
    rows = [dict(setting=name, test=f"AUC_{k}", A="", B=k, auc_A=np.nan, auc_B=fast_auc(y, p))
            for k, p in preds.items()]

    def comp(tag, a, pa, b, pb):
        r = dict(setting=name, test=tag, A=a, B=b, auc_A=fast_auc(y, pa), auc_B=fast_auc(y, pb))
        r.update(bootstrap_delta(y, pa, pb, g))
        rows.append(r)

    if "PH_DELTA" in preds and "PH_SINK" in z:
        comp("DELTA_h_beyond_SINK_h", "PH_SINK", z["PH_SINK"], "PH_SINK+PH_DELTA", preds["PH_SINK+PH_DELTA"])
    for base in ("SINK", "PH_SINK", "NONTOPO"):
        if base not in z:
            continue
        pb = oof_predictions(logit(z[base])[:, None], y, g)[0]
        for extra in ("DELTA", "PH_DELTA"):
            if extra not in preds:
                continue
            pc = oof_predictions(np.column_stack([logit(z[base]), logit(preds[extra])]), y, g)[0]
            comp(f"LF_{extra}_beyond_{base}", base, pb, f"{base}(+){extra}", pc)
    return rows


def main():
    names = sys.argv[1:] or sorted(d for d in os.listdir(ROOT)
                                   if os.path.exists(os.path.join(ROOT, d, "layers.parquet")))
    out = Parallel(n_jobs=N_JOBS)(delayed(_one)(s) for s in names)
    from sinktda.evaluate import merge_csv
    df = merge_csv(os.path.join(RES, "defect_probe.csv"),
                   [r for rows in out for r in rows])
    show = ["setting", "test", "auc_A", "auc_B", "delta", "ci95_lo", "ci95_hi", "equiv_0.015"]
    print(df.reindex(columns=show).to_string(index=False))


if __name__ == "__main__":
    main()
