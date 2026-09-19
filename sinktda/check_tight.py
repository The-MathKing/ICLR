"""
check_tight.py -- the tightened lower bounds of Theorem 1(b) and Proposition 1 on real graphs.

  P_0 >= S_0 - (N-2) delta_0                      every (example, layer) graph, layers.parquet
  d   >= pibar-score - (|R|-1)/|R| delta_P        every (example, head) graph, toha.npz

Same tolerances as the original checks (1e-3 for the P_0 sum, 1e-4 for TOHA's float32 attention).

  python -m sinktda.check_tight
Writes sinktda_results/check_tight.csv
"""
import os

import numpy as np
import pandas as pd

from sinktda.evaluate import RES, ROOT, cols
from sinktda.report import ORDER


def main():
    rows = []
    for s in ORDER:
        r = dict(setting=s)
        f = os.path.join(ROOT, s, "layers.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f)
            P0, S, dl = (d[cols(d, [k])].values for k in ("h0_total_persistence", "star_tot", "delta0"))
            N = d["seq_len"].values[:, None].astype(float)
            r.update(layer_graphs=int(P0.size), layer_viol=int((P0 < S - (N - 2) * dl - 1e-3).sum()))
        f = os.path.join(ROOT, s, "toha.npz")
        if os.path.exists(f):
            z = np.load(f)
            R = (z["seq_len"] - z["prompt_len"]).astype(float)[:, None, None]
            t, mp, dP = (z[k].astype(np.float64) for k in ("toha", "maxp", "dP"))
            r.update(head_graphs=int(t.size),
                     head_viol=int((t < mp - (R - 1) / R * dP - 1e-4).sum() + (t > mp + 1e-4).sum()))
        if len(r) > 1:
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "check_tight.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
