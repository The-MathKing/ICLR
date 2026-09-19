"""
bound_gap.py -- how much of Theorem 1's slack real attention uses.

For every (example, layer) graph with delta_0 > 0 (on coned graphs both bounds are exact):
  u_0 = (S_0 - P_0) / ((N-2) delta_0)   share of the P_0 sandwich actually used, in [0, 1]
  u_1 = max H_1 bar / delta_0           share of the bar bound used (graphs with H_1 non-empty)
  vacuous: (N-2) delta_0 >= S_0, i.e. the lower bound on P_0 says nothing
  gap/S: relative distance of P_0 below the star weight

  python -m sinktda.bound_gap
Writes sinktda_results/bound_gap.csv
"""
import os

import numpy as np
import pandas as pd

from sinktda.evaluate import RES, ROOT, cols
from sinktda.report import ORDER


def run(name):
    d = pd.read_parquet(os.path.join(ROOT, name, "layers.parquet"))
    P0 = d[cols(d, ["h0_total_persistence"])].values
    S = d[cols(d, ["star_tot"])].values
    dl = d[cols(d, ["delta0"])].values
    h1 = d[cols(d, ["h1_max_lifetime"])].values
    N = d["seq_len"].values[:, None].astype(float) * np.ones_like(P0)
    open_ = dl > 0
    bound = (N - 2) * dl
    gap = S - P0
    u0 = gap[open_] / bound[open_]
    has1 = open_ & (h1 > 0)
    u1 = h1[has1] / dl[has1]
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else np.nan
    return dict(setting=name, graphs=int(P0.size), open=int(open_.sum()),
                u0_median=q(u0, 0.5), u0_q90=q(u0, 0.9), u0_q99=q(u0, 0.99), u0_max=q(u0, 1.0),
                gap_rel_median=q(gap[open_] / np.maximum(S[open_], 1e-12), 0.5),
                gap_rel_q99=q(gap[open_] / np.maximum(S[open_], 1e-12), 0.99),
                frac_vacuous=float((bound[open_] >= S[open_]).mean()) if open_.any() else np.nan,
                n_h1=int(has1.sum()), u1_median=q(u1, 0.5), u1_q90=q(u1, 0.9), u1_max=q(u1, 1.0),
                delta0_median_open=q(dl[open_], 0.5))


def main():
    names = [s for s in ORDER if os.path.exists(os.path.join(ROOT, s, "layers.parquet"))]
    out = pd.DataFrame([run(s) for s in names])
    out.to_csv(os.path.join(RES, "bound_gap.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
