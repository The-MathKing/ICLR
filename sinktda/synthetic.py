"""
synthetic.py -- theory checks that need no language model.

  S-A  Proposition B: Morse lower bound on E[P1(N)] vs the i.i.d. simulation.
  S-B  Sink dose-response: as the BOS logit bias b grows, delta_0 -> 0, the
       1D total persistence collapses, and P0 -> star weight (Theorem A).
  S-C  Planted cycles vs sink: 1D detection AUC of a planted k-cycle as a
       function of cycle strength gamma and sink bias b; Theorem A predicts
       invisibility once delta_0 is small, whatever gamma is.

  python -m sinktda.synthetic
"""
import os
from math import comb

import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.metrics import roc_auc_score

from sinktda.features import delta_bos_causal, distance_from_attention, ph_features

OUT = "sinktda_results"


def morse_lower_bound(N):
    f = lambda p: max(0.0, comb(N, 2) * p - N - comb(N, 3) * p ** 3)
    return quad(f, 0, 1, limit=400, points=[N ** -0.5])[0]


def causal_attention(rng, N, b, scale=1.5, plant=None, gamma=0.0):
    Lg = rng.normal(size=(N, N)) * scale
    Lg[:, 0] += b
    if plant is not None:
        k = len(plant)
        for j in range(k):
            u, v = plant[(j + 1) % k], plant[j]
            hi, lo = max(u, v), min(u, v)
            Lg[hi, lo] += gamma
    Lg = np.where(np.tril(np.ones((N, N))) > 0, Lg, -np.inf)
    A = np.exp(Lg - Lg.max(1, keepdims=True))
    return A / A.sum(1, keepdims=True)


def iid_uniform(rng, N):
    """Symmetric D with i.i.d. Uniform[0,1] upper-triangular entries (Proposition 1's model)."""
    U = np.triu(rng.uniform(size=(N, N)), 1)
    return U + U.T


NS = (16, 24, 32, 48, 64, 96, 128, 160, 200, 256)


def run_morse(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for N in NS:
        draws = 20 if N <= 128 else 8
        p1 = [ph_features(iid_uniform(rng, N))["h1_total_persistence"] for _ in range(draws)]
        rows.append(dict(N=N, draws=draws, mean_P1=float(np.mean(p1)), std_P1=float(np.std(p1))))
        print(rows[-1], flush=True)
    sim = pd.DataFrame(rows)
    sim["morse_lower_bound"] = [morse_lower_bound(int(n)) for n in sim["N"]]
    sim["trivial_upper_bound"] = [comb(int(n), 2) / 2 for n in sim["N"]]
    sim["bound_holds"] = sim["mean_P1"] >= sim["morse_lower_bound"]
    a = np.polyfit(np.log(sim["N"]), np.log(sim["morse_lower_bound"].clip(1e-9)), 1)[0]
    print(sim.to_string(index=False))
    print(f"[S-A] log-log slope of lower bound over N-range: {a:.2f}")
    sim.to_csv(f"{OUT}/synthetic_morse_bound.csv", index=False)
    # reference exponents for the real-attention scaling table
    rows = [dict(reference="iid_uniform", alpha=float(np.polyfit(np.log(sim["N"]), np.log(sim["mean_P1"]), 1)[0]))]
    causal = []
    for N in NS:
        p1 = [ph_features(distance_from_attention(causal_attention(rng, N, 0.0)))["h1_total_persistence"]
              for _ in range(20 if N <= 128 else 8)]
        causal.append(float(np.mean(p1)))
        print("causal b=0", N, causal[-1], flush=True)
    rows.append(dict(reference="causal_b0", alpha=float(np.polyfit(np.log(NS), np.log(causal), 1)[0])))
    pd.DataFrame(rows).to_csv(f"{OUT}/synthetic_reference_alpha.csv", index=False)
    print(rows)


def run_dose(N=40, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        for t in range(trials):
            A = causal_attention(rng, N, b)
            D = distance_from_attention(A)
            f = ph_features(D)
            d0 = delta_bos_causal(A)
            rows.append(dict(b=b, delta0=d0, coned=d0 == 0.0, sink_mass=A[1:, 0].mean(),
                             h1_tot=f["h1_total_persistence"], h1_max=f["h1_max_lifetime"],
                             P0=f["h0_total_persistence"], star=D[0, 1:].sum()))
    df = pd.DataFrame(rows)
    df["gap"] = df["star"] - df["P0"]
    agg = df.groupby("b").agg(sink_mass=("sink_mass", "mean"), delta0=("delta0", "mean"),
                              frac_coned=("coned", "mean"), h1_tot=("h1_tot", "mean"),
                              frac_h1_zero=("h1_tot", lambda x: (x == 0).mean()),
                              star_minus_P0=("gap", "mean")).reset_index()
    corr = df.groupby("b").apply(lambda g: np.corrcoef(g["P0"], g["star"])[0, 1],
                                 include_groups=False)
    agg["corr_P0_star"] = corr.values
    assert (df["h1_max"] <= df["delta0"] + 1e-5).all()  # ripser stores float32
    print("[S-B] min gap", df["gap"].min()); assert (df["gap"] >= -1e-4).all() and (df["gap"] <= (N - 1) * df["delta0"] + 1e-4).all()
    print(agg.to_string(index=False))
    agg.to_csv(f"{OUT}/synthetic_sink_dose.csv", index=False)
    df.to_csv(f"{OUT}/synthetic_sink_dose_raw.csv", index=False)


def run_planted(N=40, k=6, n=150, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for b in [0, 2, 4, 6, 8]:
        for gamma in [0, 2, 4, 6, 8]:
            ys, s1, s0, d0s = [], [], [], []
            for c in (0, 1):
                for _ in range(n):
                    plant = sorted(rng.choice(np.arange(1, N), size=k, replace=False))
                    A = causal_attention(rng, N, b, plant=plant if c else None, gamma=gamma)
                    D = distance_from_attention(A)
                    f = ph_features(D)
                    ys.append(c)
                    s1.append(f["h1_total_persistence"])
                    s0.append(f["h0_total_persistence"])
                    d0s.append(delta_bos_causal(A))
            ys = np.array(ys)
            auc1 = roc_auc_score(ys, s1) if np.ptp(s1) > 0 else 0.5
            auc0 = roc_auc_score(ys, s0)
            rows.append(dict(b=b, gamma=gamma, auc_h1=max(auc1, 1 - auc1), auc_h0=max(auc0, 1 - auc0),
                             mean_delta0=float(np.mean(d0s)), frac_h1_zero=float(np.mean(np.array(s1) == 0))))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/synthetic_planted_cycle.csv", index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    import sys
    if "--morse" in sys.argv:
        run_morse()
    else:
        run_morse()
        run_dose()
        run_planted()
