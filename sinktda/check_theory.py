"""
check_theory.py -- numerical checks of Proposition 3 (TOHA) and Corollary 2 on random
causal attention matrices (with and without a first-token sink).

  python -m sinktda.check_theory
"""
import numpy as np
import ripser

from sinktda.features import coning_defects
from sinktda.toha import toha_head_features


def random_causal(rng, N, bias, scale):
    L = rng.normal(size=(N, N)) * scale
    L[:, 0] += bias
    L[np.triu_indices(N, 1)] = -np.inf
    A = np.exp(L - L.max(1, keepdims=True))
    return A / A.sum(1, keepdims=True)


def dist(A):
    D = 1 - np.maximum(A, A.T)
    np.fill_diagonal(D, 0)
    return D


def check_toha(n=400, seed=0):
    rng = np.random.default_rng(seed)
    bad = coned = 0
    err = 0.0
    for _ in range(n):
        N = rng.integers(4, 20)
        p = rng.integers(1, N)
        A = random_causal(rng, N, rng.choice([0, 3, 6]), rng.choice([0.5, 2, 4]))
        D = dist(A)
        D[:p, :p] = 0
        h0 = ripser.ripser(D, distance_matrix=True, maxdim=0)["dgms"][0]
        ref = h0[np.isfinite(h0[:, 1]), 1].sum() / (N - p)
        f = toha_head_features(A[None, p:, :], p)
        err = max(err, abs(f["toha"][0] - ref))
        bad += not (f["maxp"][0] - f["dP"][0] - 1e-6 <= f["toha"][0] <= f["maxp"][0] + 1e-6)
        if f["dP"][0] == 0:
            coned += 1
            bad += abs(f["toha"][0] - f["maxp"][0]) > 1e-9
    print(f"Prop. 3: {n} matrices, max |Prim - ripser| = {err:.1e}, violations = {bad}, coned = {coned}")
    return dict(toha_n=n, toha_err=err, toha_viol=bad, toha_coned=coned)


def check_diagram(n=300, seed=1):
    rng = np.random.default_rng(seed)
    bad0 = badk = 0
    for _ in range(n):
        N = rng.integers(4, 16)
        D = dist(random_causal(rng, N, rng.choice([0, 2, 5]), 2))
        d = coning_defects(D)[0]
        dg = ripser.ripser(D, distance_matrix=True, maxdim=2)["dgms"]
        deaths = np.sort(dg[0][np.isfinite(dg[0][:, 1]), 1])
        star = np.sort(D[0, 1:])
        bad0 += not (np.all(deaths <= star + 1e-6) and np.all(deaths >= star - d - 1e-6))
        badk += sum(len(dg[k]) > 0 and (dg[k][:, 1] - dg[k][:, 0]).max() > d + 1e-6 for k in (1, 2))
    print(f"Cor. 2: {n} matrices, elementwise 0D violations = {bad0}, H1/H2 bar violations = {badk}")
    return dict(diag_n=n, diag_viol0=bad0, diag_violk=int(badk))


if __name__ == "__main__":
    import pandas as pd
    pd.DataFrame([{**check_toha(), **check_diagram()}]).to_csv("sinktda_results/check_theory.csv", index=False)
