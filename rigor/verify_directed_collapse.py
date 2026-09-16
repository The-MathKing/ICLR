"""
verify_directed_collapse.py
===========================
Numerical verification of Theorem 2 (Directed Collapse).

CLAIM. Let A be a causal attention matrix: A[i,j] > 0 iff j <= i, dense on that
support. Let DFl(A) be the filtered directed flag complex of the weighted digraph
whose edge i->j (for j <= i) carries filtration value 1 - A[i,j]. Let VR(W) be the
filtered Vietoris-Rips complex of D = 1 - max(A, A^T). Then DFl(A) and VR(W) are
isomorphic as FILTERED simplicial complexes, hence have identical persistence
diagrams in every dimension.

WHY BRUTE FORCE. The claim is about filtered complexes, not about any particular
library's output, and persistence-library conventions for directed input (notably
the zero-versus-absent edge ambiguity in flagser) are the dominant source of false
negatives here. So the primary test enumerates both complexes explicitly and
compares them simplex by simplex. If the filtered complexes are identical then the
diagrams are identical by definition, and no homology computation is required.

Two checks are run:

  CHECK 1 (structural). For every non-empty vertex subset S, exactly one ordering
    of S is a directed simplex of DFl(A). This is the load-bearing step of the
    proof: causal support is a total order, so the decreasing ordering is the
    unique valid one.

  CHECK 2 (filtration). For every subset S, the DFl filtration value of that unique
    ordering equals the VR filtration value of S.

A third, optional cross-check against ripser confirms that the VR convention used
here is the same one the paper's pipeline uses (H0 total persistence == MST weight,
i.e. Theorem 1) on the very same matrices.

Usage:
    python rigor/verify_directed_collapse.py
    python rigor/verify_directed_collapse.py --n_max 9 --trials 40

Outputs: rigor/results/directed_collapse_verification.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

TOL = 1e-12


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def causal_attention(n, rng):
    """Random causal row-stochastic attention matrix.

    Lower triangular including the diagonal (token i attends to j <= i), softmax
    normalised per row, so every entry on the causal support is strictly positive
    and the upper triangle is exactly zero -- matching what a decoder emits under
    causal masking.
    """
    logits = rng.normal(size=(n, n))
    support = np.tril(np.ones((n, n), dtype=bool))
    logits = np.where(support, logits, -np.inf)
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    a = shifted / shifted.sum(axis=1, keepdims=True)
    a[~support] = 0.0
    return a


def adversarial_causal_attention(n, rng, kind):
    """Degenerate causal matrices that stress edge cases a reviewer would ask about.

    `ties`      : heavy rounding, so many simplices share a filtration value. Ties are
                  where persistence implementations differ in tie-breaking, though the
                  theorem itself is indifferent to them (both complexes receive the
                  same tied values).
    `underflow` : exact zeros on the causal support, as produced by fp16 underflow in
                  real extraction. These give D = 1, the maximum distance.
    `uniform`   : perfectly uniform attention, the maximal-degeneracy case.
    """
    a = causal_attention(n, rng)
    if kind == "ties":
        a = np.round(a, 2)
    elif kind == "underflow":
        support = np.tril(np.ones((n, n), dtype=bool))
        kill = (rng.random((n, n)) < 0.3) & support
        np.fill_diagonal(kill, False)
        a = np.where(kill, 0.0, a)
    elif kind == "uniform":
        support = np.tril(np.ones((n, n), dtype=bool))
        a = np.where(support, 1.0 / np.arange(1, n + 1)[:, None], 0.0)
    return a


def sym_distance(a):
    """The paper's distance matrix: D = 1 - max(A, A^T), zero diagonal."""
    d = 1.0 - np.maximum(a, a.T)
    np.fill_diagonal(d, 0.0)
    return d


def has_directed_edge(src, dst):
    """Causal support: token `src` attends to `dst` iff dst <= src."""
    return dst <= src


# ---------------------------------------------------------------------------
# the two filtrations
# ---------------------------------------------------------------------------
def vr_value(d, subset):
    """VR filtration value of a vertex subset: max pairwise symmetrised distance."""
    if len(subset) < 2:
        return 0.0
    return max(d[u, v] for u, v in itertools.combinations(subset, 2))


def dfl_value(a, ordering):
    """Directed-flag filtration value of an ordered tuple: max over its directed edges."""
    if len(ordering) < 2:
        return 0.0
    return max(
        1.0 - a[ordering[i], ordering[j]]
        for i in range(len(ordering))
        for j in range(i + 1, len(ordering))
    )


def valid_directed_orderings(subset):
    """Every ordering of `subset` that forms a directed simplex.

    (v_0,...,v_k) is a directed simplex iff the edge v_i -> v_j is present for all
    i < j. Enumerated exhaustively rather than assuming the decreasing ordering,
    because uniqueness is exactly what CHECK 1 is testing.
    """
    out = []
    for perm in itertools.permutations(subset):
        if all(
            has_directed_edge(perm[i], perm[j])
            for i in range(len(perm))
            for j in range(i + 1, len(perm))
        ):
            out.append(perm)
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def compare_complexes(a):
    """Enumerate both filtered complexes on the same vertex set and compare.

    Returns (n_subsets, max_abs_filtration_diff, n_uniqueness_violations).
    """
    n = a.shape[0]
    d = sym_distance(a)
    vertices = range(n)

    max_diff = 0.0
    uniqueness_violations = 0
    n_subsets = 0

    for size in range(1, n + 1):
        for subset in itertools.combinations(vertices, size):
            n_subsets += 1

            orderings = valid_directed_orderings(subset)
            if len(orderings) != 1:
                uniqueness_violations += 1
                continue

            diff = abs(dfl_value(a, orderings[0]) - vr_value(d, subset))
            max_diff = max(max_diff, diff)

    return n_subsets, max_diff, uniqueness_violations


def ripser_convention_check(a):
    """Optional: confirm VR(D) H0 total persistence == MST weight on this matrix.

    This is Theorem 1, re-checked here only to confirm that the VR convention used
    by this script is the same one the paper's pipeline uses.
    """
    try:
        import ripser
        import scipy.sparse.csgraph as csgraph
    except ImportError:
        return None

    d = sym_distance(a)
    dgm = ripser.ripser(d, distance_matrix=True, maxdim=0)["dgms"][0]
    finite = dgm[np.isfinite(dgm[:, 1])]
    h0_total = float(np.sum(finite[:, 1] - finite[:, 0]))
    mst_weight = float(csgraph.minimum_spanning_tree(d).sum())
    return abs(h0_total - mst_weight)


def bidirectional_attention(n, rng):
    """Dense NON-causal attention, as produced by a bidirectional encoder (BERT).

    Every ordered pair carries an edge, and A[i,j] != A[j,i] in general. This is the
    regime the attention-TDA literature originated in (kushnareva2021,
    cherniavskii2022), before the construction was inherited by decoder-only work.
    """
    logits = rng.normal(size=(n, n))
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def contrapositive_check(n, rng):
    """Show the collapse FAILS without causality -- i.e. Theorem 2 is not vacuous.

    Two things should break relative to the causal case:
      * uniqueness: with edges in both directions every ordering of a subset is a
        valid directed simplex, so a (k+1)-subset yields (k+1)! directed simplices
        rather than one, and DFl is strictly larger than VR;
      * filtration: those orderings do not share a filtration value, and the value
        generally differs from the VR value of the same subset.

    Returns (max_orderings_seen, max_spread_across_orderings, max_gap_to_vr).
    """
    a = bidirectional_attention(n, rng)
    d = sym_distance(a)

    max_orderings = 0
    max_spread = 0.0
    max_gap = 0.0

    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            orderings = [p for p in itertools.permutations(subset)]  # all valid: dense digraph
            values = [dfl_value(a, p) for p in orderings]
            vr = vr_value(d, subset)
            max_orderings = max(max_orderings, len(orderings))
            max_spread = max(max_spread, max(values) - min(values))
            max_gap = max(max_gap, max(abs(v - vr) for v in values))

    return max_orderings, max_spread, max_gap


def pyflagser_cross_check(a):
    """Optional cross-check against an independent directed-flag implementation.

    Skipped when pyflagser is unavailable. Treat a mismatch here as a convention
    question to investigate (pyflagser's handling of zero-weight versus absent
    edges is genuinely ambiguous for our input), NOT as a refutation -- the
    brute-force enumeration above is the authoritative test of the claim.
    """
    try:
        from pyflagser import flagser_weighted
    except ImportError:
        return None

    try:
        import ripser
    except ImportError:
        return None

    n = a.shape[0]
    # Directed weighted adjacency: edge i->j present iff j <= i, weight 1 - A[i,j].
    # Absent edges must be encoded as infinity, not zero -- zero would read as a
    # present edge entering at filtration time 0.
    adj = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j and has_directed_edge(i, j):
                adj[i, j] = 1.0 - a[i, j]
    np.fill_diagonal(adj, 0.0)

    directed = flagser_weighted(adj, max_dimension=1, directed=True)["dgms"]
    undirected = ripser.ripser(sym_distance(a), distance_matrix=True, maxdim=1)["dgms"]

    worst = 0.0
    for dim in range(min(len(directed), len(undirected))):
        lhs = np.sort(np.asarray(directed[dim]).reshape(-1, 2), axis=0)
        rhs = np.sort(np.asarray(undirected[dim]).reshape(-1, 2), axis=0)
        if lhs.shape != rhs.shape:
            return np.inf
        both = np.isfinite(lhs) & np.isfinite(rhs)
        if both.any():
            worst = max(worst, float(np.abs(lhs[both] - rhs[both]).max()))
    return worst


# ---------------------------------------------------------------------------
def main(n_max, trials, seed):
    rng = np.random.default_rng(seed)
    rows = []

    kinds = ["random", "ties", "underflow", "uniform"]

    print(f"Theorem 2 verification: causal sizes 2..{n_max}, {trials} trials each")
    print(f"matrix families: {', '.join(kinds)}\n")
    for n in range(2, n_max + 1):
        for kind in kinds:
            for trial in range(trials):
                a = (causal_attention(n, rng) if kind == "random"
                     else adversarial_causal_attention(n, rng, kind))
                n_subsets, max_diff, violations = compare_complexes(a)
                rows.append({
                    "N": n,
                    "kind": kind,
                    "trial": trial,
                    "n_subsets": n_subsets,
                    "max_filtration_diff": max_diff,
                    "uniqueness_violations": violations,
                    "ripser_mst_diff": ripser_convention_check(a),
                })
        sub = [r for r in rows if r["N"] == n]
        print(f"  N={n:2d}  subsets/trial={sub[0]['n_subsets']:5d}  "
              f"max|Δfiltration|={max(r['max_filtration_diff'] for r in sub):.2e}  "
              f"uniqueness violations={sum(r['uniqueness_violations'] for r in sub)}")

    df = pd.DataFrame(rows)
    os.makedirs("rigor/results", exist_ok=True)
    df.to_csv("rigor/results/directed_collapse_verification.csv", index=False)

    worst_filtration = df["max_filtration_diff"].max()
    total_violations = int(df["uniqueness_violations"].sum())
    worst_ripser = df["ripser_mst_diff"].dropna().max() if df["ripser_mst_diff"].notna().any() else None

    print("\n=== SUMMARY ===")
    print(f"Trials                              : {len(df)}")
    print(f"Simplices compared                  : {int(df['n_subsets'].sum()):,}")
    for kind, grp in df.groupby("kind", sort=False):
        print(f"  [{kind:9s}] max|Δ|={grp['max_filtration_diff'].max():.2e}  "
              f"violations={int(grp['uniqueness_violations'].sum())}")
    print(f"CHECK 1  uniqueness violations      : {total_violations}")
    print(f"CHECK 2  max |DFl - VR| filtration  : {worst_filtration:.3e}")
    if worst_ripser is not None:
        print(f"convention  max |H0total - MST|     : {worst_ripser:.3e}")

    flagser = pyflagser_cross_check(causal_attention(12, np.random.default_rng(seed + 1)))
    if flagser is None:
        print("pyflagser cross-check               : SKIPPED (pyflagser unavailable)")
    else:
        print(f"pyflagser cross-check max |Δ|       : {flagser:.3e}")

    # Contrapositive: without causality the collapse must FAIL, or the theorem is vacuous.
    print("\n=== CONTRAPOSITIVE (bidirectional / BERT-style attention) ===")
    bid_rng = np.random.default_rng(seed + 2)
    non_vacuous = True
    for n in (4, 5, 6):
        n_ord, spread, gap = contrapositive_check(n, bid_rng)
        print(f"  N={n}  max directed orderings per subset={n_ord} (causal case: 1)  "
              f"max spread across orderings={spread:.3e}  max |DFl - VR|={gap:.3e}")
        if n_ord <= 1 or gap <= TOL:
            non_vacuous = False
    print("  -> collapse FAILS without causality, as required"
          if non_vacuous else
          "  -> WARNING: collapse also holds bidirectionally; Theorem 2 would be vacuous")

    ok = total_violations == 0 and worst_filtration < TOL
    print("\n" + ("PASS: DFl(A) and VR(1 - max(A,A^T)) are identical filtered complexes."
                  if ok else "FAIL: Theorem 2 is violated -- see CSV."))
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_max", type=int, default=8,
                   help="largest vertex count to enumerate (cost grows as sum |S|!)")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raise SystemExit(main(args.n_max, args.trials, args.seed))
