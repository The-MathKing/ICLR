"""
features.py -- per-example attention-graph features for the sink-reduction study.

Conventions (identical to the legacy phase* scripts, so 0D/1D reproduce exactly):
  A      head-averaged causal attention, float64
  W      = max(A, A^T);  D = 1 - W;  diag(D) = 0
  PH     Vietoris-Rips via ripser on D, maxdim=1

New quantities (Theorem A in the paper):
  star_tot  = sum_{u>=1} D[0,u]           (= (N-1) - total sink attention)
  star_max  = max_{u>=1} D[0,u]
  delta0    = max_{1<=v<u} (A[u,v] - min(A[u,0], A[v,0]))^+   (coning defect of BOS)
  delta_min = min_s delta_s over all vertices s (best possible apex)
"""
import numpy as np
import ripser

LEGACY_KEYS = ("h0_max_lifetime", "h0_total_persistence", "h1_max_lifetime",
               "h1_total_persistence", "h1_max_birth", "h1_max_death")


def distance_from_attention(A):
    W = np.maximum(A, A.T)
    D = 1.0 - W
    np.fill_diagonal(D, 0.0)
    return D


H1_KEYS = ("h1_max_lifetime", "h1_total_persistence", "h1_max_birth",
           "h1_max_death", "h1_count")


def ph_features(D, prefix="", maxdim=1):
    """0D/1D summary statistics. Matches the legacy extract_features().

    maxdim=0 computes the 0D barcode only. A dense Vietoris-Rips filtration to maxdim=1 is
    what makes long sequences unaffordable, and the 0D quantities the theory is about --
    total persistence, the star weight, the coning defect -- do not need the 1D pass. The
    1D keys are then absent rather than zero: zero would assert an empty H_1 we did not
    compute, which is only implied on coned graphs (Theorem 1(c)).
    """
    out = {}
    if D.shape[0] < 2:
        keys = LEGACY_KEYS + ("h1_count",)
        if maxdim < 1:
            keys = tuple(k for k in keys if not k.startswith("h1_"))
        for k in keys:
            out[prefix + k] = 0.0
        return out
    dgms = ripser.ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]
    h0 = dgms[0]
    h0 = h0[np.isfinite(h0[:, 1])]
    if len(h0):
        life = h0[:, 1] - h0[:, 0]
        out[prefix + "h0_max_lifetime"] = float(life.max())
        out[prefix + "h0_total_persistence"] = float(life.sum())
    else:
        out[prefix + "h0_max_lifetime"] = out[prefix + "h0_total_persistence"] = 0.0
    if maxdim < 1:
        return out
    h1 = dgms[1]
    h1 = h1[np.isfinite(h1[:, 1])] if len(h1) else h1
    if len(h1):
        life = h1[:, 1] - h1[:, 0]
        i = int(np.argmax(life))
        out[prefix + "h1_max_lifetime"] = float(life[i])
        out[prefix + "h1_total_persistence"] = float(life.sum())
        out[prefix + "h1_max_birth"] = float(h1[i, 0])
        out[prefix + "h1_max_death"] = float(h1[i, 1])
        out[prefix + "h1_count"] = float((life > 0).sum())
    else:
        for k in ("h1_max_lifetime", "h1_total_persistence", "h1_max_birth",
                  "h1_max_death", "h1_count"):
            out[prefix + k] = 0.0
    return out


def coning_defects(D):
    """delta_s for every vertex s (vector of length N). O(N^3) memory-light loop over s."""
    N = D.shape[0]
    if N < 3:
        return np.zeros(N)
    deltas = np.empty(N)
    offdiag = ~np.eye(N, dtype=bool)
    for s in range(N):
        ds = D[s]
        T = np.maximum(ds[:, None], ds[None, :]) - D
        m = offdiag.copy()
        m[s, :] = False
        m[:, s] = False
        deltas[s] = max(0.0, float(T[m].max()))
    return deltas


def delta_bos_causal(A):
    """Closed form of delta_0 for causal attention (Theorem A.3)."""
    N = A.shape[0]
    if N < 3:
        return 0.0
    u, v = np.tril_indices(N, -1)
    keep = v >= 1
    u, v = u[keep], v[keep]
    return max(0.0, float((A[u, v] - np.minimum(A[u, 0], A[v, 0])).max()))


def deflate_sink(A):
    """Remove vertex 0 and renormalize rows over the remaining (causal) support."""
    return deflate_vertex(A, 0)


def deflate_vertex(A, k):
    """Remove vertex k and renormalize rows over the remaining (causal) support.

    Deflating token 0 tests prediction C4 only when token 0 is the apex. Two models here
    (Qwen2.5-7B, SmolLM-1.7B) put their mass elsewhere, so for them deleting token 0
    removes an ordinary token and says nothing about removing the sink. Deleting
    argmin_s delta_s instead is the apex-faithful version of the same intervention.
    """
    keep = np.ones(A.shape[0], dtype=bool)
    keep[int(k)] = False
    B = A[np.ix_(keep, keep)].copy()
    rs = B.sum(1, keepdims=True)
    rs[rs <= 0] = 1.0
    return B / rs


def layer_features(A, prompt_len, with_delta_min=True, with_apex_deflation=False,
                   maxdim=1):
    """All per-layer scalar features for one head-averaged attention matrix.

    with_apex_deflation adds an `apexdefl_` block: the same deflation as `defl_`, but
    deleting the layer's best apex argmin_s delta_s instead of token 0. It needs
    with_delta_min, costs one more ripser call per layer, and is off by default so that
    existing feature dumps stay byte-identical.
    """
    A = A.astype(np.float64)
    N = A.shape[0]
    D = distance_from_attention(A)
    f = ph_features(D, maxdim=maxdim)
    f["star_tot"] = float(D[0, 1:].sum()) if N > 1 else 0.0
    f["star_max"] = float(D[0, 1:].max()) if N > 1 else 0.0
    f["sink_mass"] = float(A[1:, 0].mean()) if N > 1 else 0.0
    f["delta0"] = delta_bos_causal(A)
    if with_delta_min:
        ds = coning_defects(D)
        f["delta_min"] = float(ds.min())
        f["delta_argmin"] = float(int(ds.argmin()))
    # non-topological row statistics
    rows = A[1:] if N > 1 else A
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(rows > 0, rows * np.log(rows), 0.0)).sum(1)
    f["entropy_mean"] = float(ent.mean())
    f["rowmax_mean"] = float(rows.max(1).mean())
    # sink-deflated graph (first-order sink removal)
    if N > 2:
        Ad = deflate_sink(A)
        Dd = distance_from_attention(Ad)
        f.update(ph_features(Dd, prefix="defl_", maxdim=maxdim))
        f["defl_delta0"] = delta_bos_causal(Ad)
    else:
        f.update(ph_features(np.zeros((1, 1)), prefix="defl_", maxdim=maxdim))
        f["defl_delta0"] = 0.0
    # apex-deflated graph: delete argmin_s delta_s rather than token 0
    if with_apex_deflation:
        if not with_delta_min:
            raise ValueError("with_apex_deflation needs with_delta_min (the apex comes from delta_argmin)")
        k = int(f["delta_argmin"])
        if N > 2:
            Aa = deflate_vertex(A, k)
            f.update(ph_features(distance_from_attention(Aa), prefix="apexdefl_", maxdim=maxdim))
            f["apexdefl_delta0"] = delta_bos_causal(Aa)
        else:
            f.update(ph_features(np.zeros((1, 1)), prefix="apexdefl_", maxdim=maxdim))
            f["apexdefl_delta0"] = 0.0
        f["apexdefl_apex"] = float(k)
    # answer-only induced subgraph (raw distances, no renormalization)
    p = int(prompt_len)
    if N - p >= 2:
        f.update(ph_features(D[p:, p:].copy(), prefix="ans_", maxdim=maxdim))
    else:
        f.update(ph_features(np.zeros((1, 1)), prefix="ans_", maxdim=maxdim))
    return f


def example_layer_features(A_layers, prompt_len, with_delta_min=True,
                           with_apex_deflation=False, maxdim=1):
    """A_layers: (L, N, N) head-averaged attention. Returns flat dict layer_{l}_{k}."""
    out = {}
    for l in range(A_layers.shape[0]):
        for k, v in layer_features(A_layers[l], prompt_len, with_delta_min,
                                   with_apex_deflation, maxdim).items():
            out[f"layer_{l}_{k}"] = v
    return out


# ---------------------------------------------------------------------------
# batched per-head quantities (main process, numpy)
# ---------------------------------------------------------------------------
def batched_prim(D):
    """Dense Prim MST on a batch of distance matrices D: (B, N, N).
    Returns (total weight, max edge) arrays of shape (B,). Zero-weight edges are
    real edges (unlike scipy.sparse.csgraph, which drops explicit zeros)."""
    B, N, _ = D.shape
    if N < 2:
        return np.zeros(B), np.zeros(B)
    in_tree = np.zeros((B, N), dtype=bool)
    in_tree[:, 0] = True
    best = D[:, 0, :].copy()
    best[:, 0] = np.inf
    tot = np.zeros(B)
    mx = np.zeros(B)
    ar = np.arange(B)
    for _ in range(N - 1):
        j = np.argmin(best, axis=1)
        w = best[ar, j]
        tot += w
        mx = np.maximum(mx, w)
        in_tree[ar, j] = True
        best = np.minimum(best, D[ar, j, :])
        best[in_tree] = np.inf
    return tot, mx


def perhead_features(attn, prompt_len):
    """attn: (L, H, N, N) float32. Returns dict of (L, H) arrays."""
    L, H, N, _ = attn.shape
    A = attn.astype(np.float64)
    flat = A.reshape(L * H, N, N)
    W = np.maximum(flat, flat.transpose(0, 2, 1))
    D = 1.0 - W
    idx = np.arange(N)
    D[:, idx, idx] = 0.0
    tot, mx = batched_prim(D)
    out = {"ph_h0_tot": tot.reshape(L, H), "ph_h0_max": mx.reshape(L, H)}
    if N > 1:
        out["ph_sink_mass"] = A[:, :, 1:, 0].mean(-1)
        rows = A[:, :, 1:, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(np.where(rows > 0, rows * np.log(rows), 0.0)).sum(-1)
        out["ph_entropy"] = ent.mean(-1)
        out["ph_rowmax"] = rows.max(-1).mean(-1)
    else:
        z = np.zeros((L, H))
        out.update(ph_sink_mass=z, ph_entropy=z, ph_rowmax=z)
    # coning defect of BOS per head
    if N >= 3:
        u, v = np.tril_indices(N, -1)
        keep = v >= 1
        u, v = u[keep], v[keep]
        d = A[:, :, u, v] - np.minimum(A[:, :, u, 0], A[:, :, v, 0])
        out["ph_delta0"] = np.maximum(d.max(-1), 0.0)
    else:
        out["ph_delta0"] = np.zeros((L, H))
    # LLM-Check attention score: mean_i log A_ii per head
    diag = np.clip(A[:, :, idx, idx], 1e-12, None)
    out["ph_llmcheck"] = np.log(diag).mean(-1)
    # Lookback-Lens ratio: context (prompt) vs new (answer) attention, averaged over answer tokens
    p = int(prompt_len)
    if 0 < p < N:
        ctx = A[:, :, p:, :p].mean(-1)                      # (L,H,T)
        new = np.stack([A[:, :, t, p:t + 1].mean(-1) for t in range(p, N)], axis=-1)
        out["ph_lookback"] = (ctx / np.clip(ctx + new, 1e-12, None)).mean(-1)
    else:
        out["ph_lookback"] = np.zeros((L, H))
    return out
