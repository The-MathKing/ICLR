"""
toha.py -- TOHA (Bazarova et al., ACL 2026) on the sink-reduction settings.

TOHA's per-head score is d_h = MTop-Div(R, P) / |R|: the weight of the minimum spanning
forest attaching the response tokens R to the prompt P in the attention graph with
distances 1 - A and all prompt-prompt distances set to 0, divided by |R|.
Proposition 3 (paper): contracting P to one vertex p*, the graph on {p*} u R has
  D(p*, u) = 1 - pi_u,  pi_u = max_{v in P} A[u, v],
and Theorem 1 at apex p* gives
  mean_u (1 - pi_u) - delta_P  <=  d_h  <=  mean_u (1 - pi_u),
  delta_P = max_{w < u in R} [A[u, w] - min(pi_u, pi_w)]_+,
with equality when delta_P = 0. When token 0 is each response token's most-attended prompt
token, 1 - pi_u = 1 - A[u, 0] (the response-restricted sink column).

  python -m sinktda.toha extract --bench truthfulqa --model qwen1.5b [--sink-bias B --tag sb4]
  python -m sinktda.toha evaluate [setting ...]
Extraction writes sinktda_out/<setting>/toha.npz with (rows, L, H) arrays
  toha   exact MTop-Div / |R| (dense Prim on the contracted graph)
  maxp   mean_u (1 - pi_u)               (first-order "max prompt attention")
  sinkr  mean_u (1 - A[u, 0])            (response-restricted sink)
  dP     delta_P
  arg0   fraction of u in R whose most-attended prompt token is token 0
Evaluation writes sinktda_results/toha_{checks,auc,comp}.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from sinktda import data
from sinktda.features import batched_prim

OUT_ROOT = os.environ.get("SINKTDA_OUT", "sinktda_out")
RES = "sinktda_results"
N_MAX = 10  # TOHA's cap on the number of selected heads


# ---------------------------------------------------------------------------
# per-head quantities
# ---------------------------------------------------------------------------
def toha_head_features(R_rows, p):
    """R_rows: (B, T, N) float64 attention rows of the response tokens (positions p..N-1)
    for B heads. Returns dict of (B,) arrays."""
    B, T, N = R_rows.shape
    pi = R_rows[:, :, :p].max(-1)                      # (B, T)
    a0 = R_rows[:, :, 0]
    RR = R_rows[:, :, p:]                              # (B, T, T), lower triangular
    D = np.empty((B, T + 1, T + 1))
    D[:, 0, 0] = 0.0
    D[:, 0, 1:] = D[:, 1:, 0] = 1.0 - pi
    W = np.maximum(RR, RR.transpose(0, 2, 1))
    D[:, 1:, 1:] = 1.0 - W
    idx = np.arange(1, T + 1)
    D[:, idx, idx] = 0.0
    tot, _ = batched_prim(D)
    if T >= 2:
        u, w = np.tril_indices(T, -1)
        dP = np.maximum((RR[:, u, w] - np.minimum(pi[:, u], pi[:, w])).max(-1), 0.0)
    else:
        dP = np.zeros(B)
    arg0 = (R_rows[:, :, :p].argmax(-1) == 0).mean(-1)
    return dict(toha=tot / T, maxp=(1.0 - pi).mean(-1), sinkr=(1.0 - a0).mean(-1), dP=dP, arg0=arg0)


def sink_bias_attention(bias):
    """Eager attention with a constant logit bias on key 0 for every query after token 0."""
    import torch
    from torch import nn

    def fwd(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        rep = getattr(module, "num_key_value_groups", 1)
        if rep > 1:
            b, h, n, d = key.shape
            key = key[:, :, None].expand(b, h, rep, n, d).reshape(b, h * rep, n, d)
            value = value[:, :, None].expand(b, h, rep, n, value.shape[-1]).reshape(b, h * rep, n, value.shape[-1])
        w = torch.matmul(query, key.transpose(2, 3)) * scaling
        # custom implementations may receive no (or a boolean) mask: apply the causal mask
        # explicitly (single unpadded sequence, so causal is the only mask needed)
        n_q, n_k = w.shape[-2], w.shape[-1]
        causal = torch.ones(n_q, n_k, dtype=torch.bool, device=w.device).tril(n_k - n_q)
        w = w.masked_fill(~causal, torch.finfo(w.dtype).min)
        bias_t = torch.zeros(n_q, n_k, dtype=w.dtype, device=w.device)
        bias_t[1:, 0] = bias
        w = w + bias_t
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(w, value).transpose(1, 2).contiguous()
        return out, w
    return fwd


def extract(args):
    import torch
    from sinktda.extract import generate_onpolicy, load, onpolicy_rows, pick_device
    from transformers import AttentionInterface

    model_id, template = data.MODELS[args.model]
    template = args.template or template
    name = f"{args.bench}_{args.model}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(OUT_ROOT, name)
    path = os.path.join(out_dir, "toha.npz")
    if os.path.exists(path):
        print(f"[skip] {path} exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    device, dtype = pick_device()
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    impl = "eager"
    if args.sink_bias:
        AttentionInterface.register("sinkbias", sink_bias_attention(args.sink_bias))
        impl = "sinkbias"
    tok, model = load(model_id, device, dtype, attn=impl)
    print(f"[load] {model_id} {device} {dtype} attn={impl} bias={args.sink_bias}", flush=True)

    if args.bench == "truthfulqa":
        rows = list(data.truthfulqa_rows(tok, template, args.n or data.FULL_TRUTHFULQA))
    elif args.bench == "halueval":
        rows = list(data.halueval_rows(tok, template, args.n or 2000))
    else:
        # reuse the generations of the base setting so that labels are unchanged
        base = os.path.join(OUT_ROOT, f"{args.bench}_{args.model}")
        gens = generate_onpolicy(tok, model, device, data.triviaqa_questions(args.n or 2000),
                                 os.path.join(base, "generations.csv"))
        rows = list(onpolicy_rows(tok, gens))

    feats, meta = {}, []
    t0 = time.time()
    for i, r in enumerate(rows):
        enc = data.encode(tok, r["full"], return_tensors="pt").to(device)
        N = enc["input_ids"].shape[1]
        p = min(len(tok(r["prefix"].rstrip(" "))["input_ids"]), N - 1)
        if N > 1024:
            continue
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        # rows of the response tokens only: (L, H, T, N)
        att = torch.stack([a[0, :, p:, :].float().cpu() for a in out.attentions]).numpy().astype(np.float64)
        att = np.nan_to_num(att)
        L, H, T, _ = att.shape
        if np.abs(np.triu(att[:, :, :, p:], 1)).max() > 1e-6:
            raise RuntimeError("attention is not causal; refusing to continue")
        f = toha_head_features(att.reshape(L * H, T, N), p)
        for k, v in f.items():
            feats.setdefault(k, []).append(v.reshape(L, H).astype(np.float32))
        meta.append((r["example_id"], r["label"], N, p))
        if i % 10 == 0 and device == "mps":
            del out
            torch.mps.empty_cache()
        if i % 200 == 0:
            print(f"[toha] {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    m = np.array(meta, dtype=object)
    np.savez_compressed(path, example_id=m[:, 0].astype(int), label=m[:, 1].astype(str),
                        seq_len=m[:, 2].astype(int), prompt_len=m[:, 3].astype(int),
                        **{k: np.stack(v) for k, v in feats.items()})
    print(f"[done] {path} rows={len(m)} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def toha_select(d, y, n_max=N_MAX):
    """TOHA Algorithm 1 on a training set. d: (n, K) head scores. Returns selected head indices."""
    delta = d[y == 1].mean(0) - d[y == 0].mean(0)
    order = np.argsort(-delta, kind="stable")
    from sinktda.evaluate import fast_auc
    best, n_opt, run = -1.0, 1, np.zeros(len(y))
    for n in range(1, n_max + 1):
        run = run + d[:, order[n - 1]]
        a = fast_auc(y, run)
        if a > best:
            best, n_opt = a, n
    return order[:n_opt]


def toha_oof(d, y, g):
    from sinktda.evaluate import N_SPLITS, SEEDS, _splitter
    preds, sel = [], []
    for seed in SEEDS:
        oof = np.zeros(len(y))
        for tr, te in _splitter(g, seed, N_SPLITS).split(np.zeros(len(y)), y, g):
            h = toha_select(d[tr], y[tr])
            oof[te] = d[te][:, h].mean(1)
            sel.append(h)
        preds.append(oof)
    return np.mean(preds, 0), sel


def _load(name):
    z = dict(np.load(os.path.join(OUT_ROOT, name, "toha.npz"), allow_pickle=False))
    lay = pd.read_parquet(os.path.join(OUT_ROOT, name.split("_sb")[0], "layers.parquet"),
                          columns=["example_id", "label"])
    if len(lay) != len(z["label"]) or (lay["example_id"].values != z["example_id"]).any():
        raise RuntimeError(f"{name}: rows do not align with layers.parquet")
    return z


def checks(name, z):
    toha, maxp, sinkr, dP, arg0 = (z[k].astype(np.float64) for k in ("toha", "maxp", "sinkr", "dP", "arg0"))
    tol = 1e-4  # attention stored in float32
    coned = dP <= 1e-7
    from scipy.stats import spearmanr
    n, L, H = toha.shape
    rho_mp = [spearmanr(toha[:, l, h], maxp[:, l, h])[0] for l in range(L) for h in range(H)
              if toha[:, l, h].std() > 0 and maxp[:, l, h].std() > 0]
    rho_sk = [spearmanr(toha[:, l, h], sinkr[:, l, h])[0] for l in range(L) for h in range(H)
              if toha[:, l, h].std() > 0 and sinkr[:, l, h].std() > 0]
    return dict(setting=name, rows=n, cells=int(toha.size),
                viol_upper=int((toha > maxp + tol).sum()),
                viol_lower=int((toha < maxp - dP - tol).sum()),
                frac_coned_P=float(coned.mean()),
                max_abs_gap_coned=float(np.abs(toha - maxp)[coned].max()) if coned.any() else np.nan,
                frac_arg0_all=float((arg0 == 1).mean()),
                median_head_rho_maxp=float(np.nanmedian(rho_mp)),
                median_head_rho_sinkr=float(np.nanmedian(rho_sk)),
                frac_heads_rho_maxp_ge_0_9=float(np.mean(np.array(rho_mp) >= 0.9)))


def evaluate(names):
    from joblib import Parallel, delayed
    from sinktda.evaluate import N_JOBS, bootstrap_delta, fast_auc, oof_predictions
    from sinktda.late_fusion import logit

    def one(name):
        z = _load(name)
        y = (z["label"] == "hallucinated").astype(int)
        g = z["example_id"]
        n = len(y)
        ck = checks(name, z)
        S = {k: z[k].reshape(n, -1).astype(np.float64) for k in ("toha", "maxp", "sinkr")}
        P, sel = {}, {}
        for k, d in S.items():
            P["TOHA_" + k], sel[k] = toha_oof(d, y, g)          # TOHA Algorithm 1
        for k, d in S.items():
            P["SUP_" + k] = oof_predictions(d, y, g)[0]         # supervised, all heads (TOHA Table 9)
        # properties of the heads TOHA itself selects
        hs = np.concatenate(sel["toha"])
        dP = z["dP"].reshape(n, -1)
        ck["toha_sel_frac_coned_P"] = float((dP[:, hs] <= 1e-7).mean())
        ck["toha_sel_frac_arg0"] = float((z["arg0"].reshape(n, -1)[:, hs] == 1).mean())
        ck["toha_sel_mean_nheads"] = float(np.mean([len(h) for h in sel["toha"]]))
        ck["toha_sel_overlap_sinkr"] = float(np.mean([len(set(a) & set(b)) / len(a)
                                                     for a, b in zip(sel["toha"], sel["sinkr"])]))
        auc = [dict(setting=name, bank=k, auc=fast_auc(y, p)) for k, p in P.items()]
        comp = []

        def c(tag, a, pa, b, pb):
            r = dict(setting=name, test=tag, A=a, B=b, auc_A=fast_auc(y, pa), auc_B=fast_auc(y, pb))
            r.update(bootstrap_delta(y, pa, pb, g))
            comp.append(r)

        c("TOHA_maxp_vs_toha", "TOHA_toha", P["TOHA_toha"], "TOHA_maxp", P["TOHA_maxp"])
        c("TOHA_sinkr_vs_toha", "TOHA_toha", P["TOHA_toha"], "TOHA_sinkr", P["TOHA_sinkr"])
        c("SUP_maxp_vs_toha", "SUP_toha", P["SUP_toha"], "SUP_maxp", P["SUP_maxp"])
        c("SUP_sinkr_vs_toha", "SUP_toha", P["SUP_toha"], "SUP_sinkr", P["SUP_sinkr"])
        # does TOHA add anything to its first-order counterpart, or to non-topological probes?
        pm, _ = oof_predictions(logit(P["SUP_maxp"])[:, None], y, g)
        pmt, _ = oof_predictions(np.column_stack([logit(P["SUP_maxp"]), logit(P["SUP_toha"])]), y, g)
        c("LF_toha_beyond_maxp", "SUP_maxp", pm, "SUP_maxp(+)SUP_toha", pmt)
        oof = os.path.join(RES, "oof", f"{name}.npz")
        if os.path.exists(oof):
            zz = np.load(oof)
            if "NONTOPO" in zz and len(zz["y"]) == n:
                pb, _ = oof_predictions(logit(zz["NONTOPO"])[:, None], y, g)
                pt, _ = oof_predictions(np.column_stack([logit(zz["NONTOPO"]), logit(P["SUP_toha"])]), y, g)
                c("LF_toha_beyond_nontopo", "NONTOPO", pb, "NONTOPO(+)SUP_toha", pt)
                auc.append(dict(setting=name, bank="NONTOPO", auc=fast_auc(y, zz["NONTOPO"])))
        return ck, auc, comp

    out = Parallel(n_jobs=N_JOBS)(delayed(one)(s) for s in names)
    ck = pd.DataFrame([o[0] for o in out])
    auc = pd.DataFrame([r for o in out for r in o[1]])
    comp = pd.DataFrame([r for o in out for r in o[2]])
    suffix = os.environ.get("TOHA_SUFFIX", "")
    ck.to_csv(f"{RES}/toha_checks{suffix}.csv", index=False)
    auc.to_csv(f"{RES}/toha_auc{suffix}.csv", index=False)
    comp.to_csv(f"{RES}/toha_comp{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(ck.to_string(index=False))
    print(auc.pivot(index="setting", columns="bank", values="auc").round(3).to_string())
    print(comp[["setting", "test", "auc_A", "auc_B", "delta", "ci90_lo", "ci90_hi", "equiv_0.015"]].to_string(index=False))


def main():
    if sys.argv[1:2] == ["evaluate"]:
        # default: every unmodified setting (sink-bias runs are evaluated by run_toha_causal.sh)
        names = sys.argv[2:] or sorted(d for d in os.listdir(OUT_ROOT)
                                       if os.path.exists(os.path.join(OUT_ROOT, d, "toha.npz")) and "_sb" not in d)
        return evaluate(names)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--bench", required=True, choices=["truthfulqa", "halueval", "triviaqa"])
    ap.add_argument("--model", required=True, choices=list(data.MODELS))
    ap.add_argument("--template", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--sink-bias", type=float, default=0.0)
    extract(ap.parse_args())


if __name__ == "__main__":
    main()
