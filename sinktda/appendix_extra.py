"""
appendix_extra.py -- supplementary appendix material, all from the new pipeline.

  python -m sinktda.appendix_extra            # CPU-only parts
  python -m sinktda.appendix_extra --examples # also the worked example (one TinyLlama pass)

Writes paper/sink_appendix_extra.tex and paper/fig_sink_layer_auc.pdf, paper/fig_sink_example.pdf
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr, ttest_ind

from sinktda.evaluate import cols, fast_auc, n_layers
from sinktda.report import INK2, ORDER, PRETTY, SERIES, _join_names, short, style

OUT = "sinktda_out"
RES = "sinktda_results"
PAPER = "paper"
def _ref_alpha(kind, default=np.nan):
    p = f"sinktda_results/synthetic_reference_alpha.csv"
    if not os.path.exists(p):
        return default
    r = pd.read_csv(p).set_index("reference")["alpha"]
    return float(r.get(kind, default))


IID_ALPHA = _ref_alpha("iid_uniform")      # log-log slope, i.i.d. Uniform[0,1] weights
CAUSAL_ALPHA = _ref_alpha("causal_b0")     # log-log slope, random causal attention without a sink

MODEL_SPECS = {  # parameters from the model cards
    "Qwen/Qwen2.5-3B-Instruct": "3.09B", "Qwen/Qwen2.5-1.5B-Instruct": "1.54B",
    "microsoft/Phi-3-mini-4k-instruct": "3.82B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "1.10B",
    "HuggingFaceTB/SmolLM-1.7B-Instruct": "1.71B", "mistralai/Mistral-7B-Instruct-v0.2": "7.24B",
    "Qwen/Qwen2.5-7B-Instruct": "7.62B",
}


def settings():
    return [s for s in ORDER if os.path.exists(f"{OUT}/{s}/layers.parquet")]


def guard_missing_features(force=False):
    """Refuse to regenerate the appendix when feature directories have gone missing.

    Every table here is computed from sinktda_out/<setting>/layers.parquet, and the rows
    cannot be rebuilt from sinktda_results/: the LaTeX row needs the share of rows with
    P1>0, R^2 and the Spearman correlation, none of which scaling_real_attention.csv
    stores. Running with a subset therefore silently rewrites the appendix and
    scaling_real_attention.csv with fewer settings than the paper reports, and the loss is
    not recoverable. Pass --force only if a shorter table is genuinely what you want.
    """
    have = set(settings())
    f = f"{RES}/scaling_real_attention.csv"
    known = set(pd.read_csv(f)["setting"]) if os.path.exists(f) else set()
    lost = sorted(known - have)
    if lost and not force:
        raise SystemExit(
            "appendix_extra: refusing to run.\n"
            f"  {len(lost)} setting(s) are in {f} but have no features in {OUT}/:\n"
            "    " + ", ".join(lost) + "\n"
            "  Their appendix rows cannot be rebuilt from sinktda_results/ (the table needs\n"
            "  per-row quantities the CSV does not keep), so regenerating now would drop them\n"
            "  from the appendix and from scaling_real_attention.csv permanently.\n"
            "  Restore those sinktda_out/ directories, or pass --force to accept a shorter table.")


def omitted():
    """Settings in the paper whose per-example layer features are not on this machine;
    tables built from layers.parquet say so instead of silently shrinking."""
    f = f"{RES}/summary_theory.csv"
    known = pd.read_csv(f)["setting"].tolist() if os.path.exists(f) else []
    have = set(settings())
    return [s for s in ORDER if s in known and s not in have]


def omitted_note():
    o = omitted()
    if not o:
        return ""
    n = len(pd.read_csv(f"{RES}/summary_theory.csv"))
    return (f" Rows need per-example layer features, which we retain for {n - len(o)} of the {n} settings "
            f"(not {', '.join(short(x) for x in o)}).")


def load(s):
    d = pd.read_parquet(f"{OUT}/{s}/layers.parquet")
    d["y"] = (d["label"] == "hallucinated").astype(int)
    return d


# ---------------------------------------------------------------------------
# (layers, heads, KV heads, hidden) from each model's config.json, used when the config is
# not in the local Hugging Face cache, so an offline run cannot drop a row from the table
MODEL_CONFIGS = {
    "Qwen/Qwen2.5-3B-Instruct": (36, 16, 2, 2048), "Qwen/Qwen2.5-1.5B-Instruct": (28, 12, 2, 1536),
    "microsoft/Phi-3-mini-4k-instruct": (32, 32, 32, 3072), "TinyLlama/TinyLlama-1.1B-Chat-v1.0": (22, 32, 4, 2048),
    "HuggingFaceTB/SmolLM-1.7B-Instruct": (24, 32, 32, 2048), "mistralai/Mistral-7B-Instruct-v0.2": (32, 32, 8, 4096),
    "Qwen/Qwen2.5-7B-Instruct": (28, 28, 4, 3584),
}


def table_models():
    from transformers import AutoConfig
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Model & params & layers & heads & KV heads & hidden \\", r"\midrule"]
    for mid, p in MODEL_SPECS.items():
        try:
            c = AutoConfig.from_pretrained(mid)
            cfg = (c.num_hidden_layers, c.num_attention_heads,
                   getattr(c, "num_key_value_heads", c.num_attention_heads), c.hidden_size)
        except Exception:
            cfg = MODEL_CONFIGS[mid]
        rows.append(f"\\texttt{{{mid}}} & {p} & " + " & ".join(map(str, cfg)) + " \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def table_scaling():
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Setting & rows with $P_1{>}0$ & $\hat\alpha$ (class FE) & 95\% CI & $R^2$ & Spearman$(P_1,N)$ \\",
            r"\midrule"]
    out = []
    for s in settings():
        d = load(s)
        P1 = d[cols(d, ["h1_total_persistence"])].sum(1).values
        N = d["seq_len"].values.astype(float)
        m = P1 > 0
        if m.sum() < 30:
            rows.append(f"{short(s)} & {m.mean() * 100:.1f}\\% & -- & -- & -- & -- \\\\")
            continue
        X = np.column_stack([np.ones(m.sum()), np.log(N[m]), d["y"].values[m]])
        yv = np.log(P1[m])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = len(yv) - X.shape[1]
        cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
        a = beta[1]
        se = np.sqrt(cov[1, 1])
        lo, hi = a - 1.96 * se, a + 1.96 * se
        r2 = 1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean()))
        rho = spearmanr(P1, N)[0]
        out.append(dict(setting=s, alpha=a, lo=lo, hi=hi))
        rows.append(f"{short(s)} & {m.mean() * 100:.1f}\\% & {a:.2f} & [{lo:.2f}, {hi:.2f}] & {r2:.2f} & {rho:.2f} \\\\")
    rows += [r"\midrule", f"i.i.d.\\ uniform weights (simulation) & 100\\% & {IID_ALPHA:.2f} & -- & -- & -- \\\\",
             f"random causal attention, no sink (simulation) & 100\\% & {CAUSAL_ALPHA:.2f} & -- & -- & -- \\\\",
             r"\bottomrule", r"\end{tabular}"]
    pd.DataFrame(out).to_csv(f"{RES}/scaling_real_attention.csv", index=False)
    return "\n".join(rows), pd.DataFrame(out)


def scaling_text(scal):
    slow = scal[scal["hi"] < IID_ALPHA]
    fast = scal[scal["hi"] >= IID_ALPHA]
    lin = scal[scal["lo"] > 1.0]
    txt = (f"Exponents range from {scal['alpha'].min():.2f} to {scal['alpha'].max():.2f}, against {IID_ALPHA:.2f} for i.i.d.\\ "
           f"uniform weights and {CAUSAL_ALPHA:.2f} for random causal attention without a sink over the same range of $N$. "
           f"{len(slow)} of the {len(scal)} fitted settings grow significantly more slowly than the i.i.d.\\ reference")
    if len(fast):
        txt += " (not " + ", ".join(short(x) for x in fast["setting"]) + ")"
    txt += "."
    if len(lin):
        txt += (" The exponent exceeds $1$ in " + _join_names([short(x) for x in lin["setting"]])
                + ", so growth on real attention is not uniformly sub-linear.")
    skipped = [x for x in settings() if x not in set(scal["setting"])]
    if skipped:
        txt += (" " + _join_names([short(x) for x in skipped]) + " have too few rows with $P_1>0$ to fit (first column): "
                r"almost all of their graphs are coned (Table~\ref{tab:theory}), so $1$D persistence is zero for "
                r"almost every example whatever its length.")
    txt += (r" In the other settings almost every example has some $1$D bar summed over layers, because those models "
            r"have some layers that are not coned; within an example, the coned layers contribute nothing to $P_1$ "
            r"whatever the length (Table~\ref{tab:theory}).")
    return txt


def table_lengths():
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Setting & length $N$ (correct) & length $N$ (hallucinated) & answer tokens (c./h.) & point-biserial $r$ & Welch $p$ \\",
            r"\midrule"]
    for s in settings():
        d = load(s)
        g, h = d[d.y == 0], d[d.y == 1]
        r = pointbiserialr(d["y"], d["seq_len"])[0]
        p = ttest_ind(g["seq_len"], h["seq_len"], equal_var=False).pvalue
        # one rule for every cell: two decimals down to 0.01, otherwise scientific to two
        # significant figures. Mixing "0.00013" with "$<10^{-9}$" in one column made the
        # small values hard to compare.
        ps = f"{p:.2f}" if p >= 0.01 else "$" + f"{p:.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}$"
        rows.append(f"{short(s)} & ${g.seq_len.mean():.1f}\\pm{g.seq_len.std():.1f}$ & "
                    f"${h.seq_len.mean():.1f}\\pm{h.seq_len.std():.1f}$ & "
                    f"{g.answer_len.mean():.1f} / {h.answer_len.mean():.1f} & {r:+.3f} & {ps} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


TOST_TESTS = [("T1_reduction", "Sink vs.\\ 0D"), ("T2_topo_beyond_sink", "0D $\\mid$ Sink"),
              ("T4_1D_beyond_0D", "1D $\\mid$ 0D"), ("I_0D_beyond_nontopo", "0D $\\mid$ NT")]


def table_tost():
    c = pd.read_csv(f"{RES}/summary_comp.csv")
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{{n}}}" for _, n in TOST_TESTS)
    sub = " & ".join(["{\\scriptsize .010} & {\\scriptsize .015} & {\\scriptsize .020} & {\\scriptsize .025}"] * len(TOST_TESTS))
    rows = [r"\begin{tabular}{l" + "cccc" * len(TOST_TESTS) + "}", r"\toprule",
            f"Setting & {head} \\\\", f"$m$ & {sub} \\\\", r"\midrule"]
    # built from summary_comp.csv, so every evaluated setting has a row
    for s in [x for x in ORDER if x in set(c["setting"])]:
        cells = []
        for t, _ in TOST_TESTS:
            r = c[(c.setting == s) & (c.test == t)]
            for e in ("0.01", "0.015", "0.02", "0.025"):
                if r.empty:
                    cells.append("--")
                else:
                    p = float(r.iloc[0][f"tost_p_{e}"])
                    txt = "<.001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
                    cells.append(f"\\textbf{{{txt}}}" if p < 0.05 else txt)
        rows.append(f"{short(s)} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def fig_layer_auc():
    """Univariate per-layer AUC (orientation-free: max(AUC, 1-AUC)) by depth, TruthfulQA and on-policy."""
    feats = [("h0_total_persistence", "$P_0$ (0D total)"), ("star_tot", "star weight $S_0$"),
             ("h1_total_persistence", "$P_1$ (1D total)")]
    groups = [("truthfulqa", "TruthfulQA"), ("triviaqa", "TriviaQA (on-policy)")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharey=True, sharex=True)
    for gi, (pref, gname) in enumerate(groups):
        # the chat-format Mistral twin differs by one token and would overplot its sibling
        ss = [s for s in settings() if s.startswith(pref) and not s.endswith("_chat")]
        for fi, (f, fname) in enumerate(feats):
            a = axes[gi, fi]
            for k, s in enumerate(ss):
                d = load(s)
                L = n_layers(d)
                au = []
                for l in range(L):
                    v = d[f"layer_{l}_{f}"].values
                    x = fast_auc(d["y"].values, v) if np.ptp(v) > 0 else 0.5
                    au.append(max(x, 1 - x))
                a.plot(np.arange(L) / (L - 1), au, color=SERIES[k % len(SERIES)], lw=1.2,
                       label=PRETTY[s][1])
            a.axhline(0.5, color=INK2, lw=0.6)
            if gi == 0:
                a.set_title(fname, fontsize=8)
            if fi == 0:
                a.set_ylabel(f"{gname}\nper-layer AUC", fontsize=7.5)
            if gi == 1:
                a.set_xlabel("relative layer depth")
        axes[gi, 2].legend(fontsize=6, loc="upper right")
    fig.savefig(f"{PAPER}/fig_sink_layer_auc.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def worked_example(example_id=67, label="hallucinated", layers=(5, 21)):
    import ripser
    import torch
    from sinktda import data
    from sinktda.extract import load as load_model
    from sinktda.features import coning_defects, delta_bos_causal, distance_from_attention

    tok, model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mps", torch.bfloat16)
    rows = [r for r in data.truthfulqa_rows(tok, "tinyllama", example_id + 1)
            if r["example_id"] == example_id and r["label"] == label]
    r = rows[0]
    enc = tok(r["full"], return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    fig, axes = plt.subplots(len(layers), 2, figsize=(7.0, 2.7 * len(layers)),
                             gridspec_kw=dict(width_ratios=[1, 1.3]))
    lines = []
    for i, l in enumerate(layers):
        A = out.attentions[l][0].float().mean(0).cpu().numpy().astype(np.float64)
        D = distance_from_attention(A)
        N = A.shape[0]
        dg = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"]
        h0 = dg[0][np.isfinite(dg[0][:, 1])]
        h1 = dg[1]
        d0 = delta_bos_causal(A)
        dm = coning_defects(D)
        P0 = float((h0[:, 1] - h0[:, 0]).sum())
        S = float(D[0, 1:].sum())
        life1 = (h1[:, 1] - h1[:, 0]) if len(h1) else np.array([])
        lines.append(dict(layer=l, N=N, sink=float(A[1:, 0].mean()), delta0=d0, delta_min=float(dm.min()),
                          P0=P0, S=S, n_h1=int(len(h1)), max_h1=float(life1.max()) if len(life1) else 0.0))
        ax = axes[i, 0]
        ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"layer {l}: head-averaged attention", fontsize=8)
        ax.set_xlabel("key token"); ax.set_ylabel("query token")
        ax.grid(False)
        ax = axes[i, 1]
        for j, (b, e) in enumerate(sorted(h0.tolist(), key=lambda x: x[1])):
            ax.plot([b, e], [j, j], color=SERIES[0], lw=1.2)
        off = len(h0) + 1
        for j, (b, e) in enumerate(h1.tolist()):
            ax.plot([b, e], [off + j, off + j], color=SERIES[1], lw=2.0)
        ax.set_title(f"barcode: $\\delta_0$={d0:.3f}, $P_0$={P0:.2f}, $S_0$={S:.2f}, "
                     f"{len(h1)} $H_1$ bar(s)", fontsize=8)
        ax.set_xlabel("filtration scale $\\epsilon$")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{PAPER}/fig_sink_example.pdf")
    allr = []
    for l in range(len(out.attentions)):
        A = out.attentions[l][0].float().mean(0).cpu().numpy().astype(np.float64)
        dg = ripser.ripser(distance_from_attention(A), distance_matrix=True, maxdim=1)["dgms"]
        allr.append(dict(layer=l, sink=float(A[1:, 0].mean()), delta0=delta_bos_causal(A), n_h1=int(len(dg[1]))))
    pd.DataFrame(allr).to_csv(f"{RES}/worked_example_layers.csv", index=False)
    plt.close(fig)
    pd.DataFrame(lines).to_csv(f"{RES}/worked_example.csv", index=False)
    return r, lines


def example_text():
    p = f"{RES}/worked_example.csv"
    if not os.path.exists(p):
        return ""
    e = pd.read_csv(p)
    a, b = e.iloc[0], e.iloc[1]
    return (r"\paragraph{Worked example.} Figure~\ref{fig:example} shows TruthfulQA question 67 with its incorrect answer "
            r"(``Yes, vampires are real'') under TinyLlama-1.1B ($N{=}" + f"{int(a.N)}" + r"$). "
            f"In layer {int(a.layer)} the model sends {a.sink:.2f} of its attention to token 0 on average, "
            f"$\\defect_0={a.delta0:.3f}$, and accordingly $P_0={a.P0:.3f}$ equals the star weight $\\stw_0={a.S:.3f}$ "
            f"and there are {int(a.n_h1)} $H_1$ bars. "
            f"In layer {int(b.layer)} sink attention drops to {b.sink:.2f} and $\\defect_0={b.delta0:.3f}$; "
            f"now $P_0={b.P0:.3f}$ falls below $\\stw_0={b.S:.3f}$ (within the bound $(N{{-}}2)\\defect_0={(b.N - 2) * b.delta0:.2f}$), "
            f"and {int(b.n_h1)} $H_1$ bars appear, the longest of length {b.max_h1:.3f}$\\,\\le\\defect_0$. "
            + no_force_text() + "\n")


def no_force_text():
    p = f"{RES}/worked_example_layers.csv"
    if not os.path.exists(p):
        return ""
    e = pd.read_csv(p)
    q = e[(e.delta0 > 0.05) & (e.n_h1 == 0)]
    if q.empty:
        return ""
    runs, cur = [], []
    for x in (int(v) for v in q.layer):
        if cur and x == cur[-1] + 1:
            cur.append(x)
        else:
            cur and runs.append(cur)
            cur = [x]
    runs.append(cur)
    parts = [f"{r[0]}" if len(r) == 1 else f"{r[0]}--{r[-1]}" for r in runs]
    ls = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return (r"The theorem bounds loops but does not force them: in layer" + ("s " if len(q) > 1 else " ") + ls
            + f" of the same example $\\defect_0$ lies between ${q.delta0.min():.2f}$ and ${q.delta0.max():.2f}$, yet $H_1$ is empty.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if some settings no longer have features")
    args = ap.parse_args()
    guard_missing_features(args.force)
    style()
    if args.examples:
        worked_example()
    scal_tex, scal = table_scaling()
    fig_layer_auc()
    has_ex = os.path.exists(f"{PAPER}/fig_sink_example.pdf")
    with open(f"{PAPER}/sink_appendix_extra.tex", "w") as fh:
        fh.write(r"\section{Additional analyses}" + "\n" + r"\label{app:extra}" + "\n\n")
        fh.write(r"\paragraph{Models.} Table~\ref{tab:models} lists the model architectures. All models use grouped or "
                 r"multi-head attention; we average attention over query heads.""\n")
        fh.write(r"\begin{table}[H]\centering\small\caption{\textbf{Models.}}\label{tab:models}" + "\n"
                 + r"\resizebox{\linewidth}{!}{" + table_models() + "}\n" + r"\end{table}" + "\n\n")
        if len(scal):
            fh.write(r"\paragraph{Length scaling of $1$D persistence on real attention.} "
                     r"Table~\ref{tab:scaling} regresses log total $1$D persistence (summed over layers) on $\log N$ with a "
                     r"class fixed effect, on rows with $P_1>0$, so that the exponent is not driven by a length difference between "
                     r"correct and hallucinated answers. "
                     + scaling_text(scal) + "\n")
            fh.write(r"\begin{table}[H]\centering\small\caption{\textbf{Length scaling of $1$D persistence.} "
                     r"OLS of $\log P_1$ on $\log N$ and the class label, fit on the rows with $P_1>0$; the first column is the "
                     r"share of rows that are retained, and the rows with $P_1=0$ are dropped from the fit." + omitted_note() + "}"
                     r"\label{tab:scaling}" + "\n" + r"\resizebox{\linewidth}{!}{" + scal_tex + "}\n" + r"\end{table}" + "\n\n")
        fh.write(r"\paragraph{Sequence lengths.} Table~\ref{tab:lengths} gives class-conditional lengths. TruthfulQA is "
                 r"length-balanced by construction; HaluEval's hallucinated answers are systematically longer; on-policy, "
                 r"incorrect answers differ in length from correct ones because they are the model's own.""\n")
        fh.write(r"\begin{table}[H]\centering\small\caption{\textbf{Class-conditional sequence length} (mean $\pm$ SD, full sequence) "
                 r"and mean answer length." + omitted_note() + r"}\label{tab:lengths}" + "\n" + r"\resizebox{\linewidth}{!}{" + table_lengths() + "}\n" + r"\end{table}" + "\n\n")
        fh.write(r"\paragraph{Margin sensitivity.} Table~\ref{tab:tost} reports TOST $p$-values at four equivalence margins for "
                 r"the four comparisons of Figure~\ref{fig:forest}; bold marks $p<0.05$ (equivalence).""\n")
        fh.write(r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{2.2pt}\caption{\textbf{TOST $p$-values by margin $m$.} "
                 r"NT: all non-topological features.}\label{tab:tost}" + "\n" + r"\resizebox{\linewidth}{!}{" + table_tost() + "}\n" + r"\end{table}" + "\n\n")
        fh.write(r"\paragraph{Per-layer signal.} Figure~\ref{fig:layerauc} shows single-feature AUCs by depth. $P_0$ and the "
                 r"star weight trace nearly identical curves in every model with a sink, while $P_1$ carries signal only in the "
                 r"layers that are not coned. These AUCs are orientation-free ($\max(\mathrm{AUC},1-\mathrm{AUC})$) and therefore "
                 r"slightly optimistic; they are descriptive, not a detector.""\n")
        fh.write(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{fig_sink_layer_auc.pdf}"
                 r"\caption{\textbf{Single-feature AUC by layer.}" + omitted_note().replace("Rows need", "Curves need")
                 + r" Mistral-7B is shown in its \texttt{[INST]} format only.}\label{fig:layerauc}\end{figure}" + "\n\n")
        if has_ex:
            fh.write(example_text())
            fh.write(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{fig_sink_example.pdf}"
                     r"\caption{\textbf{A coned and a non-coned layer of the same example} (TinyLlama-1.1B, TruthfulQA). "
                     r"Left: head-averaged attention; the bright first column is the sink. Right: $0$D bars (blue) and $1$D bars "
                     r"(orange) of the Vietoris--Rips filtration of $D=1-\max(A,A^\top)$.}\label{fig:example}\end{figure}" + "\n")
    print("wrote", f"{PAPER}/sink_appendix_extra.tex")


if __name__ == "__main__":
    main()
