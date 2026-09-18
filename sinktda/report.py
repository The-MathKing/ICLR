"""
report.py -- aggregate sinktda_results/ into paper tables (LaTeX) and figures.

  python -m sinktda.report
Writes paper/sink_tables.tex and paper/fig_sink_*.pdf
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "sinktda_results"
PAPER = "paper"
OUT = os.environ.get("SINKTDA_OUT", "sinktda_out")
MIRROR_LIMIT_MB = 8  # anonymous.4open.science refuses anything larger

# reference categorical order (dataviz palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
          # appended for the 7--8B settings; indices 0--7 are unchanged, so every existing
          # figure keeps its colours
          "#00838f", "#8d6e00"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = "Blues"

PRETTY = {
    "truthfulqa_qwen3b": ("TruthfulQA", "Qwen2.5-3B"),
    "truthfulqa_qwen1.5b": ("TruthfulQA", "Qwen2.5-1.5B"),
    "truthfulqa_phi3": ("TruthfulQA", "Phi-3-mini"),
    "truthfulqa_tinyllama": ("TruthfulQA", "TinyLlama-1.1B"),
    "truthfulqa_smollm": ("TruthfulQA", "SmolLM-1.7B"),
    "truthfulqa_mistral": ("TruthfulQA", "Mistral-7B"),
    "truthfulqa_mistral_chat": ("TruthfulQA", "Mistral-7B (chat)"),
    "truthfulqa_qwen7b": ("TruthfulQA", "Qwen2.5-7B"),
    "truthfulqa_llama8b": ("TruthfulQA", "Llama-3.1-8B"),
    "halueval_qwen3b": ("HaluEval", "Qwen2.5-3B"),
    "halueval_qwen1.5b": ("HaluEval", "Qwen2.5-1.5B"),
    "triviaqa_qwen3b": ("TriviaQA (on-policy)", "Qwen2.5-3B"),
    "triviaqa_qwen1.5b": ("TriviaQA (on-policy)", "Qwen2.5-1.5B"),
    "triviaqa_phi3": ("TriviaQA (on-policy)", "Phi-3-mini"),
    "triviaqa_tinyllama": ("TriviaQA (on-policy)", "TinyLlama-1.1B"),
    "triviaqa_mistral": ("TriviaQA (on-policy)", "Mistral-7B"),
    "triviaqa_qwen7b": ("TriviaQA (on-policy)", "Qwen2.5-7B"),
    "triviaqa_llama8b": ("TriviaQA (on-policy)", "Llama-3.1-8B"),
}
ORDER = list(PRETTY)

# Nominal parameter counts (B) for the model labels in PRETTY, used for the
# model-size range macro so the abstract/limitations never hardcode a range.
PARAMS_B = {
    "TinyLlama-1.1B": 1.1, "Qwen2.5-1.5B": 1.5, "SmolLM-1.7B": 1.7,
    "Qwen2.5-3B": 3.0, "Phi-3-mini": 3.8,
    "Mistral-7B": 7.0, "Qwen2.5-7B": 7.0, "Llama-3.1-8B": 8.0,
}


def _size(x):
    return f"{x:g}B"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
    })


def label(s):
    b, m = PRETTY.get(s, (s, ""))
    return f"{b} / {m}"


def load(kind):
    fs = sorted(glob.glob(f"{RES}/{kind}_*.csv"))
    fs = [f for f in fs if not os.path.basename(f).startswith(f"{kind}_layers_")] if kind == "theory" else fs
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["_o"] = df["setting"].map({s: i for i, s in enumerate(ORDER)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o")


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def fdelta(r):
    return f"${r['delta']:+.3f}$ {{\\scriptsize[{r['ci90_lo']:+.3f}, {r['ci90_hi']:+.3f}]}}"


# ---------------------------------------------------------------------------
def table_theory(th):
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Benchmark & Model & $\bar N$ & graphs & viol. & sink attn. & coned & $H_1{=}\emptyset\,|\,$coned & $\rho_{\text{raw}}$ & $\rho_{/N}$ \\",
             r"\midrule"]
    for _, r in th.iterrows():
        b, m = PRETTY.get(r["setting"], (r["setting"], ""))
        hz = "--" if pd.isna(r["frac_h1zero_given_coned"]) else f"{100 * r['frac_h1zero_given_coned']:.1f}\\%"
        lines.append(f"{b} & {m} & {r['mean_N']:.1f} & {int(r['cells']):,} & "
                     f"{int(r['bound_violations_h1'] + r['bound_violations_p0'] + r.get('bound_violations_maxdeath', 0))} & "
                     f"{r['mean_sink_mass']:.2f} & {100 * r['frac_cells_coned']:.1f}\\% & {hz} & "
                     f"{r['median_layer_spearman_P0_star']:.3f} & {r['median_layer_spearman_P0_star_norm']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


MAIN_BANKS = [("LEX", "Lex."), ("LOGPROB", "LogP"), ("LLMCHECK", "LLM-C."),
              ("SINK", "Sink"), ("0D", "0D"), ("1D", "1D"), ("PH_SINK", "Sink$_h$"),
              ("PH_0D", "0D$_h$"), ("LOOKBACK", "Lookb."), ("HIDDEN", "Hidden"),
              ("NONTOPO", "Non-topo.")]


def _grouped(df):
    prev = None
    for s, g in df.groupby("setting", sort=False):
        b = PRETTY.get(s, (s, ""))[0]
        yield s, g, (prev is not None and b != prev)
        prev = b


def table_auc(auc):
    head = " & ".join(n for _, n in MAIN_BANKS)
    lines = [r"\begin{tabular}{l" + "c" * len(MAIN_BANKS) + "}", r"\toprule",
             f"Setting & {head} \\\\", r"\midrule"]
    for s, g, brk in _grouped(auc):
        if brk:
            lines.append(r"\midrule")
        d = dict(zip(g["bank"], g["auc"]))
        vals = [d.get(k, np.nan) for k, _ in MAIN_BANKS]
        best = np.nanmax(vals)
        fmt = lambda v: "--" if pd.isna(v) else (f"{v:.3f}"[1:] if f"{v:.3f}".startswith("0") else f"{v:.2f}")
        cells = [("\\textbf{" + fmt(v) + "}") if v == best else fmt(v) for v in vals]
        lines.append(f"{short(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


KEY_TESTS = [("T1_reduction", r"Sink$-$0D"), ("T2_topo_beyond_sink", r"0D$\mid$Sink"),
             ("PH_topo_beyond_sink", r"$P_{0,h}\mid$Sink$_h$"),
             ("T3_deflated_beyond_sink", r"Defl.$\mid$Sink"),
             ("T4_1D_beyond_0D", r"1D$\mid$0D"),
             ("I_0D_beyond_nontopo", r"0D$\mid$NT"),
             ("LF_0D_beyond_NONTOPO", r"0D$\oplus$NT"),
             ("I_defl_beyond_nontopo", r"Defl.$\mid$NT")]


def table_tests(comp):
    lines = [r"\begin{tabular}{l" + "c" * len(KEY_TESTS) + "}", r"\toprule",
             "Setting & " + " & ".join(n for _, n in KEY_TESTS) + r" \\", r"\midrule"]
    for s, g, brk in _grouped(comp):
        if brk:
            lines.append(r"\midrule")
        d = {r["test"]: r for _, r in g.iterrows()}
        cells = []
        for t, _ in KEY_TESTS:
            if t not in d:
                cells.append("--")
                continue
            r = d[t]
            mark = r"$^{\equiv}$" if r["equiv_0.015"] else ("$^{*}$" if (r["ci95_lo"] > 0 or r["ci95_hi"] < 0) else "")
            cells.append(f"${r['delta']:+.3f}${mark}".replace("-0.000", "0.000").replace("+0.000", "0.000"))
        lines.append(f"{short(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


SHORT_B = {"TruthfulQA": "TQA", "HaluEval": "HE", "TriviaQA (on-policy)": "TrQA"}


def short(s):
    b, m = PRETTY.get(s, (s, ""))
    return f"{SHORT_B.get(b, b)} {m.replace('Qwen2.5-', 'Qw').replace('-mini', '').replace('TinyLlama-1.1B', 'TinyLl.').replace('SmolLM-1.7B', 'SmolLM').replace('Mistral-7B', 'Mis7B').replace('Llama-3.1-8B', 'Ll8B')}"


def table_tests_full(comp):
    lines = [r"\begin{longtable}{llrrrrcc}", r"\toprule",
             r"Setting & Test (A $\to$ B) & AUC$_A$ & AUC$_B$ & $\Delta$ & 90\% CI & $\equiv_{.015}$ & power \\",
             r"\midrule\endhead"]
    torder = [t for t, *_ in __import__("sinktda.evaluate", fromlist=["COMPARISONS"]).COMPARISONS]
    comp = comp.assign(_s=comp["setting"].map({x: i for i, x in enumerate(ORDER)}),
                       _t=comp["test"].map(lambda t: torder.index(t) if t in torder else len(torder)))
    comp = comp.sort_values(["_s", "_t"])
    for _, r in comp.iterrows():
        t = f"{r['A']}$\\to${r['B']}".replace("_", r"\_")
        lines.append(f"{short(r['setting'])} & {t} & {r['auc_A']:.3f} & {r['auc_B']:.3f} & "
                     f"{r['delta']:+.3f} & [{r['ci90_lo']:+.3f}, {r['ci90_hi']:+.3f}] & "
                     f"{'yes' if r['equiv_0.015'] else 'no'} & {r['tost_power_0.015']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def table_auc_full(auc):
    lines = [r"\begin{longtable}{llrrrr}", r"\toprule",
             r"Setting & Bank & dim & AUC & seed SD & pair acc. \\", r"\midrule\endhead"]
    order = ["LEN", "LEX", "LOGPROB", "LLMCHECK", "ROWSTAT", "SINK", "0D", "LEGACY_MSTPROXY", "1D", "DEFL", "ANS",
             "PH_SINK", "PH_0DTOT", "PH_0D", "PH_ENT", "LOOKBACK", "HIDDEN", "NONTOPO"]
    auc = auc.assign(_s=auc["setting"].map({x: i for i, x in enumerate(ORDER)}),
                     _b=auc["bank"].map(lambda b: order.index(b) if b in order else len(order)))
    auc = auc.sort_values(["_s", "_b", "bank"])
    for _, r in auc.iterrows():
        dim = "--" if pd.isna(r["dim"]) else f"{int(r['dim'])}"
        lines.append(f"{short(r['setting'])} & {r['bank'].replace('_', ' ')} & {dim} & {r['auc']:.3f} & "
                     f"{r['auc_seed_sd']:.4f} & {f3(r['within_pair_acc'])} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def fig_layers(lay):
    if lay.empty:
        return
    tq = [s for s in ORDER if s in set(lay["setting"]) and s.startswith("truthfulqa")]
    if len(tq) > len(SERIES):
        # never drop a model from the figure without saying so
        print(f"[fig_layers] WARNING: {len(tq)} TruthfulQA settings but only {len(SERIES)} "
              f"series colours; dropping {tq[len(SERIES):]}")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    fig.subplots_adjust(wspace=0.3)
    for i, s in enumerate(tq[:len(SERIES)]):
        g = lay[lay["setting"] == s]
        c = SERIES[i % len(SERIES)]
        axes[0].plot(g["depth"], g["frac_coned"], color=c, label=label(s))
        axes[1].plot(g["depth"], g["mean_sink_mass"], color=c)
        axes[2].plot(g["depth"], g["spearman_P0_star_norm"], color=c)
    axes[0].set_title("fraction exactly coned ($\\delta_0{=}0$)", fontsize=8)
    axes[1].set_title("mean attention to token 0", fontsize=8)
    axes[2].set_title("Spearman($P_0/(N{-}1)$, $1-\\bar m$)", fontsize=8)
    for a in axes:
        a.set_xlabel("relative layer depth")
    axes[2].set_ylim(-0.1, 1.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.30), fontsize=7.5)
    fig.savefig(f"{PAPER}/fig_sink_layers.pdf")
    plt.close(fig)


def fig_synthetic():
    dose = f"{RES}/synthetic_sink_dose.csv"
    plant = f"{RES}/synthetic_planted_cycle.csv"
    morse = f"{RES}/synthetic_morse_bound.csv"
    if not (os.path.exists(dose) and os.path.exists(plant) and os.path.exists(morse)):
        return
    d, p, m = pd.read_csv(dose), pd.read_csv(plant), pd.read_csv(morse)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    fig.subplots_adjust(wspace=0.35)
    a = axes[0]
    a.plot(m["N"], m["mean_P1"], color=SERIES[0], marker="o", ms=3, label="simulated $\\mathbb{E}[P_1]$")
    a.plot(m["N"], m["morse_lower_bound"], color=SERIES[1], marker="s", ms=3, label="Morse lower bound (Prop. 2)")
    a.set_xscale("log"); a.set_yscale("log")
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    a.set_xlabel("$N$ (i.i.d. uniform weights)"); a.set_title("total 1D persistence", fontsize=8)
    a.legend(fontsize=6.5)
    a = axes[1]
    a.plot(d["b"], d["frac_coned"], color=SERIES[0], marker="o", ms=3, label="fraction coned")
    a.plot(d["b"], d["frac_h1_zero"], color=SERIES[1], marker="s", ms=3, label="fraction $H_1$ empty")
    a.plot(d["b"], d["corr_P0_star"], color=SERIES[2], marker="^", ms=3, label="corr($P_0$, star)")
    a.set_xlabel("sink logit bias $b$"); a.set_ylim(-0.02, 1.02); a.set_title("sink dose-response", fontsize=8)
    a.legend(fontsize=6.5)
    a = axes[2]
    piv = p.pivot(index="b", columns="gamma", values="auc_h1")
    im = a.imshow(piv.values, origin="lower", cmap=BLUES, vmin=0.5, vmax=1.0, aspect="auto")
    a.set_xticks(range(len(piv.columns)), piv.columns); a.set_yticks(range(len(piv.index)), piv.index)
    a.set_xlabel("planted-cycle strength $\\gamma$"); a.set_ylabel("sink bias $b$"); a.grid(False); a.set_title("1D detects a planted cycle?", fontsize=8)
    for (i, j), v in np.ndenumerate(piv.values):
        a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.8 else INK)
    cb = fig.colorbar(im, ax=a, fraction=0.05); cb.set_label("1D detection AUC", fontsize=7)
    fig.savefig(f"{PAPER}/fig_sink_synthetic.pdf")
    plt.close(fig)


def fig_forest(comp):
    if comp.empty:
        return
    tests = [("T1_reduction", "Sink − 0D"), ("T2_topo_beyond_sink", "0D beyond sink"),
             ("T4_1D_beyond_0D", "1D beyond 0D"), ("I_0D_beyond_nontopo", "0D beyond non-topo.")]
    settings = [s for s in ORDER if s in set(comp["setting"])]
    fig, axes = plt.subplots(1, len(tests), figsize=(7.0, 0.28 * len(settings) + 0.8), sharey=True)
    for k, (t, name) in enumerate(tests):
        a = axes[k]
        a.axvspan(-0.015, 0.015, color=GRID, lw=0)
        a.axvline(0, color=INK2, lw=0.6)
        for i, s in enumerate(settings):
            r = comp[(comp["setting"] == s) & (comp["test"] == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            a.plot([r["ci90_lo"], r["ci90_hi"]], [i, i], color=SERIES[0], lw=1.6, solid_capstyle="round")
            a.plot(r["delta"], i, "o", color=SERIES[0], ms=4, mec="white", mew=0.8)
        a.set_title(name, fontsize=8)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        a.set_xlabel("$\\Delta$AUC (90% CI)")
        a.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(settings)), [label(s) for s in settings], fontsize=6.5)
    axes[0].invert_yaxis()
    fig.savefig(f"{PAPER}/fig_sink_forest.pdf")
    plt.close(fig)


ONP_CACHE = f"{RES}/onpolicy_summary.csv"


def onpolicy_stats():
    """Per on-policy setting: questions, accuracy (%), rows used, mean answer tokens.

    Prefers the live generations.csv/layers.parquet, and falls back to the cached summary
    for settings whose feature directory is no longer on disk. The small-model feature
    directories were lost; their cached rows were parsed from the generated table that the
    original runs produced, so those published values are carried through unchanged rather
    than silently dropped from the table. Live readings refresh the cache.
    """
    cache = pd.read_csv(ONP_CACHE).set_index("setting") if os.path.exists(ONP_CACHE) else None
    out, live = {}, []
    for s in ORDER:
        if not s.startswith("triviaqa"):
            continue
        g, lp = f"sinktda_out/{s}/generations.csv", f"sinktda_out/{s}/layers.parquet"
        if os.path.exists(g) and os.path.exists(lp):
            gen = pd.read_csv(g, keep_default_na=False)
            lay = pd.read_parquet(lp, columns=["answer_len"])
            out[s] = dict(questions=len(gen), accuracy=100 * gen["correct"].mean(),
                          rows_used=len(lay), mean_answer_tokens=float(lay["answer_len"].mean()))
            live.append(s)
        elif cache is not None and s in cache.index:
            r = cache.loc[s]
            out[s] = dict(questions=int(r["questions"]), accuracy=float(r["accuracy"]),
                          rows_used=int(r["rows_used"]),
                          mean_answer_tokens=float(r["mean_answer_tokens"]))
    if out:  # keep the cache current for settings we can still read
        df = pd.DataFrame([dict(setting=s, **v) for s, v in out.items()])
        df.to_csv(ONP_CACHE, index=False)
    if live and len(live) < len(out):
        print(f"[onpolicy] live: {len(live)}; from cache: {len(out) - len(live)} "
              f"({', '.join(s for s in out if s not in live)})")
    return out


def table_onpolicy():
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & questions & accuracy & rows used & mean answer tokens \\", r"\midrule"]
    for s, v in onpolicy_stats().items():
        lines.append(f"{PRETTY[s][1]} & {v['questions']:,} & {v['accuracy']:.1f}\\% & "
                     f"{v['rows_used']:,} & {v['mean_answer_tokens']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_timing():
    path = f"{RES}/timing.csv"
    if not os.path.exists(path):
        txt = "Timing on real attention is reported in the released \\texttt{sinktda\\_results/timing.csv}."
    else:
        t = pd.read_csv(path)
        r = t[t["kind"] == "real"].set_index("N")
        q = t[t["kind"] == "iid"].set_index("N")
        parts = [f"{r.loc[n, 'sink_ms']:.3f}/{r.loc[n, 'delta0_ms']:.2f}/{r.loc[n, 'mst_ms']:.2f}/"
                 f"{r.loc[n, 'ripser0_ms']:.2f}/{r.loc[n, 'ripser01_ms']:.2f}\\,ms at $N{{=}}{n}$" for n in r.index]
        txt = (f"Median per-matrix wall-clock time on real attention from {t['model'].iloc[0].split('/')[-1]} "
               f"(sink features / $\\defect_0$ / dense MST / ripser $0$D / ripser $0$D+$1$D) is "
               + "; ".join(parts) + ". "
               f"On i.i.d.\\ random matrices of the same size, ripser $0$D+$1$D takes "
               + ", ".join(f"{(format(q.loc[n, 'ripser01_ms'], '.3g') if q.loc[n, 'ripser01_ms'] < 1000 else format(q.loc[n, 'ripser01_ms'], ',.0f'))}\\,ms ($N{{=}}{n}$)" for n in q.index)
               + ". On real attention the same computation is four orders of magnitude cheaper at $N{=}256$, consistent with "
               "Theorem~\\ref{thm:cone}: sink-dominated graphs have few, short $1$D bars to compute. "
               "Sink features, read off one column, are one to two orders of magnitude cheaper still.")
    with open(f"{PAPER}/sink_timing.tex", "w") as fh:
        fh.write(txt + "\n")


def _rng(x, fmt="{:+.3f}"):
    x = pd.Series(x).dropna()
    if x.empty:
        return "--"
    lo, hi = x.min(), x.max()
    f = lambda v: fmt.format(v).replace("-0.000", "0.000").replace("+0.000", "0.000")
    return f(lo) if f(lo) == f(hi) else f"{f(lo)} to {f(hi)}"


def _join_names(names):
    """Oxford-free English list: "A", "A and B", "A, B and C" -- these land in prose."""
    n = sorted(names)
    if not n:
        return "--"
    return n[0] if len(n) == 1 else " and ".join([", ".join(n[:-1]), n[-1]])


def _extremes(df, col, tag, fmt="{:.2f}", scale=1.0):
    """Macros for the smallest and largest value of `col`, each with the model it belongs
    to, so prose can contrast two settings without hardcoding either number or name."""
    if not len(df):
        return {}
    lo, hi = df.loc[df[col].idxmin()], df.loc[df[col].idxmax()]
    name = lambda r: PRETTY.get(r["setting"], (r["setting"], r["setting"]))[1]
    return {tag + "Min": fmt.format(lo[col] * scale), tag + "MinModel": name(lo),
            tag + "Max": fmt.format(hi[col] * scale), tag + "MaxModel": name(hi)}


def _rho_delta_h1_floor(sink):
    """Describe the setting at the bottom of the rho(delta_0, P_1) range.

    C3 predicts that 1D persistence can only appear where the cone is violated, and the
    correlation between delta_0 and P_1 is the evidence. In the most strongly coned
    settings almost every graph has delta_0 = 0 and no 1D bars, so both quantities are
    near-constant and their rank correlation collapses towards zero -- which reads as weak
    support when it is the opposite. These macros let the text say so without naming a
    model or a number that could go stale.
    """
    if not len(sink) or "cell_spearman_delta0_h1" not in sink.columns:
        return {}
    cut = 0.9
    near, rest = sink[sink["frac_cells_coned"] >= cut], sink[sink["frac_cells_coned"] < cut]
    if not len(near) or not len(rest):
        return {}
    return {
        "RhoDeltaHoneNearCut": f"{cut * 100:.0f}\\%",
        "RhoDeltaHoneNearN": str(len(near)),
        "RhoDeltaHoneNearConed": _rng(near["frac_cells_coned"] * 100, "{:.1f}\\%"),
        "RhoDeltaHoneNearHOneZero": _rng(near["frac_cells_h1_zero"] * 100, "{:.2f}\\%"),
        "RhoDeltaHoneNear": _rng(near["cell_spearman_delta0_h1"], "{:.2f}"),
        "RhoDeltaHoneRest": f"{rest['cell_spearman_delta0_h1'].min():.2f}",
    }


def _model_size_range(models):
    """Range of nominal parameter counts over the models actually evaluated.
    Raises rather than guessing, so a new model cannot silently yield a wrong range."""
    missing = [m for m in models if m not in PARAMS_B]
    if missing:
        raise KeyError(f"PARAMS_B has no parameter count for {missing}")
    sizes = sorted(PARAMS_B[m] for m in models)
    if not sizes:
        return "--"
    return _size(sizes[0]) if sizes[0] == sizes[-1] else f"{_size(sizes[0])}--{_size(sizes[-1])}"


def write_numbers(th, auc, comp):
    models = sorted({PRETTY.get(s, (s, s))[1].replace(" (chat)", "") for s in th["setting"]})
    sink = th[th["mean_sink_mass"] > SINK_MASS_MIN]
    nosink = th[th["mean_sink_mass"] <= SINK_MASS_MIN]
    pc = sink["frac_cells_coned"] * 100
    ss = set(sink["setting"])
    c = comp[comp["setting"].isin(ss)]

    def T(test, subset=None):
        d = c[c["test"] == test]
        if subset:
            d = d[d["setting"].str.startswith(subset)]
        return d

    def a(bank, subset):
        d = auc[(auc["bank"] == bank) & auc["setting"].str.startswith(subset)]
        return d["auc"]

    ls = pd.read_csv(f"{RES}/layer_split.csv") if os.path.exists(f"{RES}/layer_split.csv") else pd.DataFrame()
    ls = ls[ls["setting"].isin(ss)] if len(ls) else ls
    m = {
        "NSettings": str(len(th)),
        "NModels": str(len(models)),
        "ModelSizeRange": _model_size_range(models),
        "NCellsTotal": f"{int(th['cells'].sum()):,}",
        "NSinkSettings": str(len(sink)),
        "PctConedRange": f"{pc.min():.0f}--{pc.max():.0f}\\%" if len(pc) else "--",
        "SinkMassRange": _rng(sink["mean_sink_mass"], "{:.2f}"),
        "NoSinkMass": _rng(nosink["mean_sink_mass"], "{:.2f}"),
        "NoSinkModels": _join_names({PRETTY.get(s, (s, s))[1] for s in nosink["setting"]}),
        "NNoSinkModels": str(len({PRETTY.get(s, (s, s))[1] for s in nosink["setting"]})),
        # share of graphs coned at *some* apex: separates a model with no cone structure
        # at all from one whose sink simply is not the first token
        "NoSinkConedAny": _rng(nosink["frac_cells_coned_any_apex"] * 100, "{:.0f}\\%"),
        **_extremes(nosink, "frac_cells_coned_any_apex", "NoSinkConedAny", "{:.0f}\\%", 100),
        "RhoMin": f"{np.floor(sink['median_layer_spearman_P0_star_norm'].min() * 100) / 100:.2f}" if len(sink) else "--",
        "RhoRawMin": f"{np.floor(sink['median_layer_spearman_P0_star'].min() * 1000) / 1000:.3f}" if len(sink) else "--",
        "RhoNoSink": _rng(nosink["median_layer_spearman_P0_star_norm"], "{:.2f}"),
        "RhoNoSinkRaw": _rng(nosink["median_layer_spearman_P0_star"], "{:.3f}"),
        "RhoLayerMinRange": _rng(sink["min_layer_spearman_P0_star_norm"], "{:.2f}"),
        "NoSinkHOneNonempty": _rng((1 - nosink["frac_cells_h1_zero"]) * 100, "{:.0f}\\%"),
        "RhoDeltaHone": _rng(sink["cell_spearman_delta0_h1"], "{:.2f}"),
        # The bottom of that range is the most strongly coned setting, where delta_0 and
        # P_1 are both near-constant, so the rank correlation is degenerate rather than
        # the prediction weak. Derived from whichever setting is the minimum.
        **_rho_delta_h1_floor(sink),
        "NViolations": str(int((th["bound_violations_h1"] + th["bound_violations_p0"] + th.get("bound_violations_maxdeath", 0)).sum())),
        "NSeedSDAbove": str(int((auc["auc_seed_sd"] > 0.004).sum())),
        "MaxSeedSD": f"{auc['auc_seed_sd'].max():.3f}" if len(auc) else "--",
        "NComp": str(c["setting"].nunique()),
        "NTOneEquiv": str(int(T("T1_reduction")["equiv_0.015"].sum())),
        "TOneRange": _rng(T("T1_reduction")["delta"]),
        "TTwoRange": _rng(T("T2_topo_beyond_sink")["delta"]),
        "NTTwoEquiv": str(int(T("T2_topo_beyond_sink")["equiv_0.015"].sum())),
        "TThreeTQA": _rng(T("T3_deflated_beyond_sink", "truthfulqa")["delta"]),
        "TThreeOnp": _rng(T("T3_deflated_beyond_sink", "triviaqa")["delta"]),
        "TThreeHE": _rng(T("T3_deflated_beyond_sink", "halueval")["delta"]),
        "NTFourEquiv": str(int(T("T4_1D_beyond_0D")["equiv_0.015"].sum())),
        "TFourRange": _rng(T("T4_1D_beyond_0D")["delta"]),
        "TFourExceptions": _join_names([short(x) for x in T("T4_1D_beyond_0D").query("not `equiv_0.015`")["setting"]]),
        # Failing the equivalence test is not the same as helping: one of those
        # settings has a negative delta with an interval too wide to certify either
        # way. These two cover the settings where 1D genuinely adds signal.
        "TFourHelps": _join_names([short(x) for x in
                                   T("T4_1D_beyond_0D").query("ci95_lo > 0")["setting"]]),
        "TFourHelpsAllN": str(int((comp[comp["test"] == "T4_1D_beyond_0D"]["ci95_lo"] > 0).sum())),
        "PHTopoRange": _rng(T("PH_topo_beyond_sink")["delta"]),
        "NPHTopoEquiv": str(int(T("PH_topo_beyond_sink")["equiv_0.015"].sum())),
        "NPHTopoTotal": str(len(T("PH_topo_beyond_sink"))),
        "PHEntRange": _rng(comp[comp["test"] == "PH_0D_vs_entropy"]["delta"]),
        "NIZeroEquiv": str(int(comp[comp["test"] == "I_0D_beyond_nontopo"]["equiv_0.015"].sum())),
        "NIZeroTotal": str(int((comp["test"] == "I_0D_beyond_nontopo").sum())),
        "IZeroRange": _rng(comp[comp["test"] == "I_0D_beyond_nontopo"]["delta"]),
        "NSettingsNT": str(int((comp["test"] == "I_0D_beyond_nontopo").sum())),
        "NLogpZeroDTQA": str(int(((c0 := comp[(comp["test"] == "B_logprob") & comp["setting"].str.startswith("truthfulqa")])["ci95_hi"] < 0).sum())),
        "NLogpZeroDTQATotal": str(len(c0)),
        "NLogpBeatsZeroDTQA": str(int((c0["delta"] < 0).sum())),
        "NLogpOnpSig": str(int(((c1 := comp[(comp["test"] == "B_logprob") & comp["setting"].str.startswith("triviaqa")])["ci95_lo"] > 0).sum())),
        "NLogpOnpTotal": str(len(c1)),
        "NDeflOnpSig": str(int((T("T3_deflated_beyond_sink", "triviaqa")["ci95_lo"] > 0).sum())),
        "NDeflOnpTotal": str(len(T("T3_deflated_beyond_sink", "triviaqa"))),
        "NDeflTQASig": str(int((T("T3_deflated_beyond_sink", "truthfulqa")["ci95_lo"] > 0).sum())),
        "NDeflTQATotal": str(len(T("T3_deflated_beyond_sink", "truthfulqa"))),
        "NIDeflEquiv": str(int(comp[comp["test"] == "I_defl_beyond_nontopo"]["equiv_0.015"].sum())),
        "IDeflRange": _rng(comp[comp["test"] == "I_defl_beyond_nontopo"]["delta"]),
        "LexTQA": _rng(a("LEX", "truthfulqa"), "{:.2f}"),
        "LexOnp": _rng(a("LEX", "triviaqa"), "{:.2f}"),
        "LexHE": _rng(a("LEX", "halueval"), "{:.2f}"),
        "LogpTQA": _rng(a("LOGPROB", "truthfulqa"), "{:.2f}"),
        "LogpOnp": _rng(a("LOGPROB", "triviaqa"), "{:.2f}"),
        "ZeroDTQA": _rng(a("0D", "truthfulqa"), "{:.2f}"),
        "ZeroDOnp": _rng(a("0D", "triviaqa"), "{:.2f}"),
        "ZeroDHE": _rng(a("0D", "halueval"), "{:.2f}"),
        "NonTopoAll": _rng(pd.concat([a("NONTOPO", "truthfulqa"), a("NONTOPO", "triviaqa")]), "{:.2f}"),
        "HiddenAll": _rng(pd.concat([a("HIDDEN", "truthfulqa"), a("HIDDEN", "triviaqa")]), "{:.2f}"),
        "LookbackAll": _rng(pd.concat([a("LOOKBACK", "truthfulqa"), a("LOOKBACK", "triviaqa")]), "{:.2f}"),
        "HiddenHE": _rng(a("HIDDEN", "halueval"), "{:.2f}"),
        "SplitConed": _rng(ls[ls["subset"] == "coned"]["delta"]) if len(ls) else "--",
        "SplitOpen": _rng(ls[ls["subset"] == "open"]["delta"]) if len(ls) else "--",
    }
    lf = pd.read_csv(f"{RES}/late_fusion.csv") if os.path.exists(f"{RES}/late_fusion.csv") else pd.DataFrame()
    # largest |delta| of 0D / deflated PH beyond NONTOPO (early and late fusion), TruthfulQA + on-policy
    inc = comp[comp["test"].isin(["I_0D_beyond_nontopo", "I_defl_beyond_nontopo",
                                  "LF_0D_beyond_NONTOPO", "LF_DEFL_beyond_NONTOPO"])]
    inc = inc[~inc["setting"].str.startswith("halueval")]
    m["IZeroMaxAbs"] = f"{np.ceil(inc['delta'].abs().max() * 1000) / 1000:.3f}" if len(inc) else "--"
    m["IZeroAllEquiv"] = "yes" if len(inc) and inc["equiv_0.015"].all() else "no"
    mb = f"{RES}/synthetic_morse_bound.csv"
    if os.path.exists(mb):
        mm = pd.read_csv(mb).set_index("N")
        ns = [n for n in (16, 32, 64, 128, 256) if n in mm.index]
        sl = lambda col: np.polyfit(np.log(mm.index.values), np.log(mm[col].values), 1)[0]
        m["MorseNumbersText"] = (
            "Numerically, the integral equals " + ", ".join(f"{mm.loc[n, 'morse_lower_bound']:.3g}" for n in ns)
            + " at $N=" + ",".join(str(n) for n in ns) + "$, against simulated means "
            + ", ".join(f"{mm.loc[n, 'mean_P1']:.3g}" for n in ns)
            + r" under i.i.d.\ Uniform$[0,1]$ weights (Figure~\ref{fig:synthetic}, left). Over $N\in[16,256]$ both grow faster than "
            + f"linearly (log-log slopes {sl('morse_lower_bound'):.2f} and {sl('mean_P1'):.2f}), a pre-asymptotic effect of the "
            + r"$-\sqrt{3N}$ term; both are $\Theta(N)$ as $N\to\infty$.")
    for tag, test in (("LFZeroNT", "LF_0D_beyond_NONTOPO"), ("LFDeflNT", "LF_DEFL_beyond_NONTOPO"),
                      ("LFSinkNT", "LF_SINK_beyond_NONTOPO"), ("LFZeroHid", "LF_0D_beyond_HIDDEN"),
                      ("LFZeroLogp", "LF_0D_beyond_LOGPROB")):
        d = lf[lf["test"] == test] if len(lf) else lf
        m[tag + "Range"] = _rng(d["delta"]) if len(d) else "--"
        m[tag + "Equiv"] = str(int(d["equiv_0.015"].sum())) if len(d) else "--"
        m[tag + "Sig"] = str(int((d["ci95_lo"] > 0).sum())) if len(d) else "--"
        m[tag + "Total"] = str(len(d))
    gens = [v["accuracy"] for v in onpolicy_stats().values()]
    m["OnpAccRange"] = _rng(pd.Series(gens), "{:.0f}\\%")
    if os.path.exists(f"{RES}/check_theory.csv"):
        ct = pd.read_csv(f"{RES}/check_theory.csv").iloc[0]
        m["CheckTohaN"], m["CheckDiagN"] = f"{int(ct['toha_n'])}", f"{int(ct['diag_n'])}"
        e = f"{ct['toha_err']:.0e}".split("e")
        m["CheckTohaErr"] = f"{e[0]}\\cdot10^{{{int(e[1])}}}"
        m["CheckViol"] = str(int(ct["toha_viol"] + ct["diag_viol0"] + ct["diag_violk"]))
    pc = f"{RES}/synthetic_planted_cycle.csv"
    if os.path.exists(pc):
        pc = pd.read_csv(pc)
        m["PlantModerate"] = f"{pc[(pc.b == 6) & (pc.gamma == 8)]['auc_h1'].iloc[0]:.2f}"
        m["PlantStrong"] = f"{pc[(pc.b == 8) & (pc.gamma <= 4)]['auc_h1'].max():.2f}"
        m["PlantNone"] = f"{pc[pc.b == 0]['auc_h1'].max():.2f}"
    m.update(numbers_defect())
    m.update(numbers_toha())
    m.update(numbers_mistral())
    m.update(numbers_native())
    m.update(numbers_audit())
    m.update(numbers_release())
    m.update(table_toha_causal()[1])
    # largest |delta| of any topological bank (0D, deflated, per-head defect, TOHA) beyond
    # NONTOPO, early or late fusion, on TruthfulQA and on-policy TriviaQA
    parts = [inc[["setting", "delta"]]]
    if os.path.exists(f"{RES}/defect_probe.csv"):
        dp = pd.read_csv(f"{RES}/defect_probe.csv")
        parts.append(dp[dp["test"].isin(["LF_PH_DELTA_beyond_NONTOPO", "LF_DELTA_beyond_NONTOPO"])][["setting", "delta"]])
    allinc = pd.concat(parts)
    allinc = allinc[~allinc["setting"].str.startswith("halueval")]
    m["TopoNTMaxAbs"] = f"{np.ceil(allinc['delta'].abs().max() * 1000) / 1000:.3f}"
    write_timing()
    with open(f"{PAPER}/sink_numbers.tex", "w") as fh:
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def numbers_release():
    """What the released artifact actually contains.

    Two constraints decide this, and neither is negotiable. The per-example dumps for most
    settings are gone with the machine that produced them. And the anonymous mirror that
    hosts the release refuses files over MIRROR_LIMIT_MB, which rules out every perhead.npz
    and hidden.npy we still hold, and both Mistral layer dumps. What is left is what these
    macros name, so the reproducibility statement promises exactly what a reviewer can open.
    """
    gens, layers = [], []
    for s in ORDER:
        for name, bucket in (("generations.csv", gens), ("layers.parquet", layers)):
            f = f"{OUT}/{s}/{name}"
            if os.path.exists(f) and os.path.getsize(f) <= MIRROR_LIMIT_MB * 1024 ** 2:
                bucket.append(s)
    return {
        "ReleaseLimitMB": str(MIRROR_LIMIT_MB),
        "ReleaseGenN": str(len(gens)),
        "ReleaseGenList": _join_names([short(x) for x in gens]),
        "ReleaseLayerN": str(len(layers)),
        "ReleaseLayerList": _join_names([short(x) for x in layers]),
    }


def numbers_audit():
    """Macros for the LLM-judge label audit (sinktda/label_audit.py). Empty if not run."""
    f = f"{RES}/label_audit.csv"
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f)
    d = d[d["judge"] >= 0]
    if not len(d):
        return {}
    rate = d.groupby("setting").apply(
        lambda x: (x["judge"] != x["string_match"]).mean(), include_groups=False)
    m = {
        "AuditSettings": str(d["setting"].nunique()),
        "AuditN": str(int(d.groupby("setting").size().min())),
        "AuditDisagreeRange": _rng(rate * 100, "{:.1f}\\%"),
        "AuditDisagreeMax": f"{rate.max() * 100:.1f}\\%",
    }
    s = f"{RES}/label_audit_sensitivity.csv"
    if os.path.exists(s):
        sn = pd.read_csv(s)
        m["AuditMaxAucShift"] = f"{sn['delta'].abs().max():.3f}"
    return m


def numbers_native():
    """Macros for the check of our TOHA score against the authors' released MTop-Div code
    (sinktda/toha_native.py). Empty when that check has not been run."""
    f = f"{RES}/toha_native.csv"
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f)
    mx = float(d["abs_diff"].max())
    if mx <= 0:
        s = "0"
    else:
        e = int(np.floor(np.log10(mx)))
        s = f"{mx / 10 ** e:.1f}\\times10^{{{e}}}"
    return {
        "NativeCells": f"{len(d):,}",
        "NativeSettings": str(d["setting"].nunique()),
        "NativeMaxDiff": f"${s}$",
    }


def numbers_mistral():
    """Macros for the Mistral-7B appendix, from the generated theory/TOHA CSVs.
    Returns {} when the setting has not been run, so the appendix degrades gracefully."""
    f = f"{RES}/theory_truthfulqa_mistral.csv"
    if not os.path.exists(f):
        return {}
    t = pd.read_csv(f).iloc[0]
    viol = int(t["bound_violations_h1"] + t["bound_violations_p0"] + t["bound_violations_maxdeath"])
    m = {
        "MistralCells": f"{int(t['cells']):,}",
        "MistralViol": str(viol),
        "MistralSinkMass": f"{t['mean_sink_mass']:.2f}",
        "MistralConed": f"{t['frac_cells_coned'] * 100:.1f}\\%",
        "MistralHOneZero": f"{t['frac_cells_h1_zero'] * 100:.2f}\\%",
        "MistralRhoNorm": f"{t['median_layer_spearman_P0_star_norm']:.3f}",
        "MistralRhoNormMin": f"{t['min_layer_spearman_P0_star_norm']:.3f}",
    }
    ck = f"{RES}/toha_checks.csv"
    if os.path.exists(ck):
        c = pd.read_csv(ck)
        c = c[c["setting"] == "truthfulqa_mistral"]
        if len(c):
            r = c.iloc[0]
            m["MistralTohaConed"] = f"{r['frac_coned_P'] * 100:.0f}\\%"
            m["MistralTohaRho"] = f"{r['median_head_rho_maxp']:.3f}"
            m["MistralTohaViol"] = str(int(r["viol_upper"] + r["viol_lower"]))
    return m


def numbers_defect():
    """Coning defect as a detector (sinktda/defect_probe.py)."""
    p = f"{RES}/defect_probe.csv"
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    aucs = d[d["test"].str.startswith("AUC_")]
    m = {"DefLayerAuc": _rng(aucs[aucs["B"] == "DELTA"]["auc_B"], "{:.2f}"),
         "DefHeadAuc": _rng(aucs[aucs["B"] == "PH_DELTA"]["auc_B"], "{:.2f}"),
         "DefTotal": str(aucs["setting"].nunique())}
    for tag, test in (("DefHSink", "LF_PH_DELTA_beyond_PH_SINK"), ("DefHNT", "LF_PH_DELTA_beyond_NONTOPO"),
                      ("DefLSink", "LF_DELTA_beyond_SINK"), ("DefLNT", "LF_DELTA_beyond_NONTOPO")):
        t = d[d["test"] == test]
        m[tag + "Range"] = _rng(t["delta"])
        m[tag + "Sig"] = str(int((t["ci95_lo"] > 0).sum()))
        m[tag + "Equiv"] = str(int(t["equiv_0.015"].sum()))
    return m


TOHA_BANKS = [("TOHA_toha", "TOHA"), ("TOHA_maxp", r"TOHA$[\bar\pi]$"), ("TOHA_sinkr", r"TOHA$[\bar a_0]$"),
              ("SUP_toha", "TOHA$_{\\text{sup}}$"), ("SUP_maxp", r"$\bar\pi_{\text{sup}}$"),
              ("SUP_sinkr", r"$\bar a_{0,\text{sup}}$"), ("NONTOPO", "Non-topo.")]


SINK_MASS_MIN = 0.2  # a setting counts as sink-dominated above this mean attention to token 0


def _sink_settings():
    """Settings whose model actually puts attention on token 0, measured from the theory
    tables rather than hardcoded by model name.

    SmolLM-1.7B used to be the only model without a first-token sink, so excluding it by
    name was equivalent to thresholding. Qwen2.5-7B-Instruct is a second one (its sink sits
    on token 2), so the name-based test would silently count it as sink-dominated and
    corrupt every "in the sink models" range. Returns None when no theory table is present.
    """
    th = load("theory")
    if not len(th) or "mean_sink_mass" not in th.columns:
        return None
    return set(th[th["mean_sink_mass"] > SINK_MASS_MIN]["setting"])


def _toha_frames():
    fs = [f"{RES}/toha_{k}.csv" for k in ("checks", "auc", "comp")]
    if not all(os.path.exists(f) for f in fs):
        return None
    ck, auc, comp = (pd.read_csv(f) for f in fs)
    o = {s: i for i, s in enumerate(ORDER)}
    ck = ck.assign(_o=ck["setting"].map(o).fillna(99)).sort_values("_o").drop(columns="_o")
    return ck, auc, comp


def table_toha():
    fr = _toha_frames()
    if fr is None:
        return r"\textit{TOHA results pending.}"
    ck, auc, comp = fr
    cols_ = [r"$\defect_P{=}0$", r"$\rho(d,\bar\pi)$"] + [n for _, n in TOHA_BANKS] + \
            [r"$\bar\pi-$TOHA", r"$\bar a_0-$TOHA", r"TOHA$\oplus\bar\pi$", r"TOHA$\oplus$NT"]
    lines = [r"\begin{tabular}{l" + "c" * len(cols_) + "}", r"\toprule",
             "Setting & " + " & ".join(cols_) + r" \\", r"\midrule"]
    prev = None
    for _, r in ck.iterrows():
        s = r["setting"]
        b = PRETTY.get(s, (s, ""))[0]
        if prev is not None and b != prev:
            lines.append(r"\midrule")
        prev = b
        a = auc[auc["setting"] == s].set_index("bank")["auc"]
        c = comp[comp["setting"] == s].set_index("test")
        cells = [f"{100 * r['frac_coned_P']:.0f}\\%", f"{r['median_head_rho_maxp']:.3f}"[1:]]
        cells += ["--" if k not in a else ("1.00" if a[k] >= 0.9995 else f"{a[k]:.3f}"[1:]) for k, _ in TOHA_BANKS]
        for t in ("TOHA_maxp_vs_toha", "TOHA_sinkr_vs_toha", "LF_toha_beyond_maxp", "LF_toha_beyond_nontopo"):
            if t not in c.index:
                cells.append("--")
                continue
            x = c.loc[t]
            mark = r"$^{\equiv}$" if x["equiv_0.015"] else ("$^{*}$" if (x["ci95_lo"] > 0 or x["ci95_hi"] < 0) else "")
            cells.append(f"${x['delta']:+.3f}${mark}".replace("-0.000", "0.000").replace("+0.000", "0.000"))
        lines.append(f"{short(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_toha_causal():
    """Sink-bias intervention (sinktda/run_toha_causal.sh)."""
    fs = [f"{RES}/toha_{k}_causal.csv" for k in ("checks", "auc")]
    if not all(os.path.exists(f) for f in fs):
        return r"\textit{Causal results pending.}", {}
    ck, auc = (pd.read_csv(f) for f in fs)
    bias = {"": 0, "_sbm2": -2, "_sbp2": 2, "_sbp4": 4}
    rows, m = [], {}
    for _, r in ck.iterrows():
        base = r["setting"].split("_sb")[0]
        b = bias[r["setting"][len(base):]]
        a = auc[auc["setting"] == r["setting"]].set_index("bank")["auc"]
        rows.append(dict(model=PRETTY[base][1], b=b, coned=r["frac_coned_P"], arg0=r["frac_arg0_all"],
                         rho=r["median_head_rho_sinkr"], toha=a["TOHA_toha"], tsink=a["TOHA_sinkr"],
                         sup=a["SUP_toha"], supsink=a["SUP_sinkr"]))
    d = pd.DataFrame(rows).sort_values(["model", "b"])
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Model & $b$ & $\defect_P{=}0$ & sink top & $\rho(d,1-\bar a_0)$ & TOHA & TOHA$[\bar a_0]$ & TOHA$_{\text{sup}}$ & $\bar a_{0,\text{sup}}$ \\",
             r"\midrule"]
    for _, r in d.iterrows():
        lines.append(f"{r['model']} & ${r['b']:+d}$ & ".replace("+0$", "0$") + f"{100 * r['coned']:.0f}\\% & {100 * r['arg0']:.0f}\\% & "
                     f"{r['rho']:.3f} & {r['toha']:.3f} & {r['tsink']:.3f} & {r['sup']:.3f} & {r['supsink']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    for mod, g in d.groupby("model"):
        k = "Qw" if mod.startswith("Qwen") else "Tl"
        lo, hi = g[g.b == g.b.min()].iloc[0], g[g.b == g.b.max()].iloc[0]
        m[f"Causal{k}ArgLo"], m[f"Causal{k}ArgHi"] = f"{100 * lo['arg0']:.0f}\\%", f"{100 * hi['arg0']:.0f}\\%"
        m[f"Causal{k}RhoLo"], m[f"Causal{k}RhoHi"] = f"{lo['rho']:.3f}", f"{hi['rho']:.3f}"
        m[f"Causal{k}ConedLo"], m[f"Causal{k}ConedHi"] = f"{100 * lo['coned']:.0f}\\%", f"{100 * hi['coned']:.0f}\\%"
        z = g[g.b == 0].iloc[0]
        m[f"Causal{k}ConedZero"], m[f"Causal{k}ArgZero"] = f"{100 * z['coned']:.0f}\\%", f"{100 * z['arg0']:.0f}\\%"
        m[f"Causal{k}RhoZero"] = f"{z['rho']:.3f}"
        m[f"Causal{k}TohaRange"] = _rng(g["toha"], "{:.2f}")
        m[f"Causal{k}GapLo"] = f"{lo['toha'] - lo['tsink']:+.3f}"
        m[f"Causal{k}GapHi"] = f"{hi['toha'] - hi['tsink']:+.3f}"
    return "\n".join(lines), m


def numbers_toha():
    fr = _toha_frames()
    if fr is None:
        return {}
    ck, auc, comp = fr
    ss = _sink_settings()
    sink = ck[ck["setting"].isin(ss)] if ss is not None else ck[ck["setting"] != "truthfulqa_smollm"]
    ab = lambda k, sub=None: auc[(auc["bank"] == k) & (auc["setting"].str.startswith(sub) if sub else True)]["auc"]
    m = {
        "TohaSettings": str(len(ck)),
        "TohaCells": f"{int(ck['cells'].sum()):,}",
        "TohaViol": str(int((ck["viol_upper"] + ck["viol_lower"]).sum())),
        "TohaConedRange": f"{100 * ck['frac_coned_P'].min():.0f}--{100 * ck['frac_coned_P'].max():.0f}\\%",
        "TohaGapMax": (lambda e: f"${e[0]}\\cdot10^{{{int(e[1])}}}$")(f"{ck['max_abs_gap_coned'].max():.0e}".split("e")),
        "TohaArgZeroRange": f"{100 * sink['frac_arg0_all'].min():.0f}--{100 * sink['frac_arg0_all'].max():.0f}\\%",
        "TohaRhoRange": _rng(ck["median_head_rho_maxp"], "{:.3f}"),
        "TohaRhoSinkRange": _rng(sink["median_head_rho_sinkr"], "{:.2f}"),
        # the same correlation in the models whose sink is not the first token
        "TohaRhoSinkNoSink": _rng(ck[~ck["setting"].isin(set(sink["setting"]))]["median_head_rho_sinkr"], "{:.2f}"),
        "TohaFracHeadsRho": _rng(ck["frac_heads_rho_maxp_ge_0_9"] * 100, "{:.0f}\\%"),
        "TohaSelConedRange": f"{100 * ck['toha_sel_frac_coned_P'].min():.0f}--{100 * ck['toha_sel_frac_coned_P'].max():.0f}\\%",
        "TohaSelArgZeroRange": f"{100 * sink['toha_sel_frac_arg0'].min():.0f}--{100 * sink['toha_sel_frac_arg0'].max():.0f}\\%",
        "TohaSelOverlap": _rng(ck["toha_sel_overlap_sinkr"] * 100, "{:.0f}\\%"),
        "TohaAuc": _rng(ab("TOHA_toha"), "{:.2f}"),
        "TohaAucTQA": _rng(ab("TOHA_toha", "truthfulqa"), "{:.2f}"),
        "TohaAucHE": _rng(ab("TOHA_toha", "halueval"), "{:.2f}"),
        "TohaAucOnp": _rng(ab("TOHA_toha", "triviaqa"), "{:.2f}"),
        "TohaMaxpAuc": _rng(ab("TOHA_maxp"), "{:.2f}"),
        "TohaSinkAuc": _rng(ab("TOHA_sinkr"), "{:.2f}"),
        "TohaSupAuc": _rng(ab("SUP_toha"), "{:.2f}"),
        "TohaSupSinkAuc": _rng(ab("SUP_sinkr"), "{:.2f}"),
    }
    tl = comp[comp["test"] == "LF_toha_beyond_maxp"]
    m["TohaLFMaxpMax"] = f"{np.ceil(tl['delta'].abs().max() * 1000) / 1000:.3f}"
    tn = comp[comp["test"] == "LF_toha_beyond_nontopo"]
    m["TohaLFNTMax"] = f"{np.ceil(tn['delta'].abs().max() * 1000) / 1000:.3f}"
    ts = comp[comp["test"] == "TOHA_sinkr_vs_toha"]
    # split by measured sink mass, not by model name: SmolLM is no longer the only model
    # without a first-token sink (Qwen2.5-7B's sink sits on token 2)
    in_sink = ts["setting"].isin(ss) if ss is not None else ts["setting"] != "truthfulqa_smollm"
    m["TohaSinkLossSink"] = f"{-ts[in_sink]['delta'].min():.3f}"
    m["TohaSinkLossNoSink"] = _rng(-ts[~in_sink]["delta"], "{:.3f}") if (~in_sink).any() else "--"
    m["TohaNoSinkModels"] = _join_names(
        {PRETTY.get(s, (s, s))[1] for s in ts.loc[~in_sink, "setting"]})
    for tag, test in (("TohaMaxp", "TOHA_maxp_vs_toha"), ("TohaSink", "TOHA_sinkr_vs_toha"),
                      ("TohaSupMaxp", "SUP_maxp_vs_toha"), ("TohaSupSink", "SUP_sinkr_vs_toha"),
                      ("TohaLFMaxp", "LF_toha_beyond_maxp"), ("TohaLFNT", "LF_toha_beyond_nontopo")):
        t = comp[comp["test"] == test]
        m[tag + "Range"] = _rng(t["delta"])
        m[tag + "Equiv"] = str(int(t["equiv_0.015"].sum()))
        m[tag + "Pos"] = str(int((t["ci95_lo"] > 0).sum()))
        m[tag + "Neg"] = str(int((t["ci95_hi"] < 0).sum()))
        m[tag + "NonInf"] = str(int((t["ci90_lo"] > -0.015).sum()))
        m[tag + "Total"] = str(len(t))
    return m


def table_native():
    """Our TOHA score against the authors' released MTop-Div routine, per setting."""
    f = f"{RES}/toha_native.csv"
    if not os.path.exists(f):
        return ""
    d = pd.read_csv(f)
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Setting & head graphs & max $|\Delta|$ & mean $|\Delta|$ & Pearson $r$ \\", r"\midrule"]
    for s, g in d.groupby("setting", sort=False):
        mx, mn = g["abs_diff"].max(), g["abs_diff"].mean()
        r = g["ours"].corr(g["native"])
        e = lambda v: "$0$" if v <= 0 else f"${v / 10 ** int(np.floor(np.log10(v))):.1f}" \
                                           f"\\times10^{{{int(np.floor(np.log10(v)))}}}$"
        rows.append(f"{short(s)} & {len(g):,} & {e(mx)} & {e(mn)} & {r:.8f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def table_audit():
    """LLM-judge audit of the on-policy labels, per setting."""
    f = f"{RES}/label_audit.csv"
    if not os.path.exists(f):
        return ""
    d = pd.read_csv(f)
    d = d[d["judge"] >= 0]
    sens = pd.read_csv(f"{RES}/label_audit_sensitivity.csv") if os.path.exists(
        f"{RES}/label_audit_sensitivity.csv") else pd.DataFrame()
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Setting & judged & disagree & match$\,$0/judge$\,$1 & match$\,$1/judge$\,$0 "
            r"& max $|\Delta$AUC$|$ \\", r"\midrule"]
    for s, g in d.groupby("setting", sort=False):
        dis = (g["judge"] != g["string_match"]).mean() * 100
        fn = int(((g["string_match"] == 0) & (g["judge"] == 1)).sum())
        fp = int(((g["string_match"] == 1) & (g["judge"] == 0)).sum())
        sh = sens[sens["setting"] == s]["delta"].abs().max() if len(sens) else np.nan
        shs = "--" if pd.isna(sh) else f"{sh:.3f}"
        rows.append(f"{short(s)} & {len(g):,} & {dis:.1f}\\% & {fn} & {fp} & {shs} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def main():
    style()
    th = load("theory")
    auc = load("auc")
    comp = load("comp")
    if os.path.exists(f"{RES}/late_fusion.csv"):
        lf = pd.read_csv(f"{RES}/late_fusion.csv")
        comp = pd.concat([comp, lf], ignore_index=True)
        comp["_o"] = comp["setting"].map({s: i for i, s in enumerate(ORDER)}).fillna(99)
        comp = comp.sort_values("_o", kind="stable").drop(columns="_o")
    lay = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/theory_layers_*.csv")], ignore_index=True) \
        if glob.glob(f"{RES}/theory_layers_*.csv") else pd.DataFrame()
    with open(f"{PAPER}/sink_tables.tex", "w") as fh:
        for name, body in [("theory", table_theory(th)), ("auc", table_auc(auc)), ("tests", table_tests(comp))]:
            fh.write(f"\\newcommand{{\\SinkTable{name.capitalize()}}}{{%\n{body}\n}}\n")
    with open(f"{PAPER}/sink_tables_tests_full.tex", "w") as fh:
        fh.write(table_tests_full(comp) + "\n")
    with open(f"{PAPER}/sink_tables_auc_full.tex", "w") as fh:
        fh.write(table_auc_full(auc) + "\n")
    with open(f"{PAPER}/sink_tables.tex", "a") as fh:
        fh.write(f"\\newcommand{{\\SinkTableOnpolicy}}{{%\n{table_onpolicy()}\n}}\n")
        fh.write(f"\\newcommand{{\\SinkTableToha}}{{%\n{table_toha()}\n}}\n")
        fh.write(f"\\newcommand{{\\SinkTableTohaCausal}}{{%\n{table_toha_causal()[0]}\n}}\n")
        for name, body in [("Native", table_native()), ("Audit", table_audit())]:
            if body:
                fh.write(f"\\newcommand{{\\SinkTable{name}}}{{%\n{body}\n}}\n")
    write_numbers(th, auc, comp)
    fig_layers(lay)
    fig_synthetic()
    fig_forest(comp)
    th.to_csv(f"{RES}/summary_theory.csv", index=False)
    auc.to_csv(f"{RES}/summary_auc.csv", index=False)
    comp.to_csv(f"{RES}/summary_comp.csv", index=False)
    print(th.to_string(index=False))


if __name__ == "__main__":
    main()
