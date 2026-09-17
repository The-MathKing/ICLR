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

# reference categorical order (dataviz palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
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
    "halueval_qwen3b": ("HaluEval", "Qwen2.5-3B"),
    "halueval_qwen1.5b": ("HaluEval", "Qwen2.5-1.5B"),
    "triviaqa_qwen3b": ("TriviaQA (on-policy)", "Qwen2.5-3B"),
    "triviaqa_qwen1.5b": ("TriviaQA (on-policy)", "Qwen2.5-1.5B"),
    "triviaqa_phi3": ("TriviaQA (on-policy)", "Phi-3-mini"),
    "triviaqa_tinyllama": ("TriviaQA (on-policy)", "TinyLlama-1.1B"),
}
ORDER = list(PRETTY)


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
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Benchmark & Model & $\bar N$ & graphs & viol. & sink attn. & coned & $H_1{=}\emptyset\,|\,$coned & $\rho(P_0,\stw_0)$ \\",
             r"\midrule"]
    for _, r in th.iterrows():
        b, m = PRETTY.get(r["setting"], (r["setting"], ""))
        hz = "--" if pd.isna(r["frac_h1zero_given_coned"]) else f"{100 * r['frac_h1zero_given_coned']:.1f}\\%"
        lines.append(f"{b} & {m} & {r['mean_N']:.1f} & {int(r['cells']):,} & "
                     f"{int(r['bound_violations_h1'] + r['bound_violations_p0'])} & "
                     f"{r['mean_sink_mass']:.2f} & {100 * r['frac_cells_coned']:.1f}\\% & {hz} & "
                     f"{r['median_layer_spearman_P0_star']:.3f} \\\\")
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
        cells = [("\\textbf{" + f3(v)[1:] + "}") if v == best else f3(v)[1:] for v in vals]
        lines.append(f"{short(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


KEY_TESTS = [("T1_reduction", r"Sink$-$0D"), ("T2_topo_beyond_sink", r"0D$\mid$Sink"),
             ("PH_topo_beyond_sink", r"0D$_h\mid$Sink$_h$"),
             ("T3_deflated_beyond_sink", r"Defl.$\mid$Sink"),
             ("T4_1D_beyond_0D", r"1D$\mid$0D"),
             ("I_0D_beyond_nontopo", r"0D$\mid$NT"),
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
    return f"{SHORT_B.get(b, b)} {m.replace('Qwen2.5-', 'Qw').replace('-mini', '').replace('TinyLlama-1.1B', 'TinyLl.').replace('SmolLM-1.7B', 'SmolLM')}"


def table_tests_full(comp):
    lines = [r"\begin{longtable}{llrrrrcc}", r"\toprule",
             r"Setting & Test (A $\to$ B) & AUC$_A$ & AUC$_B$ & $\Delta$ & 90\% CI & $\equiv_{.015}$ & power \\",
             r"\midrule\endhead"]
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
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    fig.subplots_adjust(wspace=0.3)
    for i, s in enumerate(tq[:8]):
        g = lay[lay["setting"] == s]
        c = SERIES[i % len(SERIES)]
        axes[0].plot(g["depth"], g["frac_coned"], color=c, label=label(s))
        axes[1].plot(g["depth"], g["mean_sink_mass"], color=c)
        axes[2].plot(g["depth"], g["spearman_P0_star"], color=c)
    axes[0].set_title("fraction exactly coned ($\\delta_0{=}0$)", fontsize=8)
    axes[1].set_title("mean attention to token 0", fontsize=8)
    axes[2].set_title("Spearman($P_0$, star weight)", fontsize=8)
    for a in axes:
        a.set_xlabel("relative layer depth")
    axes[2].set_ylim(0.0, 1.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.22), fontsize=6.5)
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
    a.plot(m["N"], m["morse_lower_bound"], color=SERIES[1], marker="s", ms=3, label="Prop. 2 lower bound")
    a.set_xscale("log"); a.set_yscale("log")
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    a.set_xlabel("$N$ (i.i.d. weights)"); a.set_title("total 1D persistence", fontsize=8)
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


def table_onpolicy():
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & questions & accuracy & rows used & mean answer tokens \\", r"\midrule"]
    for s in ORDER:
        if not s.startswith("triviaqa"):
            continue
        g = f"sinktda_out/{s}/generations.csv"
        lp = f"sinktda_out/{s}/layers.parquet"
        if not (os.path.exists(g) and os.path.exists(lp)):
            continue
        gen = pd.read_csv(g, keep_default_na=False)
        lay = pd.read_parquet(lp, columns=["answer_len"])
        lines.append(f"{PRETTY[s][1]} & {len(gen):,} & {100 * gen['correct'].mean():.1f}\\% & {len(lay):,} & "
                     f"{lay['answer_len'].mean():.1f} \\\\")
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


def write_numbers(th, auc, comp):
    models = sorted({PRETTY.get(s, (s, s))[1].replace(" (chat)", "") for s in th["setting"]})
    sink = th[th["mean_sink_mass"] > 0.2]
    nosink = th[th["mean_sink_mass"] <= 0.2]
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
        "NCellsTotal": f"{int(th['cells'].sum()):,}",
        "NSinkSettings": str(len(sink)),
        "PctConedRange": f"{pc.min():.0f}--{pc.max():.0f}\\%" if len(pc) else "--",
        "SinkMassRange": _rng(sink["mean_sink_mass"], "{:.2f}"),
        "NoSinkMass": _rng(nosink["mean_sink_mass"], "{:.2f}"),
        "RhoMin": f"{np.floor(sink['median_layer_spearman_P0_star'].min() * 1000) / 1000:.3f}" if len(sink) else "--",
        "RhoNoSink": _rng(nosink["median_layer_spearman_P0_star"], "{:.3f}"),
        "RhoDeltaHone": _rng(sink["cell_spearman_delta0_h1"], "{:.2f}"),
        "NViolations": str(int((th["bound_violations_h1"] + th["bound_violations_p0"]).sum())),
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
        "PHTopoRange": _rng(T("PH_topo_beyond_sink")["delta"]),
        "NPHTopoEquiv": str(int(T("PH_topo_beyond_sink")["equiv_0.015"].sum())),
        "NPHTopoTotal": str(len(T("PH_topo_beyond_sink"))),
        "PHEntRange": _rng(comp[comp["test"] == "PH_0D_vs_entropy"]["delta"]),
        "NIZeroEquiv": str(int(comp[comp["test"] == "I_0D_beyond_nontopo"]["equiv_0.015"].sum())),
        "NIZeroTotal": str(int((comp["test"] == "I_0D_beyond_nontopo").sum())),
        "IZeroRange": _rng(comp[comp["test"] == "I_0D_beyond_nontopo"]["delta"]),
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
    for s_ in ORDER:
        g = f"sinktda_out/{s_}/generations.csv"
        if s_.startswith("triviaqa") and os.path.exists(g):
            pass
    gens = [pd.read_csv(f"sinktda_out/{s_}/generations.csv", keep_default_na=False)["correct"].mean()
            for s_ in ORDER if s_.startswith("triviaqa") and os.path.exists(f"sinktda_out/{s_}/generations.csv")]
    m["OnpAccRange"] = _rng(pd.Series(gens) * 100, "{:.0f}\\%")
    write_timing()
    with open(f"{PAPER}/sink_numbers.tex", "w") as fh:
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    style()
    th = load("theory")
    auc = load("auc")
    comp = load("comp")
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
