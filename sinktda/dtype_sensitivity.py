"""
dtype_sensitivity.py -- how much do features depend on inference precision?

Compares pooled AUCs of the earlier float16 extractions (archive/fp16_backup/) with the
bfloat16 extractions used in the paper, for the settings that were run both ways.

  python -m sinktda.dtype_sensitivity
Writes sinktda_results/dtype_sensitivity.csv and paper/sink_dtype.tex
"""
import os

import pandas as pd

from sinktda.report import ORDER, short

OLD = "archive/fp16_backup/sinktda_results"
NEW = "sinktda_results"
BANKS = [("SINK", "Sink"), ("0D", "0D"), ("1D", "1D"), ("DEFL", "Defl."), ("PH_0D", "0D$_h$"),
         ("LOGPROB", "LogP"), ("LOOKBACK", "Lookb."), ("HIDDEN", "Hidden"), ("NONTOPO", "Non-topo.")]
FP16 = ["truthfulqa_qwen3b", "truthfulqa_phi3", "truthfulqa_tinyllama", "truthfulqa_smollm", "halueval_qwen3b"]


PREC = "archive/fp32_cuda/sinktda_results"
PREC_SETTINGS = ["truthfulqa_tinyllama", "truthfulqa_qwen1.5b"]


def precision():
    """Third precision column: float32 against bfloat16, both re-extracted on the same
    CUDA box, plus the bf16 CUDA-vs-MPS gap as a platform control.

    The paper's bf16 numbers were extracted on Apple-silicon MPS, so comparing a CUDA
    float32 run directly against them would confound precision with platform and library
    version. `<setting>_cuda` is bf16 on CUDA and `<setting>_fp32` is float32 on the same
    box, so `fp32 - cuda` isolates precision and `cuda - paper` measures the platform gap.
    Returns an empty frame when those runs are absent.
    """
    rows = []
    for s in PREC_SETTINGS:
        f_mps = f"{NEW}/auc_{s}.csv"
        f_cuda, f_fp32 = f"{PREC}/auc_{s}_cuda.csv", f"{PREC}/auc_{s}_fp32.csv"
        if not all(os.path.exists(x) for x in (f_mps, f_cuda, f_fp32)):
            continue
        a = pd.read_csv(f_mps).set_index("bank")["auc"]
        b = pd.read_csv(f_cuda).set_index("bank")["auc"]
        c = pd.read_csv(f_fp32).set_index("bank")["auc"]
        for k, _ in BANKS:
            if k in a and k in b and k in c:
                rows.append(dict(setting=s, bank=k, auc_bf16_mps=a[k], auc_bf16_cuda=b[k],
                                 auc_fp32_cuda=c[k], platform=b[k] - a[k], precision=c[k] - b[k]))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df.to_csv(f"{NEW}/precision_sensitivity.csv", index=False)
    lines = [r"\begin{tabular}{ll" + "r" * len(BANKS) + "}", r"\toprule",
             "Setting & Contrast & " + " & ".join(n for _, n in BANKS) + r" \\", r"\midrule"]
    for s in [x for x in ORDER if x in set(df["setting"])]:
        d = df[df["setting"] == s].set_index("bank")
        for col, lab in (("platform", r"bf16 CUDA $-$ bf16 MPS"), ("precision", r"fp32 $-$ bf16 (CUDA)")):
            cells = ["--" if k not in d.index else
                     f"${d.loc[k, col]:+.3f}$".replace("-0.000", "0.000").replace("+0.000", "0.000")
                     for k, _ in BANKS]
            lines.append(f"{short(s)} & {lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open("paper/sink_precision.tex", "w") as fh:
        fh.write(r"\newcommand{\PrecPlatformMax}{" + f"{df['platform'].abs().max():.3f}" + "}\n")
        fh.write(r"\newcommand{\PrecFp32Max}{" + f"{df['precision'].abs().max():.3f}" + "}\n")
        fh.write(r"\newcommand{\PrecFp32TopoMax}{"
                 + f"{df[df.bank.isin(['0D', '1D', 'DEFL', 'SINK'])]['precision'].abs().max():.3f}" + "}\n")
        fh.write(r"\newcommand{\PrecSettings}{" + str(df['setting'].nunique()) + "}\n")
        fh.write(r"\newcommand{\SinkTablePrecision}{%" + "\n" + "\n".join(lines) + "\n}\n")
    print("\n=== precision / platform (AUC differences) ===")
    print(df.pivot(index="setting", columns="bank", values="precision").round(3).to_string())
    return df


def main():
    rows = []
    for s in ORDER:
        if s not in FP16:
            continue
        fo, fn = f"{OLD}/auc_{s}.csv", f"{NEW}/auc_{s}.csv"
        if not (os.path.exists(fo) and os.path.exists(fn)):
            continue
        a = pd.read_csv(fo).set_index("bank")["auc"]
        b = pd.read_csv(fn).set_index("bank")["auc"]
        for k, _ in BANKS:
            if k in a and k in b:
                rows.append(dict(setting=s, bank=k, auc_fp16=a[k], auc_bf16=b[k], diff=b[k] - a[k]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{NEW}/dtype_sensitivity.csv", index=False)
    piv = df.pivot(index="setting", columns="bank", values="diff")
    lines = [r"\begin{tabular}{l" + "r" * len(BANKS) + "}", r"\toprule",
             "Setting & " + " & ".join(n for _, n in BANKS) + r" \\", r"\midrule"]
    for s in [x for x in ORDER if x in piv.index]:
        cells = ["--" if pd.isna(piv.loc[s].get(k)) else f"${piv.loc[s][k]:+.3f}$".replace("-0.000", "0.000").replace("+0.000", "0.000") for k, _ in BANKS]
        lines.append(f"{short(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    topo = df[df.bank.isin(["0D", "1D", "DEFL"])]["diff"].abs()
    nt = df[df.bank.isin(["LOGPROB", "LOOKBACK", "HIDDEN", "NONTOPO"])]["diff"].abs()
    with open("paper/sink_dtype.tex", "w") as fh:
        fh.write(r"\newcommand{\DtypeTopoMax}{" + f"{topo.max():.3f}" + "}\n")
        worst = df.loc[df[df.bank.isin(["0D", "1D", "DEFL"])]["diff"].abs().idxmax(), "setting"]
        other = df[(df.setting != worst) & df.bank.isin(["0D", "1D", "DEFL"])]["diff"].abs().max()
        fh.write(r"\newcommand{\DtypeTopoWorst}{" + short(worst) + "}\n")
        fh.write(r"\newcommand{\DtypeTopoOther}{" + f"{other:.3f}" + "}\n")
        fh.write(r"\newcommand{\DtypeNTMax}{" + f"{nt.max():.3f}" + "}\n")
        fh.write(r"\newcommand{\DtypeSinkMax}{" + f"{df[df.bank == 'SINK']['diff'].abs().max():.3f}" + "}\n")
        fh.write(r"\newcommand{\SinkTableDtype}{%" + "\n" + "\n".join(lines) + "\n}\n")
    print(piv.round(3).to_string())
    precision()


if __name__ == "__main__":
    main()
