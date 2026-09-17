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


if __name__ == "__main__":
    main()
