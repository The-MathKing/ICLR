"""check_claims.py -- re-check the paper's absolute and range claims against the regenerated CSVs.

Run after report.py. Exits non-zero if any claim no longer matches its CSV.
"""
import io
import re
import sys

import pandas as pd

RES = "sinktda_results"
ok, bad = [], []


def chk(name, cond, detail):
    (ok if cond else bad).append(("PASS  " if cond else "FAIL  ") + name + ": " + detail)


def load_macros(path="paper/sink_numbers.tex"):
    text = io.open(path, encoding="utf-8").read()
    pat = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", re.M)
    return {m.group(1): m.group(2) for m in pat.finditer(text)}


macros = load_macros()
th = pd.read_csv(RES + "/summary_theory.csv")
ck = pd.read_csv(RES + "/toha_checks.csv")

# --- Theorem 1: zero violations on every layer graph ------------------------
vcols = [c for c in th.columns if c.startswith("bound_violations")]
nviol = int(th[vcols].to_numpy().sum())
cells = int(th["cells"].sum())
chk("Theorem 1 has no bound violations", nviol == 0,
    "{} violations over {} settings / {:,} layer graphs".format(nviol, len(th), cells))
chk("NCellsTotal macro matches summary_theory",
    macros.get("NCellsTotal") == "{:,}".format(cells),
    "macro={} csv={:,}".format(macros.get("NCellsTotal"), cells))

# --- Proposition 1 (TOHA): zero violations ----------------------------------
vc = [c for c in ck.columns if "viol" in c]
tviol = int(ck[vc].to_numpy().sum())
chk("Proposition 1 has no violations", tviol == 0,
    "{} violations over {} settings".format(tviol, len(ck)))
chk("TohaSettings macro matches toha_checks",
    macros.get("TohaSettings") == str(len(ck)),
    "macro={} csv={}".format(macros.get("TohaSettings"), len(ck)))
if "cells" in ck.columns:
    tc = int(ck["cells"].sum())
    chk("TohaCells macro matches toha_checks",
        macros.get("TohaCells") == "{:,}".format(tc),
        "macro={} csv={:,}".format(macros.get("TohaCells"), tc))

# --- sink models: H1 empty on every coned graph; coned share; correlation ---
SINK_MIN = 0.1
sink = th[th["mean_sink_mass"] > SINK_MIN]
chk("H1 empty on every coned graph in every sink model",
    bool((sink["frac_h1zero_given_coned"] == 1.0).all()),
    "min={} over {} sink settings".format(sink["frac_h1zero_given_coned"].min(), len(sink)))
pc = 100 * sink["frac_cells_coned"]
want = "{:.0f}--{:.0f}".format(pc.min(), pc.max())
chk("PctConedRange macro matches the sink settings",
    macros.get("PctConedRange", "").replace(r"\%", "") == want,
    "macro={} csv={}".format(macros.get("PctConedRange"), want))
med = sink["median_layer_spearman_P0_star_norm"]
try:
    rmin = float(macros.get("RhoMin", "nan"))
except ValueError:
    rmin = float("nan")
chk("RhoMin is a true lower bound on the median per-layer rank correlation",
    rmin <= med.min() + 1e-9,
    "macro RhoMin={} smallest median={:.4f}".format(macros.get("RhoMin"), med.min()))

# --- label audit ------------------------------------------------------------
s = pd.read_csv(RES + "/label_audit_sensitivity.csv")
try:
    shift = float(macros.get("AuditMaxAucShift", "nan"))
except ValueError:
    shift = float("nan")
chk("AuditMaxAucShift macro matches the sensitivity table",
    abs(shift - s["delta"].abs().max()) < 5e-4,
    "macro={} csv={:.3f}".format(macros.get("AuditMaxAucShift"), s["delta"].abs().max()))

CLAIMS = [("NONTOPO", "0D"), ("NONTOPO", "DEFL"), ("HIDDEN", "0D"), ("NONTOPO", "PH_0D")]
flipped = []
for nm, d in s.groupby("setting"):
    p = d.set_index("bank")
    for hi, lo in CLAIMS:
        if hi in p.index and lo in p.index:
            a = p.loc[hi, "auc_string"] >= p.loc[lo, "auc_string"]
            b = p.loc[hi, "auc_judge"] >= p.loc[lo, "auc_judge"]
            if a != b:
                flipped.append("{}:{}>={}".format(nm, hi, lo))
chk("the four named dominance orderings survive relabelling", not flipped,
    "flipped: {}".format(flipped) if flipped else
    "NONTOPO>=0D, NONTOPO>=DEFL, HIDDEN>=0D, NONTOPO>=PH_0D in all audited settings")

# --- model size range -------------------------------------------------------
chk("ModelSizeRange covers the audited-models table",
    macros.get("ModelSizeRange") == "1.1B--7.62B",
    "macro={}".format(macros.get("ModelSizeRange")))

print("\n".join(ok))
if bad:
    print("\n".join(bad))
print("\n{} passed, {} failed".format(len(ok), len(bad)))
sys.exit(1 if bad else 0)
