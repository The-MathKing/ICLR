"""
c_grid.py -- is any reported AUC an artefact of the narrow regularization grid?

evaluate.py fits banks of at most 64 dimensions at a fixed C=1 and wider banks at a C
chosen from {0.01, 0.1, 1} by inner grouped 3-fold CV. That grid is narrow for the widest
banks (the hidden-state probe is 1,536-4,096 dimensions), and an under-regularized
baseline would understate exactly the comparisons the paper leans on -- topological banks
against cheap non-topological ones.

This re-runs every dense bank of every setting whose features we retain under the wider
grid {1e-4, ..., 1e2}, applied to *every* bank regardless of width, and reports the change
in pooled out-of-fold AUC against the number the paper reports. The protocol is otherwise
identical (same seeds, same splitter, same scaler), and it reuses evaluate.oof_predictions
by patching the two module-level knobs that define the rule, so there is no second copy of
the fitting code to drift.

LEX and the LEX combos are excluded: their TF-IDF vocabulary is refit inside every fold
through a separate sparse path, so they are not a dense bank with a fixed width. The banks
covered are the ones the reported AUC table shows (BANKS below), which include both of the
widest banks in the study.

  python -m sinktda.c_grid                  # every setting found in sinktda_out/
  python -m sinktda.c_grid truthfulqa_qwen3b
Writes sinktda_results/c_grid.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sinktda import evaluate as ev
from sinktda.evaluate import COMBOS, RES, banks, combine, fast_auc

WIDE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
# The banks that appear in the reported AUC table: every baseline the grid could
# disadvantage (HIDDEN and NONTOPO are the widest banks in the study) and every
# topological bank they are compared against. LEX and the LEX combos are excluded above.
BANKS = ("SINK", "0D", "1D", "DEFL", "ANS", "LEN", "ROWSTAT", "LOGPROB", "LLMCHECK",
         "LOOKBACK", "PH_SINK", "PH_0D", "PH_0DTOT", "PH_ENT", "HIDDEN", "NONTOPO")


def _bank_auc(k, X, y, g):
    """Run in a worker, so the widened knobs have to be set inside it as well."""
    ev.C_GRID, ev.TUNE_MIN_DIM = WIDE_GRID, 0
    p, _ = ev.oof_predictions(X, y, g)
    return k, X.shape[1], fast_auc(y, p)


def _load(name):
    """load_setting, but tolerant of a setting whose large arrays are no longer on disk.

    Four settings were evaluated when hidden.npy and perhead.npz existed and kept only
    layers.parquet afterwards. Their reported AUCs are still in sinktda_results/, so the
    layer-level banks can still be re-checked under the wider grid; the hidden-state and
    per-head banks simply are not rebuildable and are reported as absent rather than
    crashing the run.
    """
    d = os.path.join(ev.ROOT, name)
    df = pd.read_parquet(os.path.join(d, "layers.parquet"))
    df["y"] = (df["label"] == "hallucinated").astype(int)
    ph_path, hid_path = os.path.join(d, "perhead.npz"), os.path.join(d, "hidden.npy")
    ph = dict(np.load(ph_path)) if os.path.exists(ph_path) else {}
    if os.path.exists(hid_path):
        hid = np.load(hid_path).astype(np.float32)
    else:
        hid = None
        print(f"    [partial] {name}: no hidden.npy/perhead.npz, layer banks only", flush=True)
    return df, ph, hid


def wide_auc(name):
    """Pooled OOF AUC of every dense bank under the wider grid."""
    df, ph, hid = _load(name)
    y = df["y"].values
    g = df["example_id"].values
    B = banks(df, ph, hid if hid is not None else np.zeros((len(df), 1, 1), dtype=np.float32))
    if hid is None:
        B.pop("HIDDEN", None)
    for k, v in COMBOS.items():
        if all(n in B for n in v):
            B[k] = combine(B, v)
    jobs = [(k, B[k]) for k in BANKS if B.get(k) is not None]
    out = Parallel(n_jobs=ev.N_JOBS)(delayed(_bank_auc)(k, X, y, g) for k, X in jobs)
    return pd.DataFrame([dict(setting=name, bank=k, dim=dim, auc_wide=a) for k, dim, a in out])


def main():
    resume_all = "--all" in sys.argv[1:]
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = names or sorted(d for d in os.listdir(ev.ROOT)
                                   if os.path.exists(os.path.join(ev.ROOT, d, "layers.parquet")))
    # The rule under test is "C=1 below 65 dims, tuned above"; widening it means tuning
    # every bank over the wider grid, so both knobs move together.
    ev.C_GRID = WIDE_GRID
    ev.TUNE_MIN_DIM = 0
    done = set(pd.read_csv(f"{RES}/c_grid.csv")["setting"]) if os.path.exists(f"{RES}/c_grid.csv") else set()
    for name in names:
        f = f"{RES}/auc_{name}.csv"
        if not os.path.exists(f):
            print(f"[skip] {name}: no reported AUC to compare against")
            continue
        if name in done and not resume_all:
            print(f"[skip] {name}: already in c_grid.csv")
            continue
        print(f"[c-grid] {name}", flush=True)
        w = wide_auc(name)
        rep = pd.read_csv(f).set_index("bank")["auc"]
        w["auc_reported"] = w["bank"].map(rep)
        w["delta"] = w["auc_wide"] - w["auc_reported"]
        w = w.dropna(subset=["auc_reported"])
        # one write per setting: the run takes hours and should not lose it all to a crash
        ev.merge_csv(f"{RES}/c_grid.csv", w)
        print(w.sort_values("delta", key=abs, ascending=False)
              .head(5)[["bank", "dim", "auc_reported", "auc_wide", "delta"]].to_string(index=False), flush=True)
    if os.path.exists(f"{RES}/c_grid.csv"):
        d = pd.read_csv(f"{RES}/c_grid.csv")
        i = d["delta"].abs().idxmax()
        print(f"\n{d['setting'].nunique()} settings; max |delta| = {d.loc[i, 'delta']:+.4f} "
              f"({d.loc[i, 'setting']} / {d.loc[i, 'bank']})")


if __name__ == "__main__":
    main()
