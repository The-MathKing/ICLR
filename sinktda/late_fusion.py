"""
late_fusion.py -- incremental value of a small feature bank under late fusion.

Early fusion (evaluate.py) appends e.g. 72 topological columns to a ~2,000-dim
non-topological bank; a regularized model can then ignore the small block whatever
it contains. Here each bank is first reduced to its own out-of-fold score (already
stored in sinktda_results/oof/<setting>.npz), and a second-level logistic regression
on [logit p_base, logit p_extra] is compared with one on [logit p_base] alone, using
the same grouped repeated CV and question-level bootstrap as every other number.

  python -m sinktda.late_fusion
Writes sinktda_results/late_fusion.csv
"""
import glob
import os

import numpy as np
import pandas as pd

from sinktda.evaluate import RES, bootstrap_delta, fast_auc, oof_predictions

PAIRS = [("NONTOPO", "0D"), ("NONTOPO", "DEFL"), ("NONTOPO", "SINK"),
         ("HIDDEN", "0D"), ("LOGPROB", "0D"), ("SINK", "0D")]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    rows = []
    for f in sorted(glob.glob(f"{RES}/oof/*.npz")):
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        y, g = z["y"], z["g"]
        for base, extra in PAIRS:
            if base not in z or extra not in z:
                continue
            pb, _ = oof_predictions(logit(z[base])[:, None], y, g)
            pc, _ = oof_predictions(np.column_stack([logit(z[base]), logit(z[extra])]), y, g)
            r = dict(setting=name, test=f"LF_{extra}_beyond_{base}", A=base, B=f"{base}(+){extra}",
                     auc_A=fast_auc(y, pb), auc_B=fast_auc(y, pc))
            r.update(bootstrap_delta(y, pb, pc, g))
            rows.append(r)
            print(name, r["test"], f"{r['delta']:+.4f} [{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}]", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/late_fusion.csv", index=False)


if __name__ == "__main__":
    main()
