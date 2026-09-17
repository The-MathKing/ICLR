"""
timing.py -- per-matrix wall-clock cost on REAL attention (a model forward pass),
for N in {32, 64, 128, 256}, vs dense i.i.d. random matrices at the same N.

  python -m sinktda.timing --model tinyllama
Writes sinktda_results/timing.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import ripser
import torch
from datasets import load_dataset

from sinktda import data
from sinktda.extract import load, pick_device
from sinktda.features import batched_prim, delta_bos_causal, distance_from_attention


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layers", type=int, default=6, help="layers sampled per N")
    args = ap.parse_args()
    model_id, _ = data.MODELS[args.model]
    device, dtype = pick_device()
    tok, model = load(model_id, device, dtype)
    text = " ".join(load_dataset("pminervini/HaluEval", "qa", split="data[:40]")["knowledge"])
    ids = tok(text, return_tensors="pt")["input_ids"]
    rng = np.random.default_rng(0)
    rows = []
    for N in (32, 64, 128, 256):
        with torch.no_grad():
            out = model(input_ids=ids[:, :N].to(device), output_attentions=True)
        L = len(out.attentions)
        for l in np.linspace(0, L - 1, args.layers).astype(int):
            A = out.attentions[l][0].float().mean(0).cpu().numpy().astype(np.float64)
            D = distance_from_attention(A)
            R = rng.uniform(size=(N, N))
            R = np.triu(R, 1)
            R = R + R.T
            for kind, M in (("real", D), ("iid", R)):
                rows.append(dict(N=N, layer=int(l), kind=kind,
                                 sink_ms=timeit(lambda: (M[0, 1:].sum(), M[0, 1:].max()), args.reps),
                                 delta0_ms=timeit(lambda: delta_bos_causal(A), args.reps) if kind == "real" else np.nan,
                                 mst_ms=timeit(lambda: batched_prim(M[None]), args.reps),
                                 ripser0_ms=timeit(lambda: ripser.ripser(M, distance_matrix=True, maxdim=0), args.reps),
                                 ripser01_ms=timeit(lambda: ripser.ripser(M, distance_matrix=True, maxdim=1),
                                                    args.reps if (kind == "real" or N <= 128) else 1)))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows).groupby(["kind", "N"]).median(numeric_only=True).reset_index()
    df["model"] = model_id
    df.to_csv("sinktda_results/timing.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
