"""
toha_native.py -- check our TOHA implementation against the authors' released code.

Reviewers will ask whether the reduction in Proposition 1 is an artefact of our
reimplementation of TOHA. This script answers the narrow, decisive version of that
question: on real attention matrices, does our per-head score equal the one produced by
the authors' own MTop-Div routine?

The two functions below are transcribed verbatim from the official TOHA repository
(https://github.com/sb-ai-lab/TOHA, src/methods/mtopdiv/utils.py:
`transform_attention_scores_to_distances` and `transform_distances_to_mtopdiv`), together
with the prompt-block zeroing and the division by the response length that
`get_mtopdivs` applies. They use ripser, not our dense Prim's algorithm, so agreement is
evidence about the quantity, not a shared bug.

  python -m sinktda.toha_native --model mistral --bench truthfulqa --n 8

Writes sinktda_results/toha_native.csv.

What this does NOT do: run the authors' end-to-end pipeline on their own datasets (CoQA,
RAGTruth). That needs their generated CSVs, a Comet API key and a gated Llama-3.1-8B
checkpoint; see REVIEW_AC.md for why it was not attempted here.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from sinktda import data
from sinktda.toha import toha_head_features

RES = "sinktda_results"


# --- verbatim from the TOHA repository (see module docstring) ----------------------
def transform_attention_scores_to_distances(attention_weights: np.ndarray) -> np.ndarray:
    attention_weights = attention_weights.astype(np.float32)
    n_tokens = attention_weights.shape[-1]
    distance_mx = 1 - np.clip(attention_weights, a_min=0.0, a_max=None)
    zero_diag = np.ones((n_tokens, n_tokens)) - np.eye(n_tokens)
    distance_mx *= np.broadcast_to(zero_diag, distance_mx.shape)
    distance_mx = np.minimum(np.swapaxes(distance_mx, -1, -2), distance_mx)
    return distance_mx


def transform_distances_to_mtopdiv(distance_mx: np.ndarray) -> float:
    from ripser import ripser

    barcodes = ripser(distance_mx, distance_matrix=True, maxdim=0)["dgms"]
    if len(barcodes) > 0:
        return barcodes[0][:-1, 1].sum()
    return 0
# --- end verbatim -----------------------------------------------------------------


def native_score(attn_head: np.ndarray, response_length: int) -> float:
    """One head, exactly as get_mtopdivs() does it."""
    d = transform_attention_scores_to_distances(attn_head)
    d[:-response_length, :-response_length] = 0
    return transform_distances_to_mtopdiv(d) / response_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral", choices=list(data.MODELS))
    ap.add_argument("--bench", default="truthfulqa", choices=["truthfulqa"])
    ap.add_argument("--n", type=int, default=8, help="examples (each gives L*H head graphs)")
    ap.add_argument("--layers", type=int, default=0,
                    help="if >0, use only this many evenly spaced layers (speed)")
    a = ap.parse_args()

    import torch
    from sinktda.extract import load, pick_device

    model_id, template = data.MODELS[a.model]
    device, dtype = pick_device()
    tok, model = load(model_id, device, dtype, attn="eager")
    print(f"[load] {model_id} {device} {dtype}", flush=True)

    rows = list(data.truthfulqa_rows(tok, template, a.n))
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        enc = data.encode(tok, r["full"], return_tensors="pt").to(device)
        N = enc["input_ids"].shape[1]
        p = min(len(data.encode(tok, r["prefix"].rstrip(" "))["input_ids"]), N - 1)
        if N > 512:            # keep the ripser reference cheap
            continue
        with torch.no_grad():
            o = model(**enc, output_attentions=True)
        att = torch.stack([x[0].float().cpu() for x in o.attentions]).numpy().astype(np.float64)
        L, H = att.shape[0], att.shape[1]
        idx = range(L) if a.layers <= 0 else np.linspace(0, L - 1, a.layers).astype(int)
        # ours: response rows only, dense Prim on the contracted graph
        ours = toha_head_features(att[:, :, p:, :].reshape(L * H, N - p, N), p)["toha"].reshape(L, H)
        for l in idx:
            for h in range(H):
                nat = native_score(att[l, h], N - p)
                out.append(dict(setting=f"{a.bench}_{a.model}", example=i, layer=int(l), head=h,
                                ours=float(ours[l, h]), native=float(nat),
                                abs_diff=abs(float(ours[l, h]) - float(nat))))
        print(f"[native] {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(out)
    os.makedirs(RES, exist_ok=True)
    # keep rows for other settings so several models accumulate in one file
    path = f"{RES}/toha_native.csv"
    if os.path.exists(path):
        prev = pd.read_csv(path)
        prev = prev[prev["setting"] != f"{a.bench}_{a.model}"]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)
    df = df[df["setting"] == f"{a.bench}_{a.model}"]
    print(f"\nhead graphs compared : {len(df):,}")
    print(f"max |ours - native|  : {df['abs_diff'].max():.3e}")
    print(f"mean |ours - native| : {df['abs_diff'].mean():.3e}")
    print(f"share within 1e-5    : {(df['abs_diff'] <= 1e-5).mean():.6f}")
    print(f"corr(ours, native)   : {df['ours'].corr(df['native']):.8f}")
    print(f"wrote {RES}/toha_native.csv")


if __name__ == "__main__":
    main()
