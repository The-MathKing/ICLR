#!/usr/bin/env python3
"""
rerun_mistral_native.py -- roadmap item 6.

Closes Limitation 1. The paper's cleanest result -- Mistral-7B TOST-CONFIRMED at
every margin -- rests on a run where 99.99% of 1D features are exactly zero, and
the paper cannot say whether that degeneracy is architectural or an artifact of
the int8 CPU quantization forced by 16GB of RAM:

    "We were NOT able to determine how much of factor (ii) is specific to int8
     quantization versus Mistral-7B's architecture at native precision: an fp16
     comparison run would require holding a ~14GB model in memory on our 16GB
     evaluation machine."

32GB of system RAM removes that blocker outright. This script re-extracts the
same rows at native precision and reports the 1D degeneracy rate side by side
with the int8 run, so the question is answered either way:

  * still ~99.99% zero  -> degeneracy is architectural / sequence-length driven,
                           and the Mistral row stops being caveated;
  * materially lower    -> the published Mistral equivalence result is a
                           quantization artifact and must be re-reported. That
                           is a real finding too, and better found by you.

Memory note: Mistral-7B at bf16 is ~14.5 GiB of weights. On a 16 GiB card that
fits but leaves little headroom, which is fine here because these sequences are
~33 tokens. If you hit OOM, use --cpu: 32 GiB of system RAM holds the bf16 model
comfortably, which is exactly the configuration the paper could not run.

    python rigor/rerun_mistral_native.py                 # GPU, bf16
    python rigor/rerun_mistral_native.py --cpu           # CPU, float32
    python rigor/rerun_mistral_native.py --compare-only  # just the comparison
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from rigor.device import assert_gpu_usable, empty_cache, get_device, get_dtype, hf_cache_note, report
from rigor import settings as S

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
INT8_CSV = "phase10_results/mistral7b_truthfulqa_4stat_full.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(OUT_DIR, "mistral7b_truthfulqa_native.csv")
MAX_SEQ_LEN = 1024


def extract_features_from_dgms(dgms):
    """Byte-for-byte the same descriptor bank as phase10_*_eval.py."""
    import numpy as np
    feats = {}
    h0 = dgms[0]
    h0f = h0[h0[:, 1] != np.inf] if len(h0) > 0 else np.array([])
    if len(h0f) > 0:
        lt = h0f[:, 1] - h0f[:, 0]
        feats["h0_max_lifetime"] = float(np.max(lt))
        feats["h0_total_persistence"] = float(np.sum(lt))
    else:
        feats["h0_max_lifetime"] = 0.0
        feats["h0_total_persistence"] = 0.0
    h1 = dgms[1]
    h1f = h1[h1[:, 1] != np.inf] if len(h1) > 0 else np.array([])
    if len(h1f) > 0:
        lt = h1f[:, 1] - h1f[:, 0]
        k = int(np.argmax(lt))
        feats["h1_max_lifetime"] = float(np.max(lt))
        feats["h1_total_persistence"] = float(np.sum(lt))
        feats["h1_max_birth"] = float(h1f[k, 0])
        feats["h1_max_death"] = float(h1f[k, 1])
    else:
        feats["h1_max_lifetime"] = 0.0
        feats["h1_total_persistence"] = 0.0
        feats["h1_max_birth"] = 0.0
        feats["h1_max_death"] = 0.0
    return feats


def out_path(limit, fmt="inst"):
    """Cache key must include BOTH the row count and the prompt format, or a
    smoke test in one format silently satisfies a request for the other."""
    tag = f"_{fmt}"
    if limit is not None and limit < S.FULL_TRUTHFULQA:
        tag += f"_first{limit}"
    return OUT_CSV.replace(".csv", f"{tag}.csv")


def run_extraction(device, dtype, limit=None, fmt="inst"):
    import ripser
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_csv = out_path(limit, fmt)
    if os.path.exists(out_csv):
        print(f"cached -> {out_csv}")
        return pd.read_csv(out_csv)

    print(f"Loading {MODEL} at {dtype} on {device} (NOT quantized) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="eager",
    ).to(device)
    model.eval()

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    n_q = S.FULL_TRUTHFULQA if limit is None else min(limit, S.FULL_TRUTHFULQA)
    df = pd.DataFrame(ds).head(n_q)
    print(f"extracting {n_q} question(s), prompt format={fmt!r} -> {out_csv}")

    rows = []
    for idx, row in df.iterrows():
        question, good, bad = S.truthfulqa_pair(row)
        for label, ans in [("grounded", good), ("hallucinated", bad)]:
            if fmt == "inst":   # matches phase10_mistral_eval.py exactly
                full = S.mistral_inst_full(question, ans)
            else:               # matches every OTHER TruthfulQA setting
                full = S.truthfulqa_chat_full(tokenizer, question, ans)
            inputs = tokenizer(full, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            if seq_len > MAX_SEQ_LEN:
                continue
            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            rec = {"example_id": idx, "label": label, "seq_len": seq_len}
            for l, attn in enumerate(out.attentions):
                a = np.nan_to_num(attn.float().cpu().numpy()[0], nan=0.0)
                avg = a.mean(axis=0)
                W = np.maximum(avg, avg.T)
                D = 1.0 - W
                np.fill_diagonal(D, 0.0)
                res = ripser.ripser(D, distance_matrix=True, maxdim=1)
                for k, v in extract_features_from_dgms(res["dgms"]).items():
                    rec[f"layer_{l}_{k}"] = v
            rows.append(rec)
            del out
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(df)}")
            empty_cache(device)

    del model
    gc.collect()
    empty_cache(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"saved {len(out_df)} rows -> {out_csv}")
    return out_df


def degeneracy(df, tag):
    cols = [c for c in df.columns if "_h1_" in c]
    if not cols:
        return None
    vals = df[cols].to_numpy(dtype=float)
    cells = vals.size
    nonzero_cells = int(np.count_nonzero(vals))
    rows_nonzero = int((vals != 0).any(axis=1).sum())
    stat = {
        "run": tag,
        "n_rows": len(df),
        "n_1d_cols": len(cols),
        "pct_1d_cells_exactly_zero": round(100.0 * (cells - nonzero_cells) / cells, 4),
        "n_rows_with_any_nonzero_1d": rows_nonzero,
        "pct_rows_with_any_nonzero_1d": round(100.0 * rows_nonzero / len(df), 4),
        "mean_seq_len": round(float(df["seq_len"].mean()), 2) if "seq_len" in df else None,
    }
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true",
                    help="run on CPU in float32 (needs ~28GB RAM; you have 32)")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--prompt-format", choices=["inst", "chat"], default="inst",
                    help="'inst' reproduces the published Mistral protocol "
                         "([INST] ... [/INST], no system turn). 'chat' uses the "
                         "template every OTHER TruthfulQA setting uses. Only 'inst' "
                         "is a valid precision comparison against the published run.")
    ap.add_argument("--limit", type=int, default=None,
                    help="extract only the first N questions (smoke test). "
                         "--limit 100 answers the int8-vs-native question in ~20 min.")
    args = ap.parse_args()

    hf_cache_note()
    if args.compare_only:
        cand = out_path(args.limit, args.prompt_format)
        native = pd.read_csv(cand) if os.path.exists(cand) else None
    else:
        device = "cpu" if args.cpu else get_device()
        assert_gpu_usable(device)
        report(device)
        dtype = torch.float32 if args.cpu else get_dtype(device)
        native = run_extraction(device, dtype, limit=args.limit, fmt=args.prompt_format)

    stats = []
    if os.path.exists(INT8_CSV):
        int8 = pd.read_csv(INT8_CSV)
        if native is not None and args.limit:
            # compare like with like: restrict int8 to the same example_ids
            int8 = int8[int8["example_id"].isin(set(native["example_id"]))]
        stats.append(degeneracy(int8, f"int8 (published){' [subset]' if args.limit else ''}"))
    if native is not None:
        tag = (f"native precision, prompt={args.prompt_format}"
               + (f" [first {args.limit} q]" if args.limit else ""))
        stats.append(degeneracy(native, tag))

    if stats:
        out = pd.DataFrame([s for s in stats if s])
        print("\n=== 1D degeneracy: int8 vs native precision ===")
        print(out.to_string(index=False))
        path = os.path.join(OUT_DIR, "mistral_precision_comparison.csv")
        os.makedirs(OUT_DIR, exist_ok=True)
        out.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
        print("\nIf pct_1d_cells_exactly_zero stays ~99.99, the degeneracy is NOT a\n"
              "quantization artifact and Limitation 1 can be closed. If it drops\n"
              "materially, Table 2/3's Mistral row must be re-reported at native\n"
              "precision and the abstract's 'tightest equivalence' claim revised.")


if __name__ == "__main__":
    main()
