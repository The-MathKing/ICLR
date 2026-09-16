"""
verify_mst_equivalence.py
==========================
Implementation-level sanity check for Theorem 1: computes both the Ripser 0D
total persistence and the Kruskal MST weight on the identical attention
distance matrix D^(l), for real attention graphs, and asserts they agree to
floating-point precision. This is expected from the proof (the two are the
same number by construction), not a new empirical finding -- this script
exists to rule out an implementation-level discrepancy (e.g. in how ties or
disconnected components are handled by a specific library) rather than to
test the mathematical claim itself.

Uses int8 CPU quantization (see phase10_mistral_eval.py) since a 14GB fp16
7B model does not reliably fit alongside the rest of the pipeline on
16GB-RAM evaluation hardware; any cached instruct model works equally well
for this check since the identity is exact regardless of model or precision.

Usage:
    python verify_mst_equivalence.py [--model MODEL_NAME] [--n_examples N]

Outputs: master_results/mst_equivalence_verification.csv
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import scipy.sparse.csgraph as csgraph
import ripser
from datasets import load_dataset


def kruskal_mst_weight(D):
    return float(csgraph.minimum_spanning_tree(D).sum())


def ripser_h0_total_persistence(D):
    res = ripser.ripser(D, distance_matrix=True, maxdim=0)
    h0 = res["dgms"][0]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    return float(np.sum(h0_finite[:, 1] - h0_finite[:, 0]))


def main(model_name: str, n_examples: int):
    print(f"Loading {model_name} (int8, CPU) ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True, device_map={"": "cpu"},
        quantization_config=QuantoConfig(weights="int8"),
    )
    model.eval()

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation").select(range(n_examples))

    rows = []
    for i, ex in enumerate(ds):
        text = f"Question: {ex['question']}\nAnswer: {ex['best_answer']}"
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
        N = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        for layer_idx, attn in enumerate(out.attentions):
            avg_attn = attn[0].float().numpy().mean(axis=0)
            W = np.maximum(avg_attn, avg_attn.T)
            D = (1.0 - W).astype(np.float64)
            np.fill_diagonal(D, 0.0)

            t0 = time.perf_counter(); mst_w = kruskal_mst_weight(D); t_mst = time.perf_counter() - t0
            t0 = time.perf_counter(); h0_total = ripser_h0_total_persistence(D); t_ripser = time.perf_counter() - t0

            abs_diff = abs(mst_w - h0_total)
            rows.append({
                "example_id": i, "layer": layer_idx, "N": N,
                "mst_weight": mst_w, "ripser_h0_total_persistence": h0_total,
                "abs_diff": abs_diff, "time_mst_s": t_mst, "time_ripser_h0_s": t_ripser,
            })
        print(f"[{i+1}/{n_examples}] N={N} max|diff| this example: "
              f"{max(r['abs_diff'] for r in rows if r['example_id']==i):.2e}")

    df = pd.DataFrame(rows)
    os.makedirs("master_results", exist_ok=True)
    df.to_csv("master_results/mst_equivalence_verification.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"Rows (examples x layers): {len(df)}")
    print(f"Max |MST - H0_total_persistence|: {df['abs_diff'].max():.2e}")
    assert df["abs_diff"].max() < 1e-6, "Theorem 1 identity violated beyond floating-point tolerance!"
    print("PASS: MST weight == Ripser H0 total persistence to floating-point precision.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--n_examples", type=int, default=25)
    args = p.parse_args()
    main(args.model, args.n_examples)
