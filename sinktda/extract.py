"""
extract.py -- one extractor for every setting in the sink-reduction study.

  python -m sinktda.extract --bench truthfulqa --model qwen3b
  python -m sinktda.extract --bench halueval   --model qwen3b --n 2000 --no-perhead
  python -m sinktda.extract --bench triviaqa   --model phi3   --n 2000
  python -m sinktda.extract --bench truthfulqa --model mistral --template generic_chat --tag chat

Outputs in sinktda_out/<bench>_<model>[_<tag>]/:
  layers.parquet   meta + per-layer features (legacy 0D/1D, sink, delta, deflated, answer-only)
  perhead.npz      (rows, L, H) arrays: per-head 0D, sink mass, entropy, delta0, LLM-Check, Lookback
  hidden.npy       (rows, 3, d) float16 mean-pooled answer-token hidden states (L/2, 3L/4, L)
  generations.csv  (triviaqa only) model answers + correctness
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sinktda import data
from sinktda.features import example_layer_features, perhead_features

OUT_ROOT = os.environ.get("SINKTDA_OUT", "sinktda_out")


def _worker(args):
    i, A_layers, p = args
    return i, example_layer_features(A_layers, p)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16  # fp16 on MPS overflows (NaNs, degenerate generations)
    return "cpu", torch.float32


def load(model_id, device, dtype, attn="eager"):
    """SINKTDA_OFFLOAD=1 keeps the dtype but lets accelerate place layers that do not fit
    in GPU memory on the CPU (needed for 8B models in bf16 on a 16 GB card)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if os.environ.get("SINKTDA_OFFLOAD") == "1" and device == "cuda":
        gpu = os.environ.get("SINKTDA_GPU_MEM", "13GiB")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn,
                                                     device_map="auto", max_memory={0: gpu, "cpu": "64GiB"})
        return tok, model.eval()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    model.to(device).eval()
    return tok, model


def generate_onpolicy(tok, model, device, qdf, path, bs=32, max_new=24):
    import torch
    if os.path.exists(path):
        return pd.read_csv(path, keep_default_na=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    answers = []
    t0 = time.time()
    for s in range(0, len(qdf), bs):
        qs = qdf["question"].iloc[s:s + bs].tolist()
        prompts = [data.onpolicy_prompt(tok, q) for q in qs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            txt = tok.decode(row, skip_special_tokens=True).strip().split("\n")[0].strip()
            answers.append(txt)
        if s % (bs * 10) == 0:
            print(f"[gen] {s + len(qs)}/{len(qdf)}  {time.time() - t0:.0f}s", flush=True)
    g = qdf.copy()
    g["pred"] = answers
    g["correct"] = [data.is_correct(p, al) for p, al in zip(answers, g["aliases"])]
    bad = np.mean([a.strip("!").strip() == "" for a in answers])
    if bad > 0.05:
        raise RuntimeError(f"{100 * bad:.0f}% degenerate generations (numerical overflow?); refusing to continue")
    g["aliases"] = g["aliases"].apply(lambda a: "|".join(a))
    g.to_csv(path, index=False)
    tok.padding_side = "right"
    return g


def onpolicy_rows(tok, gens):
    for i, r in gens.iterrows():
        a = str(r["pred"])
        if not a.strip():
            continue
        yield dict(example_id=int(i), label="grounded" if r["correct"] else "hallucinated",
                   full=data.onpolicy_full(tok, r["question"], a),
                   prefix=data.onpolicy_prompt(tok, r["question"]), answer=a, question=r["question"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["truthfulqa", "halueval", "triviaqa"])
    ap.add_argument("--model", required=True, choices=list(data.MODELS))
    ap.add_argument("--template", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--no-perhead", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--dtype", default=None, choices=[None, "float16", "bfloat16", "float32"],
                    help="default: bfloat16 on CUDA and MPS")
    ap.add_argument("--limit", type=int, default=None, help="debug: stop after this many rows")
    args = ap.parse_args()

    import torch
    model_id, template = data.MODELS[args.model]
    template = args.template or template
    name = f"{args.bench}_{args.model}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "layers.parquet")) and args.limit is None:
        print(f"[skip] {out_dir} exists")
        return

    device, dtype = pick_device()
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    elif args.bench == "triviaqa" and device == "mps":
        dtype = torch.bfloat16
    print(f"[load] {model_id} on {device} ({dtype}); template={template}", flush=True)
    tok, model = load(model_id, device, dtype)

    if args.bench == "truthfulqa":
        rows = list(data.truthfulqa_rows(tok, template, args.n or data.FULL_TRUTHFULQA))
    elif args.bench == "halueval":
        rows = list(data.halueval_rows(tok, template, args.n or 2000))
    else:
        qdf = data.triviaqa_questions(args.n or 2000)
        gens = generate_onpolicy(tok, model, device, qdf, os.path.join(out_dir, "generations.csv"))
        print(f"[gen] accuracy = {gens['correct'].mean():.3f}", flush=True)
        rows = list(onpolicy_rows(tok, gens))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[rows] {len(rows)}", flush=True)

    meta, layer_feats = [], {}
    ph_store, hid_store = {}, []
    pool = ProcessPoolExecutor(max_workers=args.workers)
    pending = []
    nonfinite = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        enc = tok(r["full"], return_tensors="pt").to(device)
        N = enc["input_ids"].shape[1]
        # a trailing space in the prefix ("Answer: ") is merged into the first answer
        # token in the full string, so tokenize the prefix without it
        p = len(tok(r["prefix"].rstrip(" "))["input_ids"])
        p = min(p, N - 1)
        if N > 1024:
            continue
        with torch.no_grad():
            out = model(**enc, output_attentions=True, output_hidden_states=True)
        if not torch.isfinite(out.logits).all():
            nonfinite += 1
            if i < 5 or nonfinite > 0.01 * (i + 1):
                raise RuntimeError(f"non-finite logits at row {i} under {dtype}; rerun with --dtype bfloat16")
        attn = torch.stack(out.attentions)[:, 0].float().cpu().numpy()  # (L,H,N,N)
        attn = np.nan_to_num(attn, nan=0.0)
        L = attn.shape[0]
        pending.append(pool.submit(_worker, (i, attn.mean(1), p)))

        # log-prob statistics over answer tokens
        logits = out.logits[0].float()
        lp = torch.log_softmax(logits[:-1], -1)
        ids = enc["input_ids"][0, 1:]
        tok_lp = lp.gather(1, ids[:, None])[:, 0]
        ent = -(lp.exp() * lp).sum(-1)
        msp = torch.softmax(logits[:-1], -1).max(-1).values.mean().item()
        a0 = max(p - 1, 0)
        seg = tok_lp[a0:] if N - 1 > a0 else tok_lp[-1:]
        eseg = ent[a0:] if N - 1 > a0 else ent[-1:]
        meta.append(dict(row=i, example_id=r["example_id"], label=r["label"], seq_len=N,
                         prompt_len=p, answer_len=N - p, answer=r["answer"],
                         msp_score=msp, lp_mean=seg.mean().item(), lp_min=seg.min().item(),
                         lp_sum=seg.sum().item(), ent_mean=eseg.mean().item(),
                         ent_max=eseg.max().item()))

        hs = out.hidden_states
        take = [L // 2, (3 * L) // 4, L]
        sl = slice(p, N) if N > p else slice(N - 1, N)
        hid_store.append(np.stack([hs[k][0, sl].float().mean(0).cpu().numpy() for k in take]).astype(np.float16))

        if not args.no_perhead:
            for k, v in perhead_features(attn, p).items():
                ph_store.setdefault(k, []).append(v.astype(np.float32))

        if len(pending) >= 4 * args.workers:
            for f in pending[: 2 * args.workers]:
                j, feats = f.result()
                layer_feats[j] = feats
            pending = pending[2 * args.workers:]
        if i % 10 == 0 and device == "mps":
            del out, attn, logits, lp, tok_lp, ent, hs
            torch.mps.empty_cache()
        if i % 200 == 0:
            el = time.time() - t0
            print(f"[x] {i + 1}/{len(rows)}  {el:.0f}s  eta {el / (i + 1) * (len(rows) - i - 1):.0f}s", flush=True)

    for f in pending:
        j, feats = f.result()
        layer_feats[j] = feats
    pool.shutdown()

    m = pd.DataFrame(meta)
    lf = pd.DataFrame([layer_feats[j] for j in m["row"]])
    df = pd.concat([m.reset_index(drop=True), lf], axis=1)
    df["model"] = model_id
    df["template"] = template
    df["dtype"] = str(dtype)
    if args.limit:
        out_dir = out_dir + "_debug"
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "layers.parquet"))
    if ph_store:
        np.savez_compressed(os.path.join(out_dir, "perhead.npz"),
                            **{k: np.stack(v) for k, v in ph_store.items()})
    np.save(os.path.join(out_dir, "hidden.npy"), np.stack(hid_store))
    print(f"[done] {out_dir}  rows={len(df)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
