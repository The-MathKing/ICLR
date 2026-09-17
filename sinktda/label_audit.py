"""
label_audit.py -- audit the on-policy TriviaQA string-match labels with an LLM judge.

The on-policy benchmark labels an answer "grounded" iff a normalized gold alias occurs in
it (sinktda.data.is_correct). That rule mislabels paraphrases ("the Bard" for
"Shakespeare"), hedged answers and answers that contain a gold alias while asserting
something false. This script samples rows per setting, asks Qwen2.5-7B-Instruct to judge
them, and reports how often the judge disagrees with the string match.

The judge is itself a language model, not a human annotator. A disagreement is evidence
that the label is uncertain, not proof that the string match is wrong: it is reported as
a disagreement rate, and both directions are broken out.

  python -m sinktda.label_audit --n 200 [--settings triviaqa_mistral ...]

Writes sinktda_results/label_audit.csv (one row per judged answer) and prints a summary.
"""
import argparse
import ast
import os
import time

import numpy as np
import pandas as pd

OUT_ROOT = os.environ.get("SINKTDA_OUT", "sinktda_out")
RES = "sinktda_results"
JUDGE = "Qwen/Qwen2.5-7B-Instruct"

PROMPT = (
    "You are grading a trivia answer. Decide whether the ANSWER is a correct response to "
    "the QUESTION, given the list of accepted gold answers.\n\n"
    "Judge the meaning, not the wording: an answer that means the same as an accepted "
    "answer is correct even if worded differently. An answer that is vague, refuses, or "
    "states something false is incorrect, even if it happens to contain one of the words.\n\n"
    "QUESTION: {q}\nACCEPTED ANSWERS: {a}\nANSWER: {p}\n\n"
    "Reply with exactly one word: correct or incorrect."
)


def _aliases(cell, k=12):
    """generations.csv stores the normalized alias list as a repr'd python list."""
    try:
        v = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError):
        v = [str(cell)]
    if not isinstance(v, (list, tuple)):
        v = [str(v)]
    return [str(x) for x in v][:k]


BANKS = ["NONTOPO", "HIDDEN", "LOOKBACK", "0D", "SINK", "DEFL", "PH_0D", "PH_SINK", "LOGPROB", "LEX"]


def sensitivity(df):
    """Do the detection conclusions survive the judge's labels?

    Re-scores the existing out-of-fold predictions on the audited rows under the
    string-match labels and under the judge's, so the comparison isolates the labelling
    and refits nothing. n is only the audited subset, so these AUCs are noisy in absolute
    terms; what matters is whether the ordering of the banks changes.
    """
    from sinktda.evaluate import fast_auc

    out = []
    for name, a in df.groupby("setting"):
        a = a[a["judge"] >= 0]
        f = f"{RES}/oof/{name}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        pos = {int(v): i for i, v in enumerate(z["g"])}
        if not set(a["row"]).issubset(pos):
            continue
        rp = [pos[r] for r in a["row"]]
        y_str, y_jud = z["y"][rp], 1 - a["judge"].values
        assert (y_str == 1 - a["string_match"].values).all(), f"{name}: oof/label mismatch"
        for b in BANKS:
            if b not in z or len(set(y_str)) < 2 or len(set(y_jud)) < 2:
                continue
            p = z[b][rp]
            s, j = fast_auc(y_str, p), fast_auc(y_jud, p)
            out.append(dict(setting=name, bank=b, n=len(rp), auc_string=s, auc_judge=j, delta=j - s))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="rows sampled per setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settings", nargs="*", default=None)
    a = ap.parse_args()

    import torch
    from sinktda import data
    from sinktda.extract import load, pick_device

    names = a.settings or sorted(
        d for d in os.listdir(OUT_ROOT)
        if d.startswith("triviaqa_") and os.path.exists(os.path.join(OUT_ROOT, d, "generations.csv")))
    if not names:
        print("no triviaqa settings with generations.csv")
        return

    device, dtype = pick_device()
    tok, model = load(JUDGE, device, dtype, attn="eager")
    print(f"[judge] {JUDGE} {device} {dtype}", flush=True)
    print(f"[judge] settings: {names}", flush=True)

    rows, t0 = [], time.time()
    for name in names:
        g = pd.read_csv(os.path.join(OUT_ROOT, name, "generations.csv"), keep_default_na=False)
        idx = np.random.default_rng(a.seed).choice(len(g), size=min(a.n, len(g)), replace=False)
        for j, i in enumerate(sorted(idx.tolist())):
            r = g.iloc[i]
            msg = PROMPT.format(q=r["question"], a="; ".join(_aliases(r["aliases"])), p=r["pred"])
            text = tok.apply_chat_template([{"role": "user", "content": msg}],
                                           tokenize=False, add_generation_prompt=True)
            enc = data.encode(tok, text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            ans = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            judge = 1 if ans.startswith("correct") else (0 if ans.startswith("incorrect") else -1)
            rows.append(dict(setting=name, row=int(i), question=r["question"], pred=r["pred"],
                             string_match=int(bool(r["correct"])), judge=judge, judge_raw=ans))
            if j % 50 == 0:
                print(f"[audit] {name} {j + 1}/{len(idx)} {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(f"{RES}/label_audit.csv", index=False)

    print("\n=== label audit (judge is Qwen2.5-7B-Instruct, not a human) ===")
    for name, d in df.groupby("setting"):
        ok = d[d["judge"] >= 0]
        dis = (ok["judge"] != ok["string_match"]).mean()
        fn = ((ok["string_match"] == 0) & (ok["judge"] == 1)).sum()   # string match too strict
        fp = ((ok["string_match"] == 1) & (ok["judge"] == 0)).sum()   # string match too lax
        note = "  <-- judge is the same model that produced these answers" if name.endswith("qwen7b") else ""
        print(f"{name:20s} n={len(ok):4d} unparsed={int((d['judge'] < 0).sum()):3d} "
              f"disagree={dis:.3f}  match=0/judge=1: {fn:3d}  match=1/judge=0: {fp:3d}{note}")
    sens = sensitivity(df)
    if len(sens):
        sens.to_csv(f"{RES}/label_audit_sensitivity.csv", index=False)
        print("\n=== AUC on the audited rows under each labelling (same predictions) ===")
        # A full ranking flips whenever two near-tied banks swap, which says nothing about
        # the paper. Check instead the dominance claims the paper actually makes.
        CLAIMS = [("NONTOPO", "0D"), ("NONTOPO", "DEFL"), ("HIDDEN", "0D"), ("NONTOPO", "PH_0D")]
        for name, d in sens.groupby("setting"):
            p = d.set_index("bank")
            kept, flipped = [], []
            for hi, lo in CLAIMS:
                if hi not in p.index or lo not in p.index:
                    continue
                s = p.loc[hi, "auc_string"] >= p.loc[lo, "auc_string"]
                j = p.loc[hi, "auc_judge"] >= p.loc[lo, "auc_judge"]
                (kept if s == j else flipped).append(f"{hi}>={lo}")
            print(f"{name}: max |delta| = {d['delta'].abs().max():.3f}; "
                  f"dominance claims unchanged {len(kept)}/{len(kept) + len(flipped)}"
                  + (f"; FLIPPED: {flipped}" if flipped else ""))
        print(f"wrote {RES}/label_audit_sensitivity.csv")

    print(f"\nwrote {RES}/label_audit.csv")


if __name__ == "__main__":
    main()
