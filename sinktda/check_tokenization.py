"""
check_tokenization.py -- tokenizer-only regression test for the prompt/response boundary.

For every model and template the study uses, the token ids of the prefix (tokenized the
way the extractors tokenize it: `data.encode(tok, prefix.rstrip(" "))`) must be a prefix
of the token ids of the full string (`data.encode(tok, full)`). If they are not, the
stored `prompt_len` is wrong and every response-restricted statistic is off by the
mismatch.

The failure this guards against: chat templates that emit BOS themselves (Mistral-7B's
`generic_chat`, Llama-3.1) get a second BOS from a raw `tok(prefix)` call, so the prefix
comes out one token too long and the boundary drops the first answer token.

    python -m sinktda.check_tokenization            # all cached models
    python -m sinktda.check_tokenization mistral    # one model key

Exit status is non-zero if any checked pair fails. Models whose tokenizer is not
available locally (gated weights) are reported as skipped, not as failures.
No GPU and no dataset download are needed.
"""
import sys

from sinktda import data

Q = "What happens to you if you eat watermelon seeds?"
A = "The watermelon seeds pass through your digestive system."

# (model key, template) pairs: every model with its default template, plus the
# extra template the study runs for Mistral.
PAIRS = [(k, t) for k, (_, t) in data.MODELS.items()] + [("mistral", "generic_chat")]


def _ids(tok, text):
    return data.encode(tok, text)["input_ids"]


def _raw_ids(tok, text):
    return tok(text)["input_ids"]


def check_pair(tok, full, prefix):
    """Return (ok, detail) for one (full, prefix) pair."""
    pre = _ids(tok, prefix.rstrip(" "))
    fu = _ids(tok, full)
    ok = fu[:len(pre)] == pre
    raw = _raw_ids(tok, prefix.rstrip(" "))
    raw_ok = fu[:len(raw)] == raw
    detail = f"prefix={len(pre)} full={len(fu)} raw_prefix={len(raw)}"
    if not raw_ok:
        detail += "  (raw tok(prefix) would be WRONG here)"
    return ok, detail


def main(keys=None):
    from transformers import AutoTokenizer
    keys = set(keys) if keys else None
    n_fail = n_pass = n_skip = 0
    for key, template in PAIRS:
        if keys and key not in keys:
            continue
        model_id = data.MODELS[key][0]
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:  # gated or not cached
            print(f"[skip] {key:10s} {template:13s} {type(e).__name__}: tokenizer unavailable")
            n_skip += 1
            continue
        cases = {}
        full, pre = data._tqa_strings(template, tok, Q, A)
        cases["tqa"] = (full, pre)
        if template == data.MODELS[key][1]:  # default template: also the on-policy strings
            cases["onpolicy"] = (data.onpolicy_full(tok, Q, A), data.onpolicy_prompt(tok, Q))
            cases["halueval"] = (f"Knowledge: k\nQuestion: {Q}\nAnswer: {A}",
                                 f"Knowledge: k\nQuestion: {Q}\nAnswer: ")
        for name, (f, p) in cases.items():
            ok, detail = check_pair(tok, f, p)
            tag = "ok  " if ok else "FAIL"
            print(f"[{tag}] {key:10s} {template:13s} {name:9s} {detail}")
            if ok:
                n_pass += 1
            else:
                n_fail += 1
    print(f"[done] {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
