"""
data.py -- benchmark row builders. Each builder yields dicts with
  example_id, label ('grounded'|'hallucinated'), full (text), prefix (text), answer (text)

TruthfulQA/HaluEval builders reproduce the legacy phase* prompts exactly.
"""
import re
import string

import pandas as pd
from datasets import load_dataset

FULL_TRUTHFULQA = 817

# model key -> (hf id, template family)
MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", "qwen_chat"),
    "qwen1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "qwen_chat"),
    "phi3": ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
    "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama"),
    "smollm": ("HuggingFaceTB/SmolLM-1.7B-Instruct", "smollm"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.2", "mistral_inst"),
}


def _tqa_strings(template, tok, q, a):
    if template == "qwen_chat":
        sys = {"role": "system", "content": "You are a helpful assistant."}
        full = tok.apply_chat_template([sys, {"role": "user", "content": q},
                                        {"role": "assistant", "content": a}],
                                       tokenize=False, add_generation_prompt=False)
        pre = tok.apply_chat_template([sys, {"role": "user", "content": q}],
                                      tokenize=False, add_generation_prompt=True)
        return full, pre
    if template == "phi3":
        pre = f"<|user|>\n{q}<|end|>\n<|assistant|>\n"
        return pre + a, pre
    if template == "tinyllama":
        pre = f"<|user|>\n{q}</s>\n<|assistant|>\n"
        return pre + a, pre
    if template == "smollm":
        pre = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        return pre + a, pre
    if template == "mistral_inst":
        pre = f"[INST] {q} [/INST] "
        return f"[INST] {q} [/INST] {a}", pre
    if template == "generic_chat":  # model's own chat template, no system turn
        full = tok.apply_chat_template([{"role": "user", "content": q},
                                        {"role": "assistant", "content": a}],
                                       tokenize=False, add_generation_prompt=False)
        pre = tok.apply_chat_template([{"role": "user", "content": q}],
                                      tokenize=False, add_generation_prompt=True)
        return full, pre
    raise ValueError(template)


def truthfulqa_rows(tok, template, n=FULL_TRUTHFULQA):
    df = pd.DataFrame(load_dataset("truthfulqa/truthful_qa", "generation", split="validation")).head(n)
    for idx, row in df.iterrows():
        q = row["question"]
        bad = row["incorrect_answers"][0] if len(row["incorrect_answers"]) > 0 else "I don't know."
        for lab, a in (("grounded", row["best_answer"]), ("hallucinated", bad)):
            full, pre = _tqa_strings(template, tok, q, a)
            yield dict(example_id=int(idx), label=lab, full=full, prefix=pre, answer=a, question=q)


def halueval_rows(tok, template, n=2000):
    ds = load_dataset("pminervini/HaluEval", "qa", split=f"data[:{n}]")
    df = pd.DataFrame(ds)
    for idx, row in df.iterrows():
        pre = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer: "
        for lab, key in (("grounded", "right_answer"), ("hallucinated", "hallucinated_answer")):
            a = str(row[key])
            yield dict(example_id=int(idx), label=lab, full=pre + a, prefix=pre, answer=a,
                       question=row["question"])


# ---------------------------------------------------------------------------
# on-policy TriviaQA
# ---------------------------------------------------------------------------
ONPOLICY_INSTR = "Answer the following question with a short phrase only, no explanation.\nQuestion: "


def triviaqa_questions(n=2000, seed=0):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    df = pd.DataFrame({"question": ds["question"], "qid": ds["question_id"],
                       "aliases": [a["normalized_aliases"] for a in ds["answer"]]})
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def onpolicy_prompt(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": ONPOLICY_INSTR + q}],
                                   tokenize=False, add_generation_prompt=True)


def onpolicy_full(tok, q, a):
    return tok.apply_chat_template([{"role": "user", "content": ONPOLICY_INSTR + q},
                                    {"role": "assistant", "content": a}],
                                   tokenize=False, add_generation_prompt=False)


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred, aliases):
    p = " " + normalize_answer(pred) + " "
    for al in aliases:
        a = normalize_answer(al)
        if a and (" " + a + " ") in p:
            return True
    return False
