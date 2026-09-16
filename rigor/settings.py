"""
settings.py -- one description of the seven audited settings, shared by every
script in rigor/.

`n_questions` is the CURRENT cap, mirroring the `.head(N)` calls in the phase
scripts. Roadmap item 7 ("harmonize N") is: set every TruthfulQA setting to
FULL_TRUTHFULQA and re-extract, which removes both the power problem on the
150-group settings and the paper's own caveat that cross-model comparisons are
confounded with sample size.
"""

FULL_TRUTHFULQA = 817  # questions in truthful_qa/generation:validation (1634 rows)

SETTINGS = [
    dict(
        name="HaluEval (Qwen2.5-3B)",
        benchmark="halueval",
        model="Qwen/Qwen2.5-3B-Instruct",
        features="phase3_results/train_features.csv",
        n_questions=2000,
    ),
    dict(
        name="HaluEval (Qwen2.5-1.5B)",
        benchmark="halueval",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        features="phase3_results/train_features_qwen1_5b.csv",
        n_questions=500,
    ),
    dict(
        name="TruthfulQA (Qwen2.5-3B)",
        benchmark="truthfulqa",
        model="Qwen/Qwen2.5-3B-Instruct",
        features="phase10_results/qwen3b_truthfulqa_4stat.csv",
        n_questions=150,          # -> FULL_TRUTHFULQA for item 7
    ),
    dict(
        name="TruthfulQA (SmolLM-1.7B)",
        benchmark="truthfulqa",
        model="HuggingFaceTB/SmolLM-1.7B-Instruct",
        features="phase7_results/truthfulqa_smollm_features.csv",
        n_questions=500,          # -> FULL_TRUTHFULQA
    ),
    dict(
        name="TruthfulQA (Phi-3-mini-3.8B)",
        benchmark="truthfulqa",
        model="microsoft/Phi-3-mini-4k-instruct",
        features="phase10_results/phi3_truthfulqa_4stat.csv",
        n_questions=150,          # -> FULL_TRUTHFULQA
    ),
    dict(
        name="TruthfulQA (Mistral-7B)",
        benchmark="truthfulqa",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        features="phase10_results/mistral7b_truthfulqa_4stat_full.csv",
        n_questions=FULL_TRUTHFULQA,
    ),
    dict(
        name="TruthfulQA (TinyLlama-1.1B)",
        benchmark="truthfulqa",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        features="phase10_results/tinyllama_truthfulqa_4stat.csv",
        n_questions=150,          # -> FULL_TRUTHFULQA
    ),
]

SETTINGS_BY_NAME = {s["name"]: s for s in SETTINGS}


def slug(name):
    return (name.replace(" ", "_").replace("(", "").replace(")", "")
                .replace(".", "").replace("/", "-"))


# --------------------------------------------------------------------------
# prompt construction -- must match the phase scripts exactly, or the log-prob
# rows will not correspond to the attention rows they are compared against.
# --------------------------------------------------------------------------
def halueval_pair(row):
    """(prefix, grounded_answer, hallucinated_answer) for one HaluEval QA row."""
    prefix = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer: "
    return prefix, str(row["right_answer"]), str(row["hallucinated_answer"])


def truthfulqa_pair(row):
    """(question, best_answer, incorrect_answer) for one TruthfulQA row."""
    incorrect = row["incorrect_answers"]
    bad = incorrect[0] if len(incorrect) > 0 else "I don't know."
    return row["question"], row["best_answer"], bad


def truthfulqa_chat_prefix(tokenizer, question):
    """The phase10 scripts build the prefix with the chat template and
    add_generation_prompt=True; the full sequence appends the answer as an
    assistant turn."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def truthfulqa_chat_full(tokenizer, question, answer):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

# --- Mistral-7B published protocol -------------------------------------------
# phase10_mistral_eval.py does NOT use apply_chat_template. It uses the raw
# Mistral instruct format with NO system message:
#     prompt      = f"[INST] {question} [/INST] {answer}"
#     prompt_only = f"[INST] {question} [/INST] "
# This differs from every other TruthfulQA setting, which uses the chat template
# WITH a "You are a helpful assistant." system turn. The two differ by ~10 tokens,
# which matters a great deal here because candidate chordless 4-cycles scale as
# Theta(N^4). Keep them separate and never mix them in one comparison.
def mistral_inst_full(question, answer):
    return f"[INST] {question} [/INST] {answer}"


def mistral_inst_prefix(question):
    return f"[INST] {question} [/INST] "
