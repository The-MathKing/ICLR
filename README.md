# The Topology Is the Sink — code and data

Code, per-example features, and results for *"The Topology Is the Sink: Topological Hallucination Detectors Reduce to First-Order Attention Statistics"* (ICLR submission).

Everything in the paper comes from the `sinktda/` package. `rigor/verify_directed_collapse.py` checks the directed-flag proposition (Appendix A). An earlier, superseded version of this study used a separate set of scripts; none of them produces any number in the current paper, so they are not part of this release.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets ripser scikit-learn scipy numpy pandas pyarrow joblib matplotlib
```

Versions used for the paper: torch 2.13.0, transformers 5.16.1, ripser 0.6.15, scikit-learn 1.9.0, numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, Python 3.13. The models of at most 3.8B parameters were run in bfloat16 on Apple-silicon MPS with eager attention; the 7B models (Mistral-7B and Qwen2.5-7B) were run in bfloat16 on an NVIDIA RTX 5080. Nothing was quantized.

The 7--8B settings were extracted on an RTX 5080 (16 GB, CUDA) with torch 2.11.0+cu128, transformers 5.17.0, ripser 0.6.15, scikit-learn 1.9.1, numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, Python 3.14.7, also in bfloat16 with eager attention. Qwen2.5-7B uses `SINKTDA_OFFLOAD=1` (some layers on the CPU, dtype unchanged); Mistral-7B fits on the GPU. Nothing is quantized.

## Pipeline

| step | command | output |
|---|---|---|
| features (one forward pass per example) | `python -m sinktda.extract --bench {truthfulqa,halueval,triviaqa} --model {qwen3b,qwen1.5b,phi3,tinyllama,smollm,mistral,qwen7b,llama8b} [--n N]` | `sinktda_out/<bench>_<model>/` |
| evaluation (AUCs, bootstrap/TOST, theory checks) | `python -m sinktda.evaluate [setting ...]` | `sinktda_results/{auc,comp,theory,theory_layers}_*.csv`, `oof/*.npz` |
| late-fusion incremental tests | `python -m sinktda.late_fusion` | `sinktda_results/late_fusion.csv` |
| coning defect as a detector | `python -m sinktda.defect_probe` | `sinktda_results/defect_probe.csv` |
| TOHA reduction (Prop. 1): per-head MTop-Div and first-order counterparts | `python -m sinktda.toha extract --bench ... --model ... [--n N]` then `python -m sinktda.toha evaluate` | `sinktda_out/<setting>/toha.npz`, `sinktda_results/toha_{checks,auc,comp}.csv` |
| causal sink-bias probe | `bash sinktda/run_toha_causal.sh` | `sinktda_results/toha_*_causal.csv` |
| prompt-boundary tokenization check (no GPU) | `python -m sinktda.check_tokenization` | pass/fail per model and template |
| re-check the paper's claims against the regenerated CSVs | `python -m sinktda.check_claims` | pass/fail per claim (run after `report`) |
| numerical checks of Prop. 1 and Cor. 2 | `python -m sinktda.check_theory` | `sinktda_results/check_theory.csv` |
| TOHA vs the authors' released MTop-Div code | `python -m sinktda.toha_native --model mistral --n 8` | `sinktda_results/toha_native.csv` |
| LLM-judge audit of the on-policy labels | `python -m sinktda.label_audit --n 200` (re-run just the sensitivity table after a re-extraction with `--sensitivity-only`) | `sinktda_results/label_audit{,_sensitivity}.csv` |
| fp16 vs bf16 sensitivity | `python -m sinktda.dtype_sensitivity` | `sinktda_results/dtype_sensitivity.csv`, `paper/sink_dtype.tex` |
| label-free layer split | `python -m sinktda.layer_split` | `sinktda_results/layer_split.csv` |
| synthetic checks (Prop. 1, dose-response, planted cycle) | `python -m sinktda.synthetic` | `sinktda_results/synthetic_*.csv` |
| timing | `python -m sinktda.timing --model tinyllama` | `sinktda_results/timing.csv` |
| paper tables, figures, number macros | `python -m sinktda.report && python -m sinktda.appendix_extra [--examples]` | `paper/sink_*.tex`, `paper/fig_sink_*.pdf` |

The exact settings are listed in `sinktda/run_all.sh`, `sinktda/run_seq.sh`, `sinktda/run_fix.sh` and `sinktda/run_toha.sh`. The runners execute one heavy job at a time, which is needed on a 16 GB machine:

- TruthfulQA: all 817 questions.
- HaluEval QA: `--n 1000` for Qwen2.5-3B and `--n 500` for Qwen2.5-1.5B.
- TriviaQA on-policy: `--n 2000`.

Mistral-7B, Qwen2.5-7B and Llama-3.1-8B need a 16 GB GPU (the latter two with CPU offload); see `sinktda/RUN_ON_5080.md`.

## Prompt tokenization

Feature extraction tokenizes through `sinktda.data.encode`, which suppresses the tokenizer's
automatic BOS when the prompt string already begins with one. Some chat templates
(Mistral-7B-Instruct, Llama-3.1) emit BOS themselves, so calling the tokenizer with the default
`add_special_tokens=True` would prepend a second one. A repeated BOS splits the attention sink
across two tokens and changes every sink and coning statistic: on `triviaqa_mistral` it moved the
exactly-prompt-coned share from 0.75 to 0.69, the share of head graphs whose top prompt token is
the sink from 0.68 to 0.18, and the median per-head rank correlation with the sink score from 0.97
to 0.81. `encode` is a no-op for every other template used here (Qwen has no BOS; Phi-3 adds none;
TinyLlama and Mistral's manual `[INST]` string add exactly one), so no result for a model of at
most 3.8B parameters is affected.

## Seeds and protocol

| item | value |
|---|---|
| TriviaQA question sample | seed 0 |
| outer CV | `StratifiedGroupKFold(10)` grouped by question, seeds 42/43/44 |
| inner C selection | 3-fold grouped CV |
| bootstrap | 10,000 question-level resamples, seed 0 |
| equivalence margin | ±0.015 AUC |

Prompts are defined in `sinktda/data.py`.

## Reproducing from the released features

Only part of `sinktda_out/` is released. The per-head (`perhead.npz`, `toha.npz`) and
hidden-state (`hidden.npy`) dumps are all larger than the 8 MB per-file limit of the
anonymous host, and the dumps for the other settings were lost with the machine that
produced them. What ships is every file that clears the limit:

| file | settings |
| --- | --- |
| `layers.parquet` (per-layer features) | `truthfulqa_{mistral,mistral_chat,qwen7b}`, `triviaqa_qwen7b` |
| `generations.csv` (on-policy answers and labels) | `triviaqa_{mistral,qwen7b}` |

These are enough to re-derive the per-layer results for those settings -- the coning
shares and `P_0`/star-weight correlations in Table 1, and the layer-level banks -- and to
audit the on-policy labels directly. They are **not** enough to rebuild the per-head
tables, the TOHA results, or the hidden-state probe; those banks need the dumps that
could not ship.

The numbers themselves do not depend on this. Every table, figure and macro in the paper
is generated from the CSVs in `sinktda_results/`, which are released in full:

```bash
python -m sinktda.report
```

To regenerate the feature dumps for any setting and reproduce the whole chain from
scratch, re-extract first. The model and setting identifiers are in the paper's
implementation appendix:

```bash
python -m sinktda.extract --bench triviaqa --model qwen7b --n 2000
python -m sinktda.toha extract --bench triviaqa --model qwen7b --n 2000
python -m sinktda.evaluate triviaqa_qwen7b
```
