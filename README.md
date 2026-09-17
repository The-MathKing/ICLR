# The Topology Is the Sink — code and data

Code, per-example features, and results for *"The Topology Is the Sink: Topological Hallucination Detectors Reduce to First-Order Attention Statistics"* (ICLR submission).

Everything in the paper comes from the `sinktda/` package. The top-level `phase*`, `master_pipeline.py`, `rigor/`, and `archive/` files belong to an earlier, superseded version of the study and are **not** used for any number in the current paper. The one exception is `rigor/verify_directed_collapse.py`, which checks the directed-flag proposition (Appendix A).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets ripser scikit-learn scipy numpy pandas pyarrow joblib matplotlib
```

Versions used for the paper: torch 2.13.0, transformers 5.16.1, ripser 0.6.15, scikit-learn 1.9.0, numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, Python 3.13. The models were run in bfloat16 on Apple-silicon MPS with eager attention.

## Pipeline

| step | command | output |
|---|---|---|
| features (one forward pass per example) | `python -m sinktda.extract --bench {truthfulqa,halueval,triviaqa} --model {qwen3b,qwen1.5b,phi3,tinyllama,smollm,mistral} [--n N]` | `sinktda_out/<bench>_<model>/` |
| evaluation (AUCs, bootstrap/TOST, theory checks) | `python -m sinktda.evaluate [setting ...]` | `sinktda_results/{auc,comp,theory,theory_layers}_*.csv`, `oof/*.npz` |
| late-fusion incremental tests | `python -m sinktda.late_fusion` | `sinktda_results/late_fusion.csv` |
| coning defect as a detector | `python -m sinktda.defect_probe` | `sinktda_results/defect_probe.csv` |
| TOHA reduction (Prop. 1): per-head MTop-Div and first-order counterparts | `python -m sinktda.toha extract --bench ... --model ... [--n N]` then `python -m sinktda.toha evaluate` | `sinktda_out/<setting>/toha.npz`, `sinktda_results/toha_{checks,auc,comp}.csv` |
| causal sink-bias probe | `bash sinktda/run_toha_causal.sh` | `sinktda_results/toha_*_causal.csv` |
| numerical checks of Prop. 1 and Cor. 2 | `python -m sinktda.check_theory` | `sinktda_results/check_theory.csv` |
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

The feature caches in `sinktda_out/` are enough to regenerate every table and figure without running a model:

```bash
python -m sinktda.evaluate && python -m sinktda.late_fusion && python -m sinktda.layer_split
python -m sinktda.defect_probe && python -m sinktda.toha evaluate && python -m sinktda.check_theory
python -m sinktda.synthetic && python -m sinktda.report && python -m sinktda.appendix_extra
```
