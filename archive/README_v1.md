# Attention TDA Is a Minimum Spanning Tree
## An Exact, Faster Replacement for Persistent Homology in LLM Hallucination Detection

This repository contains the code, extracted feature caches, and evaluation pipelines for reproducing all experiments, tables, and figures in the paper:
> **Attention TDA Is a Minimum Spanning Tree: An Exact, Three-Orders-of-Magnitude-Faster Replacement for Persistent Homology in LLM Hallucination Detection** (ICLR Submission).

---

## 1. Overview & Key Findings

1. **0D Persistence Is Exactly the MST (Theorem 1):** 0-dimensional Vietoris–Rips persistent homology on symmetrized attention distance graphs $D = 1 - \max(A, A^T)$ is exactly Kruskal's Minimum Spanning Tree (MST) under single-linkage clustering (Gower & Ross, 1969) — not approximately, the same number. We ship **MST-TDA**, a drop-in replacement (`verify_mst_equivalence.py`) that is $>\!1{,}200\times$ cheaper to compute than Ripser and, extended per-head, more accurate (AUC $0.892$ vs.\ $0.677$ head-averaged; `per_head_tda_ablation.py`).
2. **The Sequence Length Confound (1D persistence):** 1D cyclic persistence scales mechanically with token length (Proposition 2); its length-normalized incremental value beyond 0D is small but real on HaluEval (+0.010–0.016 AUC, CI excludes zero) and statistically indistinguishable from zero on TruthfulQA — a cost-benefit failure against its $>\!1{,}200\times$ compute premium, not a bare null result. See `master_pipeline.py`, the single canonical evaluation script for every AUC/TOST/power number in the paper.
3. **A Second, Independent Confound — Surface Form in HaluEval QA:** Beyond sequence length, HaluEval QA's grounded/hallucinated answers are separable by answer character length or bag-of-words alone (AUC 0.96–0.98), and this survives the benchmark's standard token-length matching (`compute_lexical_confound.py`). This is consistent with concurrent, independent benchmark-artifact audits (Hussain & Kantarcioglu, 2026 "PARALLAX"; Janiak et al., 2025 "The Illusion of Progress") and applies to any probe evaluated on this benchmark, not only topological ones.
4. **Grouped Cross-Validation:** Paired grounded/hallucinated answers from the same question must stay in the same fold; `StratifiedGroupKFold` prevents question-level leakage.
5. **Multi-Model Audit:** Evaluated across `Qwen2.5-1.5B`, `Qwen2.5-3B`, `SmolLM-1.7B`, `Phi-3-mini-3.8B`, `Mistral-7B-Instruct-v0.2` (int8 CPU quantized), and `TinyLlama-1.1B-Chat-v1.0` (a genuine Llama-family model, `LlamaForCausalLM`) on `HaluEval QA` and `TruthfulQA` (Mistral-7B and TinyLlama on TruthfulQA only), all under one canonical scaled logistic-regression pipeline.

**Note on a prior internal inconsistency:** an earlier revision of this codebase fit logistic regression on *unscaled* persistence features in some evaluation scripts (`compute_equivalence_tests.py`, `run_mega_bootstrap_10k.py`) and *scaled* features in others (`phase1_stats_improvements.py`). Unscaled fitting silently suppresses the numerically smaller length-normalized 1D features under L2 regularization relative to 0D, making the two feature banks look nearly identical and producing an artificially tight, near-zero ΔAUC with implausibly narrow bootstrap confidence intervals — this was the source of a "Δ = +0.000" claim in an earlier draft of the paper that contradicted this repository's own correctly-scaled result files. `master_pipeline.py` is now the single, canonical, scaled-pipeline source for every number in the paper; the two inconsistent scripts above are retained for provenance but are no longer used to generate paper results.

---

## 2. Quickstart & Installation

### Environment Setup
Create a Python virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Reproducing Paper Results

All extracted feature banks across models and datasets are cached in this repository. You do not need to rerun multi-hour LLM forward passes to replicate the results.

### Reproduce All Tables (Single Command)
```bash
python master_pipeline.py            # canonical AUC / TOST / power / scaling-exponent / info-gain tables
python compute_lexical_confound.py   # surface-form confound analysis (Table `tab:lexical`)
python reproduce_paper_tables.py     # legacy table formatting (some tables predate the canonical pipeline; see script docstrings and the paper's appendix caveats for which ones)
```

### Script Directory Reference
* `master_pipeline.py`: **The canonical evaluation pipeline.** StandardScaler + L2 logistic regression under grouped 10-fold CV, used for every AUC, TOST equivalence test, power/MDE, and leakage-free α̂-calibration number in the paper. Outputs to `master_results/`.
* `compute_lexical_confound.py`: Quantifies the surface-form/lexical confound in HaluEval QA and TruthfulQA (answer-text-only TF-IDF and character-length classifiers).
* `verify_mst_equivalence.py`: Implementation-level check that MST-TDA's Kruskal weight and Ripser's 0D total persistence agree to floating-point precision on real attention graphs, plus wall-clock comparison.
* `per_head_tda_ablation.py`: Per-head vs. head-averaged 0D/1D comparison (Appendix, Head-Averaging Ablation).
* `reproduce_paper_tables.py`, `run_comprehensive_evals.py`, `compute_equivalence_tests.py`, `run_mega_bootstrap_10k.py`, `phase1_stats_improvements.py`: Earlier-stage scripts, some of which use an inconsistent (unscaled) evaluation pipeline superseded by `master_pipeline.py` (see the note in §1 above) — retained for provenance, not for generating headline paper numbers.
* `phase1_controls.py`: Phase 1 diagnostic extraction script for non-topological statistics, MST weights, and length controls.
* `phase8_length_normalization.py`: Analysis of 1D length scaling, correlation scatter plots, and length normalization.
* `phase9_toha_baseline.py`: TOHA (MTop-Div) baseline re-implemented as a single summed scalar per the paper's own protocol (HaluEval QA, TruthfulQA/SmolLM-1.7B) — not a reproduction of TOHA's originally reported numbers on their own benchmarks; see the paper's fairness caveat in §6.1.
* `phase10_qwen3b_eval.py`, `phase10_phi3_eval.py`, `phase10_mistral_eval.py`: Per-model TruthfulQA extraction and evaluation, using the full four-statistic descriptor bank (max lifetime, total persistence, max birth, max death). `phase10_mistral_eval.py` targets a 7B model via int8 CPU quantization (16GB-RAM hardware cannot hold a 14GB fp16 model); its output (`phase10_results/mistral7b_truthfulqa_4stat_full.csv`, $N{=}817$ questions) is included in the paper (Tables 2–3) and is the largest-scale setting audited.
* `phase11_significance_tests.py`: Paired cluster-bootstrap hypothesis testing with Holm-Bonferroni correction.

---

## 4. Dataset & Feature Schema

* `phase3_results/train_features.csv`: Full feature bank for HaluEval QA ($N=4000$) using `Qwen2.5-3B-Instruct` (0D persistence, 1D persistence, layerwise statistics).
* `phase10_results/qwen3b_truthfulqa_4stat.csv`: Feature bank for TruthfulQA ($N=300$) using `Qwen2.5-3B-Instruct`, full four-statistic descriptor bank (used by the reproduction pipeline).
* `phase7_results/truthfulqa_smollm_features.csv`: Feature bank for TruthfulQA ($N=1000$) using `SmolLM-1.7B-Instruct`.
* `phase10_results/phi3_truthfulqa_4stat.csv`: Feature bank for TruthfulQA ($N=300$) using `microsoft/Phi-3-mini-4k-instruct`, full four-statistic descriptor bank (used by the reproduction pipeline).
* `phase1_controls_halueval.csv`: Per-layer non-topological attention statistics (mean, max, sink mass) and MST weights ($N=1000$).

---

## 5. License & Anonymity
This repository is formatted for double-blind peer review. Code and artifacts are hosted anonymously.
