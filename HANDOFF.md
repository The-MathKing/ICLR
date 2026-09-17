# Handoff — "The Topology Is the Sink"

**As of:** 2026-09-17 (RTX 5080 session) · **Deadline:** Saturday 2026-09-19

> The section "Handoff (2026-09-16)" below is the **previous** handoff, kept for history.
> Read this one first.

## THE ONE THING THAT MATTERS

**`sinktda_out/` is gitignored, so the 7–8B features are NOT in this push.** They exist
only on the Windows box, under `ICLR/sinktda_out/`, in these directories:

    triviaqa_mistral   triviaqa_qwen7b   truthfulqa_mistral
    truthfulqa_mistral_chat   truthfulqa_qwen7b
    truthfulqa_tinyllama_cuda   truthfulqa_tinyllama_fp32
    truthfulqa_qwen1.5b_cuda    truthfulqa_qwen1.5b_fp32

That is roughly 350 MB and it is the only copy. **Copy it off that machine before wiping
it.** Everything else — code, result CSVs, docs — is in this push.

## What is in this push

| | |
|---|---|
| 7–8B extractions | Mistral-7B and Qwen2.5-7B, TruthfulQA + on-policy TriviaQA, bf16, ~72 min GPU |
| Results | 28 new CSVs in `sinktda_results/` plus 5 `oof/*.npz`, all per-setting and additive |
| Headline | **18,906,944 head graphs, 0 violations.** Theorem 1: 16 settings, 838,356 layer graphs, 0 violations |
| New code | `sinktda/toha_native.py`, `sinktda/label_audit.py` |
| Bug fixes | double-BOS in extraction (S8), sink-group hardcoding in `report.py` (S11), figure truncation (S12) |

Full detail with evidence and a re-scored review is in `REVIEW_AC.md` §9 ("5080 results",
items S1–S12) and `REVITALIZE_PLAN.md` §8.

## Two blockers, both on you

1. **Llama-3.1-8B never ran.** It is gated and there is no HF token on the box
   (`huggingface-cli login` cannot run non-interactively). Everything else for it is
   ready: the offload path is smoke-tested, and the BOS fix it *needs* — it uses
   `generic_chat`, which double-counted BOS — is in. One block in `sinktda/RUN_ON_5080.md`.
2. **`sinktda_out/` for the 11 small-model settings was never on the box**, so Steps 2–3
   could not be finished. `toha evaluate` and `defect_probe` glob that directory and
   rewrite their CSVs *wholesale*; running them with only the new settings present would
   have replaced 11-setting results with 4-setting ones. **They were deliberately not
   run.** The committed CSVs are verified intact at 11 settings.

## Resume here

1. Get every setting's `sinktda_out/` into one place (copy the new dirs to the Mac, or the
   11 old ones to the PC).
2. In that one place:

       python -m sinktda.evaluate <settings not yet evaluated>
       python -m sinktda.late_fusion && python -m sinktda.defect_probe
       python -m sinktda.toha evaluate
       python -m sinktda.report && python -m sinktda.appendix_extra

3. Then the paper edits, which I deliberately did **not** make: editing prose to use macros
   before `report.py` regenerates `sink_numbers.tex` would leave undefined macros and a
   document that does not compile.
   - abstract and Limitations: replace `1.1B--3.8B` with `\ModelSizeRange{}` (already
     implemented, yields `1.1B--8B`) and drop "7B in the appendix when available";
   - `\MISTRALAPPENDIX` in `paper/sink_results.tex`: rewrite from `\MistralConed{}`,
     `\MistralHOneZero{}`, `\MistralSinkMass{}`. The old "99.99% zeros" audit is
     **verified** (0.999924), and δ₀ = 0 is now directly checked on 98.2% of graphs, which
     the old text said had not been done — that hedge must go;
   - §5.4 and the abstract: add Qwen2.5-7B (S6/S7). Its sink sits on **token 2**, not token
     0, so ρ(d, sink score) is negative there while ρ(d, π̄) stays 0.98+. Theorem 1 is
     untouched; it is Corollary 1's choice of s=0 that is model-specific;
   - the "model without a first-token sink as a contrast" paragraph now has **two** such
     models, and they behave differently: SmolLM cones at no apex at all, Qwen2.5-7B at
     about 10%;
   - report the label-audit disagreement (14% on Mistral). Conclusions are unchanged — all
     four dominance claims survive — but the number belongs in the paper.
4. Compile. **There is no TeX on the Windows box** (no tectonic, pdflatex, xelatex or
   latexmk), so the 9-page check was never done this session.

## Half-finished, deliberately

- **Precision study (Step 5.2).** All four extractions finished
  (`truthfulqa_{tinyllama,qwen1.5b}_{cuda,fp32}`) but were **not evaluated**. Do not simply
  run `evaluate` on them: `report.load()` globs `sinktda_results/auc_*.csv`, so tagged
  settings get swept into the **main** AUC table as extra models. Evaluate them, then move
  their CSVs into `archive/fp32_cuda/sinktda_results/`, mirroring `archive/fp16_backup/`.
  `sinktda/dtype_sensitivity.py` already has a `precision()` function that reads from there
  and is a no-op while they are absent. `_cuda` is bf16 on CUDA and `_fp32` is float32 on
  the same box, so `fp32 − cuda` isolates precision while `cuda − paper` is the platform
  gap; comparing fp32 straight against the paper's MPS numbers would confound the two.
- **TOHA on its own benchmarks (Step 4)** was not run: their pipeline needs a Comet API key
  (a third-party upload), their pre-generated dataset CSVs, and defaults to the gated
  Llama-3.1-8B. Only the implementation-equivalence check was done, and it is clean —
  21,696 head graphs, max difference 1.2e-8 against their own released code.

## Caution when you regenerate

The Windows box has scikit-learn 1.9.1 against the paper's 1.9.0, and torch 2.11.0+cu128
against 2.13.0. Whether the published numbers reproduce bit-for-bit could **not** be
tested, because that test needs the missing `sinktda_out/`. If you regenerate on a
different machine from the one that produced the paper, diff `summary_auc.csv` before
trusting the tables.

---

# Handoff (2026-09-16)

**As of:** 2026-09-16 · **Deadline:** Saturday 2026-09-19 · **Branch:** `main`

State: Phase 1 corrections are **done and verified**. Two experiments are **done**.
One long extraction is **staged but not started**. Nothing is half-finished.

---

## 1. Where things stand

| | Status |
|---|---|
| Manuscript Phase-1 corrections | **done** — 43 edits, structurally verified |
| Mistral precision experiment (item 6) | **done** — result below, already written into the paper |
| BC-interval robustness check (item 3) | **done** — new Appendix, already written in |
| CUDA port of the codebase | **done** — 15 files patched |
| TruthfulQA N-harmonisation (item 7) | **staged, not started** — caches renamed, scripts ready |
| Log-prob baseline at full N (item 1) | script written, not run |
| Lookback Lens baseline (item 8) | script written, not run |
| On-policy benchmark (item 9) | **deliberately deferred** — not feasible before Saturday |

---

## 2. What the review found (all verified against released CSVs)

Three claims in the manuscript were contradicted by the repo's own result files:

1. **`Δ < 0.003` normalisation-invariance claim was false** on 3 of 7 settings, by up to
   `0.0086` (~2.9× the stated bound), and **two TOST verdicts flip**: SmolLM-1.7B *gains*
   equivalence at ε=0.020, TinyLlama-1.1B *loses* it at ε=0.025. Stated in three places.
2. **"TruthfulQA is audited on all six model instances" was false** — it is five.
   Qwen2.5-1.5B is HaluEval-only. `master_pipeline.py` defines exactly 7 settings.
3. **§5.3 contradicted Table 3 on the same page** ("equivalence not confirmed for any
   setting" vs. Mistral-7B showing CONFIRMED at ε=0.015).

Plus: Table 13 was sourced from a superseded 2,000-resample run while its caption claimed
10,000; the Mistral row was silently omitted from four tables (and in one case its omission
is what made the caption's claim true); `Best Non-Topo. Baseline` was not non-topological
(the MST proxy is a 0D subset by Theorem 1); the complexity table's asymptotics contradicted
its own N^6.9 measurement; HaluEval and TruthfulQA were never cited; six labels including
Figure 2 were never `\ref`'d.

---

## 3. Experiments run, and what they showed

### Mistral 1D degeneracy (roadmap item 6) — **resolved**

Three cells, 100 questions each:

| precision | prompt | % 1D cells zero | rows with any 1D | mean N |
|---|---|---|---|---|
| int8 (published) | `[INST]` | 100.00 | 0 % | 32.47 |
| bfloat16 | `[INST]` | 100.00 | 0 % | 32.47 |
| bfloat16 | chat template | 88.41 | 100 % | 42.48 |

- **Quantization is irrelevant.** Native bf16 reproduces int8 exactly. Limitation 1 closed.
- **The paper's length explanation is refuted** by its own data. Sorted by mean N:
  Phi-3 27.8 → 99.7% non-degenerate; SmolLM 32.2 → 100%; **Mistral 33.0 → 0.2%**;
  TinyLlama 41.0 → 100%; Qwen 42.9 → 100%. Mistral is only third-shortest.
- **Attention concentration is the surviving explanation.** `P̄₀/N` rank-orders the 1D
  zero-rate across all five settings (Spearman ρ = 0.90; n=5, p≈0.08 — descriptive, not
  a significance test).
- **The result is prompt-sensitive**, and prompt format is *not* uniform across settings:
  each model uses its own native format, and Qwen2.5-3B alone gets a system turn.

### Bootstrap symmetry (roadmap item 3) — **no action needed**

Computed from the shipped `master_results/bootstrap_diffs_*.npy`. All seven published TOST
p-values **reproduce exactly**. Distributions are near-symmetric (|skew| ≤ 0.078, |z₀| ≤ 0.040)
and **no verdict changes** across 28 setting×margin cells. Caveat: bias-corrected only
(a = 0); full BCa needs a jackknife over raw out-of-fold predictions, which the released
artifacts don't include.

---

## 4. Changes made

### `paper/paper.tex` (backups: `.bak`, `.bak2`)

Phase 1: all three false claims corrected; Table 13 repopulated from the canonical 10k run;
Mistral rows restored to Tables 15 and 18 (with the `fit_alpha_hat` fallback footnoted —
its α̂ = 1.0 is a code default, not a fit); complexity table asymptotics and speedups fixed
(13.2 / 84.8 / 1218.8 / 32583.2, recomputed as `(t_0D+t_1D)/t_MST`); title claim scoped;
HaluEval/TruthfulQA/attention-sink/reproducibility citations added; six orphan cross-references
repaired; Case-2's "96% shrinkage" unit artifact removed.

New tables: `tab:alpha_vs_zn` (Z/N vs α̂), `tab:bca` (bootstrap symmetry),
`tab:degeneracy` (what explains the Mistral degeneracy).

§5.1 gained a **Prompt protocol** paragraph and a fully rewritten Mistral diagnostic.
Limitations 1 and 4 updated.

### Code

- **CUDA port**: 15 scripts were MPS-only with no CUDA path — they'd have run silently on
  CPU. Patched via `rigor/patch_device.py` (undo: `--revert`). Also removed the hardcoded
  `/Volumes/2TB/hf_cache`, and redirected `generate_paper_figures.py` from
  `/Volumes/2TB/iclr/paper/` to `PAPER_DIR` (defaults to `./paper`).
- **`master_pipeline.py`**: two docstrings corrected to match their implementations — the
  TOST rule (uses `tost_p<0.05` alone, i.e. the 90% CI, not the 95% CI its docstring
  claimed) and the permutation null (global shuffle, not the group-preserving one claimed).
  **No behaviour changed; no reported number moves.**
- **Four `.head()` caps raised** to `FULL_TRUTHFULQA = 817` (set back to 150 to reproduce
  the old run).
- **New `rigor/` package** — see `rigor/README.md`.

---

## 5. Known open issues

1. **The permutation null in `predictive_info_gain` is not design-exact.** It shuffles `y`
   globally; the data are paired one-positive-per-group. A pure-Python check showed the
   global shuffle breaks that structure in ~34% of draws even in the minimal 2-group case.
   Fix is written (`rigor/stats_upgrades.within_group_permutation_null`) but **not wired in**.
   Table 18's null is a conservative overfitting band, not a calibrated null. Disclosed in
   the table caption.
2. **`rigor/` scripts are unrun except the Mistral one.** Written without torch/sklearn
   available; treat as reviewed drafts. Only `permute_within_groups` has been logic-tested.
3. **No independent topological baseline.** Both TOHA MTop-Div variants are, by code
   inspection, MST statistics — every topological comparator is in the family Theorem 1
   covers. Now disclosed in Limitation 4.
4. **`refrences/` (sic) ships six third-party conference PDFs** under `accepted/`/`denied/`.
   De-anonymisation and licensing hazard. **Remove before camera-ready.**
5. Python 3.14 + torch 2.11 + transformers 5.x is a very new stack. `torch_dtype` is
   already deprecated in favour of `dtype`.

---

## 6. Resume here

### Immediately: start item 7 (staged, ~1–4 h)

Caches are already renamed to `*.old150`. Just run:

```powershell
python phase10_qwen3b_eval.py; python phase7_truthfulqa_smollm.py; python phase10_phi3_eval.py; python phase10_tinyllama_eval.py
```

Projected power at Δ=0.015 (extrapolating published SEs at 1/√n_groups — projections, not
measurements): Qwen2.5-3B 14.4→~55%, TinyLlama 21.2→~77%, Phi-3 42.7→~99%, SmolLM 53.8→~75%.
Qwen2.5-3B still won't reach 80%; it would need ~1,500 groups and TruthfulQA has 817.

### Then

1. `python master_pipeline.py` — regenerates `master_results/`.
2. **Recompute every affected table** from the new CSVs: Tables 2, 3, 4, 12, 13, 18, 20, 21,
   plus `tab:alpha_vs_zn` and `tab:degeneracy`. Mistral's rows do **not** move (already 817).
3. Update the abstract's power/sample-size claims.
4. `python rigor/logprob_baseline_full.py` (item 1) and
   `python rigor/lookback_lens_baseline.py` (item 8).
5. Compile and proofread. **There is no LaTeX toolchain on this machine** — the manuscript
   is structurally verified but has never been compiled. Do this early enough to fix fallout.

### Verification harness

`verify.py`-style check (refs, envs, braces, `$` parity, tabular column counts, citation
keys, stale-string scan) was run after every edit batch. Re-run it after any manuscript
change. Current state: 21 tables, 0 unresolved refs, 0 column mismatches, 0 missing keys.
