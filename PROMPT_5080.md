# Task: finish the remaining work on the ICLR paper "The Topology Is the Sink" (deadline 2026-09-19)

You are a co-author on an ICLR submission and are running on the author's Windows machine: RTX 5080 (16 GB VRAM), plenty of system RAM. You start with no context. The repository is this checkout (`git pull` first). Another Claude session on a 16 GB Mac did everything except runs that need a GPU; your job is to do those and to fold the results into the paper.

## Read first (in this order)
1. `REVIEW_AC.md`: especially §9 "Revitalization" (current scores, what "Contribution 3" depends on, and the author's to-do list).
2. `REVITALIZE_PLAN.md` §7 (what worked and what failed).
3. `README.md`, `sinktda/RUN_ON_5080.md`.
4. `paper/paper.tex`, `paper/sink_results.tex`, `paper/appendix_sink.tex`.

## What the paper now claims
- **Theorem 1 + Corollary 1:** Vietoris–Rips barcodes of attention graphs are within a computable "coning defect" of a star diagram read off the first-token (sink) column.
- **Proposition 1 (headline):** TOHA's (Bazarova et al., ACL 2026) per-head topological divergence equals the mean over response tokens of 1 − (largest prompt attention) whenever a prompt-level defect δ_P = 0, and is within δ_P otherwise.
  - Verified on 12.3M head graphs from 11 settings, all ≤3.8B parameters, with 0 violations.
  - TOHA adds ≤0.007 AUC over that first-order statistic.
- **The weakness reviewers will hit:** no model ≥7B, and TOHA is not tested on its own benchmarks. Fixing that is your main job.

## Hard rules
- **Precision:** use bfloat16 always. Llama-3.1-8B and Qwen2.5-7B do not fit in 16 GB at bf16, so set `SINKTDA_OFFLOAD=1`: layers are offloaded to the CPU, and the dtype stays bf16. Never quantize for the main results.
- **Numbers:**
  - Never invent a number, result or citation.
  - Every number in the paper text must come from a macro that `sinktda/report.py` generates from a CSV. Add macros to `write_numbers()` / `numbers_toha()` rather than typing numbers.
  - If something fails or contradicts the paper, report it and change the text; do not spin it.
  - Re-check every "all / every / no / nothing" against the regenerated tables.
- **Git:** do not commit or push. Leave changes in the working tree and list them at the end.
- **Jobs:** run one GPU job at a time. The `*.sh` runners are macOS/bash-specific (they use `memory_pressure`), so run the equivalent commands sequentially in PowerShell instead.
- **Environment:** set `HF_HOME` to a disk with ≥60 GB free. Llama-3.1-8B is gated: run `huggingface-cli login` (ask the author if there is no token).
- **transformers 5 attention masks:** custom attention implementations may receive no additive mask. `sinktda/toha.py` applies the causal mask itself and aborts on non-causal attention. Do not remove that guard.

## Setup
```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 torch   # Blackwell needs cu128
pip install transformers accelerate datasets ripser scikit-learn scipy pandas pyarrow joblib matplotlib
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- Library versions used so far: transformers 5.16.1, ripser 0.6.15, scikit-learn 1.9.0. Note any version you differ on.
- Confirm that `sinktda_out/` contains the 11 existing settings (the author copies it by hand; it is gitignored). If it is missing, still do Step 1, and tell the author that Steps 2–3 must run where the full `sinktda_out/` lives.

## Step 0: smoke test (10 min)
```powershell
$env:SINKTDA_OUT = ".tmp_smoke"
python -m sinktda.toha extract --bench truthfulqa --model mistral --n 3
python -m sinktda.extract --bench truthfulqa --model mistral --n 3 --workers 2 --limit 6
$env:SINKTDA_OFFLOAD = "1"; $env:SINKTDA_GPU_MEM = "13GiB"
python -m sinktda.toha extract --bench truthfulqa --model llama8b --n 3
Remove-Item Env:SINKTDA_OUT; Remove-Item Env:SINKTDA_OFFLOAD
```
- Check that no errors or non-finite logits appear, and that `toha.npz` arrays have shape (rows, 32, 32).
- If the offloaded run is extremely slow, raise `SINKTDA_GPU_MEM` (14GiB); do not change the dtype.
- Delete `.tmp_smoke` afterwards.

## Step 1: 7–8B extractions (the priority; hours of GPU time)
Follow `sinktda/RUN_ON_5080.md` exactly. The order is highest value first, and each command skips existing outputs, so the list is safe to restart.
1. **Mistral-7B** (no offload):
   - `toha extract` on TruthfulQA;
   - `extract` on TriviaQA `--n 2000 --workers 4` (this generates the on-policy answers), then `toha extract` on TriviaQA;
   - full `extract` on TruthfulQA, with the default `[INST]` template and with `--template generic_chat --tag chat`.
2. **Llama-3.1-8B, then Qwen2.5-7B** (`SINKTDA_OFFLOAD=1`): same pattern, TruthfulQA and TriviaQA.

Details:
- `toha evaluate` requires each `toha.npz` to align row by row with that setting's `layers.parquet`, so every TOHA setting needs its full `extract` too.
- Log each run to `logs\<setting>.log`.
- Record the wall-clock time and the TriviaQA accuracy printed by `[gen] accuracy`.
- If the on-policy accuracy is <5% or >95%, or the extractor refuses because of degenerate generations, stop and report. Do not paper over it.

## Step 2: evaluation (CPU)
```powershell
$env:SINKTDA_JOBS = "4"
python -m sinktda.evaluate truthfulqa_mistral truthfulqa_mistral_chat triviaqa_mistral truthfulqa_llama8b triviaqa_llama8b truthfulqa_qwen7b triviaqa_qwen7b
python -m sinktda.late_fusion
python -m sinktda.defect_probe
python -m sinktda.toha evaluate
python -m sinktda.report
python -m sinktda.appendix_extra
```
- Only list settings that exist.
- `toha evaluate` with no arguments evaluates every unmodified setting; this is intended.
- `report.py` already knows the new setting names.
- Then sanity-check `sinktda_results/summary_theory.csv` (0 bound violations expected) and `toha_checks.csv` (0 violations expected).
  - **Any violation is a bug or a numerical issue: investigate before writing.** Check the tolerance: ripser uses float32.

## Step 3: update the paper (keep the main text ≤ 9 pages)
- **Abstract, §4, Limitations:** replace "1.1B–3.8B" and "7B in the appendix when available" with what was actually run. Macro-ize model counts and size ranges if they change.
- **Mistral appendix:** rewrite `\MISTRALAPPENDIX` in `paper/sink_results.tex` from the new data. It currently cites an old unverified "99.99% zeros" audit: verify or remove that claim using `theory_mistral` numbers.
- **Consistency across scale:**
  - Check whether the 7–8B results agree with the small-model conclusions: coning share, 1D emptiness, TOHA ≈ π̄-score, and TOHA adding ≈0 over π̄.
  - If they differ, say so in §5.4 and in the abstract. A scale-dependent result is still a result.
  - Consider one sentence in §5.4 on scale, with numbers from macros.
- **Compile.** tectonic may not be installed here. If it isn't, use any TeX distribution with the same source; keep `\usepackage[T1]{fontenc}` before `times`. Tectonic command:
  `cd paper; tectonic -X compile paper.tex --outdir ..\.tmp\build`
- **After compiling, check:**
  - no undefined references and no overfull boxes;
  - the reproducibility statement starts on page 10, i.e. Limitations ends on page 9. If it doesn't, move material to the appendix; do not shrink fonts or margins.
  - Render each main-text page to an image and look at it. Table 4 (TOHA) will get wider rows; keep it legible.

## Step 4 (high value if time allows): TOHA on its own benchmark
Reviewers will ask whether the reduction holds where TOHA claims state of the art.
1. Clone the authors' code: https://github.com/sb-ai-lab/TOHA.
2. Run their pipeline for Mistral-7B (and Llama-3.1-8B if feasible) on one of their datasets (CoQA or RAGTruth/MS MARCO) with their prompts, labels and probe-set protocol.
3. Add a first-order variant: per head, the mean over response tokens of 1 − max prompt attention (and 1 − attention to token 0). Use their head-selection and evaluation code unchanged.
4. Report, per dataset and model:
   - their TOHA AUC (compare with their Table 1; note any gap);
   - the first-order variant's AUC;
   - the fraction of head graphs with δ_P = 0, where δ_P = max over response pairs w<u of [A_uw − min(π_u, π_w)]⁺ (`sinktda.toha.toha_head_features` computes all of this from response-row attention).

Write the results to `sinktda_results/toha_native.csv`, add a small appendix table plus one main-text sentence via macros, and put the script in `sinktda/toha_native.py`. If their code does not run, or the numbers are far from their paper, stop and report rather than improvising a different protocol.

## Step 5 (optional, in this order)
1. **Label audit.** Sample 200 rows per model from `sinktda_out/triviaqa_*/generations.csv` (seed 0) and judge them with Qwen2.5-7B-Instruct as an LLM judge (question, gold aliases, answer → correct/incorrect). Report disagreement with the string-match label per model in the appendix, and state that the judge is itself a model. If disagreement exceeds ~10%, re-evaluate TriviaQA with judge labels on the audited subset and report whether conclusions change.
2. **Precision study.** fp32 re-extraction of `truthfulqa_tinyllama` and `truthfulqa_qwen1.5b` (use `--dtype float32` with a `--tag fp32`). Extend `sinktda/dtype_sensitivity.py` to a third column.

## Step 6: finish and report back
- **Append** a short "5080 results" subsection to `REVIEW_AC.md` §9, with re-scored Soundness / Presentation / Contribution / Overall / Confidence and one-line justifications. Be a harsh reviewer.
- **Update** `README.md` (models, commands, versions) and `REVITALIZE_PLAN.md` §7.
- **Give the author a summary covering:**
  - what ran, and the wall-clock times;
  - the key numbers at 7–8B, compared with the small models;
  - whether the headline survives at scale;
  - every file changed or created;
  - what is still left.
- **Anonymous mirror:** do not upload anything. If asked to prepare it, create a local export folder that excludes `refrences/`, `logs/`, `archive/`, `.tmp*/`, `HANDOFF.md`, `PROMPT_*.md`, `REVIEW_AC.md` and `REVITALIZE_PLAN.md`, then grep it for the author's name, email and absolute paths.
- **Still manual for the author:** confirm the ICLR style file, and check venues for barbero2025first, kostenok2023, perez2022topological and gardinazzi2024persistent. You may verify these on the web and fix `paper/paper.bib` if you can confirm them.
