# 7B–8B models on the RTX 5080 box (16 GB)

The Mac has 16 GB of unified memory, so it cannot hold 7B models in bf16. Everything below runs in **bfloat16**, like the rest of the study.

| model | key | bf16 weights | how |
|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | `mistral` | ~14.5 GiB | fits on the GPU |
| Qwen2.5-7B-Instruct | `qwen7b` | ~15.2 GiB | `SINKTDA_OFFLOAD=1` (some layers are placed on the CPU; still bf16) |
| Llama-3.1-8B-Instruct | `llama8b` | ~16.1 GiB | `SINKTDA_OFFLOAD=1`; gated model, so run `huggingface-cli login` first |

## Setup

```powershell
# torch >= 2.7 built for cu128 is required for Blackwell (sm_120)
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install transformers accelerate datasets ripser scikit-learn pandas pyarrow scipy joblib
git pull
```

## Runs

Order: highest value first. Every command skips outputs that already exist.

```powershell
$env:SINKTDA_OFFLOAD = "0"
# 1. TOHA reduction (Proposition 3): light, per-head only, no ripser (~10-30 min each)
python -m sinktda.toha extract --bench truthfulqa --model mistral
python -m sinktda.extract    --bench triviaqa   --model mistral --n 2000 --workers 4   # generates answers + full features
python -m sinktda.toha extract --bench triviaqa   --model mistral --n 2000

# 2. full feature extraction (Theorem 1 checks, detection banks)
python -m sinktda.extract --bench truthfulqa --model mistral --workers 4
python -m sinktda.extract --bench truthfulqa --model mistral --template generic_chat --tag chat --workers 4

# 3. the two larger models, with CPU offload (slower; keep --workers low: RAM)
$env:SINKTDA_OFFLOAD = "1"; $env:SINKTDA_GPU_MEM = "13GiB"
python -m sinktda.toha extract --bench truthfulqa --model llama8b
python -m sinktda.toha extract --bench truthfulqa --model qwen7b
python -m sinktda.extract --bench triviaqa --model llama8b --n 2000 --workers 4
python -m sinktda.toha extract --bench triviaqa --model llama8b --n 2000
python -m sinktda.extract --bench truthfulqa --model llama8b --workers 4
python -m sinktda.extract --bench triviaqa --model qwen7b --n 2000 --workers 4
python -m sinktda.toha extract --bench triviaqa --model qwen7b --n 2000
python -m sinktda.extract --bench truthfulqa --model qwen7b --workers 4
```

`toha evaluate` checks that the rows of `toha.npz` align with `layers.parquet`, so each TOHA setting needs the matching `extract` output.

## Back on the Mac

Copy `sinktda_out/{truthfulqa,triviaqa}_{mistral,mistral_chat,llama8b,qwen7b}/` back. Per-head arrays for 32×32 heads are ~40 MB compressed. Then run, one at a time:

```bash
S="truthfulqa_mistral truthfulqa_mistral_chat truthfulqa_llama8b truthfulqa_qwen7b triviaqa_mistral triviaqa_llama8b triviaqa_qwen7b"
python -m sinktda.evaluate $S            # list only settings that were copied back
python -m sinktda.late_fusion && python -m sinktda.defect_probe
python -m sinktda.toha evaluate          # all settings with toha.npz
python -m sinktda.report && python -m sinktda.appendix_extra
```

`report.py` already knows these setting names; the tables and macros pick them up automatically.

Once the results exist, remove "(pending)" from the Mistral appendix and update the model-size range in the abstract and limitations.

## What actually happened on the 5080 (2026-09-17)

Environment: Python 3.14.7, torch 2.11.0+cu128, transformers 5.17.0, ripser 0.6.15,
scikit-learn 1.9.1. `accelerate` must be installed for `SINKTDA_OFFLOAD=1`. Set
`HF_HOME` to a disk with >=60 GB free. The `*.sh` runners are macOS-specific; run the
python commands directly (PowerShell is fine, one job at a time).

Timings (bf16, nothing quantized):

| block | wall clock |
|---|---|
| Mistral-7B, 5 runs (no offload) | 21 min |
| Qwen2.5-7B, 4 runs (offload, 13GiB) | 30 min |

On-policy accuracy: Mistral-7B 0.665, Qwen2.5-7B 0.514.

Three things that bite:

1. **`--workers 4` runs the host out of RAM on the offloaded TriviaQA extraction.** The
   offloaded 7.6B model holds ~15 GB of the 31 GB host, and TriviaQA sequences are about
   twice as long as TruthfulQA's, so the per-head arrays are much larger. It dies with a
   `numpy ArrayMemoryError` on a *small* allocation. Use `--workers 2` for
   `extract --bench triviaqa` under `SINKTDA_OFFLOAD=1`; TruthfulQA is fine at 4.
2. **Do not run `toha evaluate` or `defect_probe` unless every setting's `sinktda_out/`
   directory is present.** Both glob `sinktda_out/` and rewrite
   `toha_{checks,auc,comp}.csv` and `defect_probe.csv` from scratch, so running them with
   only the new settings present silently replaces the full results with a subset.
   `evaluate` is safe: it writes one file per setting.
3. **Chat templates that emit BOS.** Mistral and Llama-3.1 `apply_chat_template` already
   include the BOS token; extraction now goes through `sinktda.data.encode`, which
   suppresses the tokenizer's automatic one in that case. Do not revert this -- a doubled
   BOS splits the attention sink across two tokens and changes every coning statistic.

Llama-3.1-8B is gated: `huggingface-cli login` must be run interactively before the
`llama8b` block will download.

## If you add the fp32 precision study

Do **not** extract it into `sinktda_out/` and evaluate into `sinktda_results/`.
`report.load()` globs `sinktda_results/auc_*.csv`, so a `--tag fp32` setting is swept
into the main AUC and comparison tables as an extra model. The existing fp16 study avoids
this by keeping its results in `archive/fp16_backup/sinktda_results/`; mirror that layout
(e.g. `archive/fp32_cuda/sinktda_results/`) and extend `dtype_sensitivity.py` to read the
third directory.

Also note that the committed bf16 numbers were produced on Apple-silicon MPS with
scikit-learn 1.9.0. Comparing a CUDA fp32 extraction against them conflates precision with
platform and library version. To isolate precision, re-extract bf16 on the same box under
its own tag and compare fp32 against *that*; the MPS-vs-CUDA bf16 difference is then a
separate, and independently useful, reproducibility number.
