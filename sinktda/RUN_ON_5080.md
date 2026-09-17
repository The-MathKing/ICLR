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
