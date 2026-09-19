#!/bin/bash
# TOHA reduction (Proposition 3): per-head MTop-Div and its first-order counterparts for all
# 11 settings, then evaluation. One heavy job at a time; no ripser, so each pass is short.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false TMPDIR=${TMPDIR:-$PWD/.tmp} JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-$PWD/.tmp}
export SINKTDA_JOBS=2 OMP_NUM_THREADS=4
PY=${PY:-.venv/bin/python}

# The prompt/response boundary is only correct when the prefix and the full string are
# tokenized the same way; a chat template that emits BOS silently shifted it once.
$PY -m sinktda.check_tokenization || { echo '[abort] tokenization check failed'; exit 1; }
MINFREE=${MINFREE:-35}
waitmem() {
  while true; do
    f=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
    [ "${f:-0}" -ge "$MINFREE" ] && break
    echo "[mem] ${f}% free, waiting"; sleep 20
  done
}
tx() { waitmem; echo "=== toha $* ($(date +%H:%M:%S))"; $PY -m sinktda.toha extract "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback"; }

tx --bench truthfulqa --model qwen1.5b
tx --bench truthfulqa --model qwen3b
tx --bench truthfulqa --model phi3
tx --bench truthfulqa --model tinyllama
tx --bench truthfulqa --model smollm
tx --bench halueval --model qwen1.5b --n 500
tx --bench halueval --model qwen3b --n 1000
tx --bench triviaqa --model qwen1.5b --n 2000
tx --bench triviaqa --model qwen3b --n 2000
tx --bench triviaqa --model phi3 --n 2000
tx --bench triviaqa --model tinyllama --n 2000
waitmem; echo "=== toha evaluate ($(date +%H:%M:%S))"
$PY -m sinktda.toha evaluate truthfulqa_qwen3b truthfulqa_qwen1.5b truthfulqa_phi3 truthfulqa_tinyllama truthfulqa_smollm \
    halueval_qwen3b halueval_qwen1.5b triviaqa_qwen3b triviaqa_qwen1.5b triviaqa_phi3 triviaqa_tinyllama
echo "=== TOHA DONE ($(date +%H:%M:%S))"
