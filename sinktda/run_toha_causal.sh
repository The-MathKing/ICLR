#!/bin/bash
# Causal probe: shift the first-token logit by b in every attention head (teacher-forced
# TruthfulQA, labels unchanged) and re-measure how close TOHA is to its sink counterpart.
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

for m in qwen1.5b tinyllama; do
  tx --bench truthfulqa --model $m --sink-bias -2 --tag sbm2
  tx --bench truthfulqa --model $m --sink-bias 2 --tag sbp2
  tx --bench truthfulqa --model $m --sink-bias 4 --tag sbp4
done
waitmem; echo "=== causal evaluate ($(date +%H:%M:%S))"
TOHA_SUFFIX=_causal $PY -m sinktda.toha evaluate truthfulqa_qwen1.5b truthfulqa_qwen1.5b_sbm2 truthfulqa_qwen1.5b_sbp2 truthfulqa_qwen1.5b_sbp4 \
   truthfulqa_tinyllama truthfulqa_tinyllama_sbm2 truthfulqa_tinyllama_sbp2 truthfulqa_tinyllama_sbp4
echo "=== CAUSAL DONE ($(date +%H:%M:%S))"
