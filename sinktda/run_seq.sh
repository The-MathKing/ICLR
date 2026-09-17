#!/bin/bash
# Memory-safe sequential runner for a 16 GB machine: one heavy job at a time.
# Each step waits until at least MINFREE% of RAM is free (macOS memory_pressure).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false TMPDIR=${TMPDIR:-$PWD/.tmp} JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-$PWD/.tmp}
export SINKTDA_JOBS=2 OMP_NUM_THREADS=4
PY=${PY:-.venv/bin/python}
MINFREE=${MINFREE:-35}
waitmem() {
  while true; do
    f=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
    [ "${f:-0}" -ge "$MINFREE" ] && break
    echo "[mem] ${f}% free, waiting"; sleep 20
  done
}
ext() { waitmem; echo "=== extract $* ($(date +%H:%M:%S))"; $PY -m sinktda.extract --workers 4 "$@" 2>&1 | grep --line-buffered -v -i "warn\|Loading weights"; }
ev()  { [ -f sinktda_results/comp_$1.csv ] && return; [ -f sinktda_out/$1/layers.parquet ] || { echo "[skip-eval] $1 missing"; return; }
        waitmem; echo "=== eval $1 ($(date +%H:%M:%S))"; $PY -m sinktda.evaluate "$1" 2>&1 | grep --line-buffered -E "^\{|Error|Traceback" ; }

ev  triviaqa_qwen1.5b
ext --bench triviaqa --model phi3 --n 2000;          ev triviaqa_phi3
ext --bench triviaqa --model tinyllama --n 2000;     ev triviaqa_tinyllama
ext --bench truthfulqa --model qwen1.5b --dtype bfloat16;            ev truthfulqa_qwen1.5b
ext --bench halueval --model qwen1.5b --n 500 --dtype bfloat16;      ev halueval_qwen1.5b
rm -rf sinktda_out/halueval_qwen3b
ext --bench halueval --model qwen3b --n 1000 --no-perhead;           ev halueval_qwen3b
waitmem; echo "=== layer_split ($(date +%H:%M:%S))"; $PY -m sinktda.layer_split 2>&1 | grep --line-buffered -E "^\{|Error"
waitmem; echo "=== timing ($(date +%H:%M:%S))"; $PY -m sinktda.timing --model tinyllama 2>&1 | tail -12
$PY -m sinktda.report > /dev/null
echo "=== SEQ DONE ($(date +%H:%M:%S))"
