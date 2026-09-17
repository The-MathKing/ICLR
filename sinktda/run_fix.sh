#!/bin/bash
# Review fixes: re-extract every fp16 setting (and both HaluEval settings, after the
# prefix-boundary fix) in bfloat16, then re-evaluate. One heavy job at a time.
export HF_HOME=${HF_HOME:-/Volumes/2TB/hf_cache}
export TOKENIZERS_PARALLELISM=false TMPDIR=/Volumes/2TB/iclr/.tmp JOBLIB_TEMP_FOLDER=/Volumes/2TB/iclr/.tmp
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
ext() { waitmem; echo "=== extract $* ($(date +%H:%M:%S))"; $PY -m sinktda.extract --workers 4 --dtype bfloat16 "$@" 2>&1 | grep --line-buffered -v -i "warn\|Loading weights"; }
ev()  { [ -f sinktda_results/comp_$1.csv ] && return; [ -f sinktda_out/$1/layers.parquet ] || { echo "[skip-eval] $1 missing"; return; }
        waitmem; echo "=== eval $1 ($(date +%H:%M:%S))"; $PY -m sinktda.evaluate "$1" 2>&1 | grep --line-buffered -E "^\{|Error|Traceback" ; }

ext --bench truthfulqa --model qwen3b;               ev truthfulqa_qwen3b
ext --bench truthfulqa --model phi3;                 ev truthfulqa_phi3
ext --bench truthfulqa --model tinyllama;            ev truthfulqa_tinyllama
ext --bench truthfulqa --model smollm;               ev truthfulqa_smollm
ext --bench halueval --model qwen1.5b --n 500;       ev halueval_qwen1.5b
ext --bench halueval --model qwen3b --n 1000;        ev halueval_qwen3b
waitmem; echo "=== theory-only for unchanged settings ($(date +%H:%M:%S))"
$PY -m sinktda.evaluate --theory-only truthfulqa_qwen1.5b triviaqa_qwen3b triviaqa_qwen1.5b triviaqa_phi3 triviaqa_tinyllama 2>&1 | grep -c setting
waitmem; echo "=== layer_split ($(date +%H:%M:%S))"; $PY -m sinktda.layer_split 2>&1 | grep --line-buffered -E "^\{|Error"
waitmem; echo "=== worked example ($(date +%H:%M:%S))"; $PY -m sinktda.appendix_extra --examples 2>&1 | tail -2
echo "=== FIX DONE ($(date +%H:%M:%S))"
