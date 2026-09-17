#!/bin/bash
# Sequential extraction queue for the Mac (one model in memory at a time).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export TMPDIR=${TMPDIR:-$PWD/.tmp} JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-$PWD/.tmp}
PY=${PY:-.venv/bin/python}
run() { echo "=== $* ($(date +%H:%M:%S))"; $PY -m sinktda.extract "$@" 2>&1 | grep --line-buffered -v -i "warn\|Loading weights" ; }
run --bench truthfulqa --model qwen3b
run --bench truthfulqa --model phi3
run --bench truthfulqa --model tinyllama
run --bench truthfulqa --model qwen1.5b
run --bench truthfulqa --model smollm
run --bench triviaqa --model qwen3b --n 2000
run --bench triviaqa --model qwen1.5b --n 2000
run --bench triviaqa --model phi3 --n 2000
run --bench triviaqa --model tinyllama --n 2000
run --bench halueval --model qwen1.5b --n 500
run --bench halueval --model qwen3b --n 2000 --no-perhead
echo "=== ALL DONE ($(date +%H:%M:%S))"
