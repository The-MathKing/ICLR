#!/bin/bash
# Follow-up queue: waits for run_all.sh, then reruns the Qwen2.5-1.5B settings in bf16
# (fp16 on MPS gives non-finite logits for this model).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false TMPDIR=${TMPDIR:-$PWD/.tmp} JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-$PWD/.tmp}
PY=${PY:-.venv/bin/python}
while pgrep -f "sinktda/run_all.sh" >/dev/null; do sleep 20; done
run() { echo "=== $* ($(date +%H:%M:%S))"; $PY -m sinktda.extract "$@" 2>&1 | grep --line-buffered -v -i "warn\|Loading weights" ; }
rm -rf sinktda_out/halueval_qwen1.5b
run --bench truthfulqa --model qwen1.5b --dtype bfloat16
run --bench halueval --model qwen1.5b --n 500 --dtype bfloat16
echo "=== MORE DONE ($(date +%H:%M:%S))"
