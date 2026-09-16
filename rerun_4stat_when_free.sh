#!/bin/bash
# Waits until system memory is free (and the user's known intensive test process
# has exited) before re-extracting Phi-3 / Qwen2.5-3B TruthfulQA features with the
# full four-statistic descriptor bank (matching HaluEval/Qwen2.5-3B and
# TruthfulQA/SmolLM). Logs progress so the driving session can check back later
# without polling this shell directly.

set -uo pipefail
cd /Volumes/2TB/iclr
LOG=/Volumes/2TB/iclr/rerun_4stat.log
MIN_FREE_MB=8000   # require ~8GB free before loading a 3.8B-class model in fp16
MAX_WAIT_SECS=36000  # 10 hours safety cap
POLL_SECS=300

echo "[$(date)] Waiting for memory to free up (need ${MIN_FREE_MB}MB, checking every ${POLL_SECS}s)..." >> "$LOG"

elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT_SECS" ]; do
    # "Available" memory in MB: free + inactive + speculative pages (16KB pages
    # on Apple Silicon). Raw "Pages free" alone drastically undercounts what's
    # actually available, since macOS keeps reclaimable file-backed/cached pages
    # marked "inactive" rather than "free" until something needs them. This
    # free+inactive+speculative sum is what Activity Monitor effectively reports
    # as available memory.
    free_mb=$(vm_stat | awk '
        /Pages free/ {gsub("\\.","",$3); f=$3}
        /Pages inactive/ {gsub("\\.","",$3); i=$3}
        /Pages speculative/ {gsub("\\.","",$3); s=$3}
        END {print int((f+i+s)*16384/1024/1024)}
    ')

    # Also back off while the user's known intensive job is still running
    busy=$(pgrep -f "run_ablation_study|run_comprehensive_benchmarks" | wc -l | tr -d ' ')

    echo "[$(date)] available_mb=${free_mb} busy_procs=${busy}" >> "$LOG"

    if [ "$free_mb" -ge "$MIN_FREE_MB" ] && [ "$busy" -eq 0 ]; then
        echo "[$(date)] Memory available and no busy process detected. Starting extraction." >> "$LOG"
        break
    fi
    sleep "$POLL_SECS"
    elapsed=$(( elapsed + POLL_SECS ))
done

if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
    echo "[$(date)] Timed out waiting for free memory after ${MAX_WAIT_SECS}s. Aborting without running." >> "$LOG"
    exit 1
fi

source .venv/bin/activate 2>/dev/null

echo "[$(date)] Running phase10_qwen3b_eval.py (4-stat bank) ..." >> "$LOG"
python3 -c "import phase10_qwen3b_eval as m; df = m.run_extraction(); print('qwen3b rows:', len(df)); m.run_evaluation(df)" >> "$LOG" 2>&1
echo "[$(date)] Finished phase10_qwen3b_eval.py" >> "$LOG"

echo "[$(date)] Running phase10_phi3_eval.py (4-stat bank) ..." >> "$LOG"
python3 -c "import phase10_phi3_eval as m; df = m.run_extraction(); print('phi3 rows:', len(df)); m.run_evaluation(df)" >> "$LOG" 2>&1
echo "[$(date)] Finished phase10_phi3_eval.py" >> "$LOG"

echo "[$(date)] DONE. New CSVs: phase10_results/qwen3b_truthfulqa_4stat.csv, phase10_results/phi3_truthfulqa_4stat.csv" >> "$LOG"
echo "[$(date)] NOTE: run_comprehensive_evals.py / paper.tex were NOT modified automatically — review results first." >> "$LOG"
