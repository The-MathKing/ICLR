#!/bin/bash
# Evaluate every finished extraction that has no results yet; loop until the queue is done.
export TMPDIR=${TMPDIR:-$PWD/.tmp} JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-$PWD/.tmp}
while true; do
  for d in sinktda_out/*/; do
    n=$(basename $d)
    [[ -f $d/layers.parquet && ! -f sinktda_results/comp_$n.csv && $n != *debug* ]] && .venv/bin/python -m sinktda.evaluate $n
  done
  grep -q "ALL DONE" logs/extract_all.log && { 
    for d in sinktda_out/*/; do n=$(basename $d); [[ -f $d/layers.parquet && ! -f sinktda_results/comp_$n.csv ]] && .venv/bin/python -m sinktda.evaluate $n; done; echo EVAL-ALL-DONE; break; }
  sleep 60
done
