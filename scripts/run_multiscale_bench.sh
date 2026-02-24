#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-bench_runs}
mkdir -p "$OUT_DIR"

for N in 50000 200000 1000000; do
  echo "=== Running N=$N ==="
  /usr/bin/time -l tau-infinite \
    --tokens "$N" \
    --compare-ann-baselines \
    --ann-sample-tokens "$N" \
    --ann-queries 25 \
    --ann-top-k 64 \
    2>&1 | tee "$OUT_DIR/tau_ann_${N}.txt"
done
