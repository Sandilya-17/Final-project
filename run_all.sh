#!/bin/bash
# Runs everything. Safe to stop and re-run: finished runs are skipped.
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
[ -f data/train.csv ] || python prepare_data.py
[ -f results/logreg_1.0.json ] || python baseline.py
for m in xlmr mbert indicbert; do
  for f in 0.1 0.25 0.5 1.0; do
    [ -f results/${m}_${f}.json ] && { echo "skip $m $f"; continue; }
    python finetune.py $m $f
  done
done
python analyze.py
