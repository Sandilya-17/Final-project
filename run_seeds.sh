#!/bin/bash
# Extra seeds for the full-data runs (about 1.5 hours on an M2 Mac).
# Run from inside the project folder:  caffeinate -i bash run_seeds.sh
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
for s in 1 2; do
  for m in xlmr mbert indicbert; do
    [ -f results/${m}_1.0_s${s}.json ] && { echo "skip $m seed $s"; continue; }
    python finetune.py $m 1.0 $s || BATCH=4 python finetune.py $m 1.0 $s
  done
done
python seeds_stats.py
