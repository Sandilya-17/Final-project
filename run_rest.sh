#!/bin/bash
cd "$(dirname "$0")"; source venv/bin/activate
export BATCH=4 SEEDS="42 1 2"

# wait until stage A (finetune.py) has been gone for 2 minutes
n=0; while [ $n -lt 6 ]; do
  if pgrep -f "finetune.py" >/dev/null; then n=0; else n=$((n+1)); fi; sleep 20
done
echo "=== A finished, starting B (second dataset) ==="; bash run_extended.sh dataset2
echo "=== C (extra seeds) ===";                        bash run_extended.sh seeds
echo "=== D (new models, full protocol) ===";          bash run_extended.sh newmodels
echo "=== stats ===";                                  bash run_extended.sh stats
python dedupe_eval.py
DATASET=indicsent python significance.py
DATASET=indicsent python seeds_stats.py
git add -A && git commit -m "Add extended results: new models, second dataset, seeds, dedup eval" && git push
echo "=== ALL DONE ==="
