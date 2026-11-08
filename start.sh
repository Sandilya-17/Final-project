#!/bin/bash
# ONE command to run the whole project:  bash start.sh
cd "$(dirname "$0")"
[ -d venv ] || bash setup.sh
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
[ -f data/train.csv ] || python prepare_data.py || { echo "DATA DOWNLOAD FAILED - send me the error above"; exit 1; }
[ -f results/logreg_1.0.json ] || python baseline.py
for m in xlmr mbert indicbert; do
  for f in 0.1 0.25 0.5 1.0; do
    [ -f results/${m}_${f}.json ] && { echo "skip $m $f"; continue; }
    # auto-retry with smaller batch if memory runs out
    python finetune.py $m $f || BATCH=8 python finetune.py $m $f || BATCH=4 python finetune.py $m $f
  done
done
python analyze.py
echo "DONE. Results are in the results/ folder."
