#!/bin/bash
# Small tuning study (about 1.5 hours on an M2 Mac). Run:  caffeinate -i bash run_tuning.sh
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
for m in xlmr mbert indicbert; do
  # 1) low-data fix: more epochs and a higher learning rate at 10% of the data
  [ -f results/${m}_0.1_lr5e-5_ep10.json ] || LR=5e-5 EPOCHS=10 python finetune.py $m 0.1 || LR=5e-5 EPOCHS=10 BATCH=4 python finetune.py $m 0.1
  # 2) learning-rate sweep on the full data
  for lr in 3e-5 5e-5; do
    [ -f results/${m}_1.0_lr${lr}_ep4.json ] || LR=$lr EPOCHS=4 python finetune.py $m 1.0 || LR=$lr EPOCHS=4 BATCH=4 python finetune.py $m 1.0
  done
done
python analyze.py
