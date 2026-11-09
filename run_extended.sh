#!/bin/bash
# Extended experiments for the journal revision. Safe to stop and re-run: finished runs are skipped.
# Run from the project folder, one stage at a time (times are rough, for an Apple M2):
#
#   python verify_labels.py                        # label mapping check (minutes) - do this first
#   caffeinate -i bash run_extended.sh seeds       # more seeds: data size + learning rate     (3 models)   ~5-7 h
#   caffeinate -i bash run_extended.sh ablation    # warmup + trainable embeddings, full data  (3 models)   ~7-9 h
#   caffeinate -i bash run_extended.sh newmodels   # MuRIL + Telugu-BERT, whole protocol                    ~6-8 h
#   bash run_extended.sh stats                     # tables, plots, significance tests
#
# SEEDS="42" bash run_extended.sh ablation   -> one seed first (about a third of the time), add seeds later.
# SEEDS defaults to "42 1 2". Use SEEDS="42 1 2 3 4" for five seeds.
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
SEEDS="${SEEDS:-42 1 2}"

# ft <model> <fraction> <seed> [VAR=value ...]   (retries with a smaller batch if memory runs out;
# gradient accumulation is raised so the effective batch size stays 16)
ft() {
  local m=$1 f=$2 s=$3; shift 3
  env SKIP_DONE=1 "$@" python finetune.py "$m" "$f" "$s" \
    || env SKIP_DONE=1 BATCH=4 "$@" python finetune.py "$m" "$f" "$s" \
    || echo "FAILED: $m $f $s $*"
}

# data-size study + learning-rate sensitivity for one model, all seeds
grid() {
  local m=$1
  for s in $SEEDS; do
    for f in 0.1 0.25 0.5 1.0; do ft "$m" $f "$s"; done
    ft "$m" 1.0 "$s" LR=3e-5 EPOCHS=4
    ft "$m" 1.0 "$s" LR=5e-5 EPOCHS=4
    ft "$m" 0.1 "$s" LR=5e-5 EPOCHS=10
  done
}

stage_seeds()    { for m in xlmr mbert indicbert; do grid $m; done; }
stage_newmodels(){ for m in muril telugubert; do grid $m; done; }
stage_ablation() {
  for m in xlmr mbert indicbert; do
    for s in $SEEDS; do
      ft $m 1.0 $s WARMUP=0.1                      # warmup at the default learning rate
      ft $m 1.0 $s LR=5e-5 EPOCHS=4 WARMUP=0.1     # does warmup remove the collapse at 5e-5?
      ft $m 1.0 $s UNFREEZE_EMB=1                  # word embeddings trainable (needs more memory)
    done
  done
}
stage_stats() {
  python analyze.py; python aggregate.py; python significance.py; python seeds_stats.py
}

case "${1:-}" in
  seeds)     stage_seeds ;;
  newmodels) stage_newmodels ;;
  ablation)  stage_ablation ;;
  stats)     stage_stats ;;
  all)       stage_seeds; stage_newmodels; stage_ablation; stage_stats ;;
  *) echo "usage: bash run_extended.sh seeds|ablation|newmodels|stats|all"; exit 1 ;;
esac
