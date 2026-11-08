# Extended experiments (journal revision)

Everything below is additive: the original files, commands and results keep working
(`DATASET` defaults to `inltkh`, which uses `data/` and `results/`).

| Reviewer concern | What to run | Output |
|---|---|---|
| Label mapping 7/8/9 not verified | `python verify_labels.py` | `results/label_verification.txt` |
| Single seed for data-size / learning-rate studies | `bash run_extended.sh seeds` | mean ± std tables, error-bar plots |
| No warmup, frozen embeddings | `bash run_extended.sh ablation` | ablation table in `aggregate_tables.md` |
| Only three models | `bash run_extended.sh newmodels` (MuRIL, L3Cube Telugu-BERT) | same tables, 5 models |
| One dataset | `bash run_extended.sh dataset2` | `results_indicsent/` |
| Tables for the paper | `bash run_extended.sh stats` | `results/aggregate_tables.md`, `learning_curve_seeds.png`, `lr_sensitivity_seeds.png`, `significance.txt`, `seeds_stats.txt` |

## Order

1. `python verify_labels.py` first. If the ids do not match 7 = positive, 8 = negative, 9 = neutral,
   edit `class_names.json`; the model comparison stays valid, the error analysis and class names change.
2. `SEEDS="42" caffeinate -i bash run_extended.sh ablation` (one seed, fastest answer to
   "does warmup remove the collapse at 5e-5?"), then the other stages with the default `SEEDS="42 1 2"`.
3. `bash run_extended.sh stats`, then update the paper tables from `results/aggregate_tables.md`.

Stages skip finished runs, so you can stop and restart. Times are rough estimates for an Apple M2,
based on the ~15 minutes per full-data run implied by `run_seeds.sh`.

## Settings added to `finetune.py` (all off by default)

`WARMUP=0.1` (linear warmup over 10% of the optimizer steps), `UNFREEZE_EMB=1` (train the word-embedding
layer; needs more memory), `SKIP_DONE=1`, `DATASET=<name>`, and two new models: `muril`
(`google/muril-base-cased`) and `telugubert` (`l3cube-pune/telugu-bert`).
`BATCH=4` now raises gradient accumulation so the effective batch size stays 16 (before, it silently
became 8). With the default `BATCH=8` nothing changes.

## Second dataset

`prepare_data2.py indicsentiment` builds `data_indicsent/` from `ai4bharat/IndicSentiment` (Telugu).
It is binary, review-domain and machine-translated, so it tests whether the ranking survives a domain and
label-set change. It is not a native Telugu three-class set. To use another Telugu dataset:
`python prepare_data2.py csv file.csv myname` (columns `text`, `label`), then `DATASET=myname ...`.

## After the runs: what to change in the paper

- Report mean ± std (and n) for data size, learning rate and the new ablations; replace "single seed" statements.
- Say whether warmup removes the 5e-5 collapse and whether trainable embeddings change the ranking.
- Update the Bonferroni threshold: it is 0.05 divided by the number of model pairs
  (printed at the top of the McNemar section of `significance.txt`).
- Report the label verification result in Materials and Methods and in the validity section.
- Add the second dataset and the two new models to the abstract, RQs and limitations.
