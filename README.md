# Telugu Sentiment Classification with Multilingual Transformers

A comparative study of five pretrained transformers (**mBERT**, **XLM-RoBERTa**, **IndicBERT v2**, **MuRIL**, **Telugu-BERT**) on three-class Telugu sentiment classification (positive, negative, neutral), with TF-IDF baselines, a training-data-size study, a learning-rate / warmup ablation, multi-seed statistics, significance tests and error analysis.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

---

## Overview

Telugu is spoken by over 80 million people but remains under-resourced in NLP. This project fine-tunes five pretrained models on the Telugu subset of IndicGLUE, compares them with classical TF-IDF baselines, and reports how stable each model is across random seeds and training-set sizes.

## Dataset

- **Source:** IndicGLUE `inltkh.te` (Telugu), loaded through Hugging Face Datasets
- **Task:** 3-class sentiment classification
- **Size:** 4,328 train / 541 validation / 541 test examples
- **Classes:** the dataset stores integer ids 7, 8 and 9, which the Hugging Face label list for `inltkh` names *positive*, *negative* and *neutral*. These are remapped to 0, 1 and 2 (`class_names.json`).

### Label check

The mapping was checked in two ways, and the results are not fully consistent:

- A manual annotation of 90 sentences (30 per class; `label_annotation.csv`, `latest_labels.csv`, key in `label_key.csv`; one annotator who saw only the sentence) agrees with the Hugging Face mapping on 61 of 90 sentences (68%, Cohen's kappa 0.52). Agreement is 22/30 for id 7 (positive) and 22/30 for id 8 (negative), but only 17/30 for id 9 (neutral): 11 of the 30 neutral sentences were judged negative. The neutral class is therefore noisy, which may also limit achievable accuracy.
- An automatic comparison with `actsa-sc.te` by sentence text (`verify_labels.py`, `results/label_verification.txt`) matched most sentences but gave a conflicting mapping (ids 7 and 8 both positive). We could not resolve this and treat the manual check as the stronger evidence. Per-class results for *neutral* should be read with caution.

### Data quality

Removing test sentences that also occur in train, carry conflicting labels, or repeat within the test set leaves 523 of 541 test examples (`results/dedupe_eval.txt`). Macro-F1 changes by at most 0.002 for every model (3 seeds), so the results below are not materially affected.

## Results

### Main results (full training data, default settings, 5 seeds)

Test macro-F1, mean ± standard deviation over seeds 42, 1, 2, 3, 4:

| Model | Macro-F1 | Notes |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.472 | single run |
| Linear SVM (TF-IDF) | 0.500 | single run |
| mBERT | 0.578 ± 0.011 | |
| XLM-R base | 0.659 ± 0.015 | |
| MuRIL | 0.581 ± 0.206 | one of 5 runs collapsed to the majority class; **0.673 ± 0.015** over the other 4 |
| IndicBERT v2 | 0.679 ± 0.012 | |
| Telugu-BERT | 0.688 ± 0.011 | |

**Summary:** Telugu-BERT, IndicBERT v2, XLM-R and MuRIL (excluding its collapsed run) form a top group at about 0.66 to 0.69 macro-F1 and are not statistically distinguishable. mBERT is clearly lower, and the top group beats the best TF-IDF baseline by about 16 to 19 macro-F1 points (mBERT by about 8). We do not claim a single best model.

### Significance (seed 1)

Pairwise exact McNemar tests on test-set accuracy, Bonferroni threshold 0.05 / 10 = 0.005 (`results/significance_seed1.txt`):

- mBERT is significantly worse than XLM-R (p = 0.0003), IndicBERT (p = 0.0025), Telugu-BERT (p < 0.001) and MuRIL (p < 0.001).
- No pair among XLM-R, IndicBERT, Telugu-BERT and MuRIL is significant (all p ≥ 0.08).

Seed 42 is not used for these tests because MuRIL collapsed in that run, which would make every MuRIL comparison an artefact (`results/significance.txt` is kept for reference).

### Training collapses

Some runs predict a single class and score a macro-F1 of about 0.214. Across the experiments this happened in 25 of 138 runs (the 10 runs added in the last round had none):

| Setting | Collapsed runs | Interpretation |
|---|---|---|
| 10% of the data, default (2e-5, 4 epochs) | 12 of 15 | undertrained (about 108 optimizer steps) |
| Full data, learning rate 5e-5 | 5 (mBERT 2, XLM-R 2, Telugu-BERT 1) | learning rate too high |
| XLM-R, 5e-5 with 0.1 warmup | 3 of 3 | warmup does not fix the high-learning-rate collapse |
| MuRIL, full data, default | 1 of 5 (seed 42) | instability |
| MuRIL, 25% of the data, default | 2 of 3 | instability |
| 10% of the data at 5e-5 (MuRIL, XLM-R) | 2 | high learning rate on little data |

At 10% of the data, longer training (5e-5, 10 epochs) rescues IndicBERT (0.613 ± 0.010, n = 3), but MuRIL stays unstable (0.480 ± 0.231, n = 3). Treat collapsed runs as optimisation failures, not as a property of the model.

### Data-size study, ablations and error analysis

Macro-F1 versus training-set size (10%, 25%, 50%, 100%), the learning-rate and warmup ablation, the bootstrap 95% confidence intervals and the error rate by sentence length are in `results/aggregate_tables.md`, `results/summary.csv` and `results/significance.txt`. The data-size and ablation studies use 3 seeds; the full-data table above uses 5.

![Learning curve](results/learning_curve.png)
![Confusion matrix](results/confusion_matrix.png)

## Training setup

| Setting | Value |
|---|---|
| Epochs | 4 (10 in the 10%-data rescue setting) |
| Learning rate | 2e-5 (3e-5 and 5e-5 in the ablation) |
| Effective batch size | 16 (8 per device x 2 gradient accumulation; falls back to batch 4 if memory runs out) |
| Max sequence length | 128 |
| Frozen layers | Word-embedding layer (to fit in memory); an unfrozen-embedding ablation is included |
| Seeds | 42, 1, 2, 3, 4 |
| Hardware | Apple M2 (MPS backend) |

## Project structure

```
.
├── prepare_data.py          # downloads the dataset to data/
├── baseline.py              # TF-IDF + Logistic Regression / Linear SVM
├── finetune.py              # fine-tunes one model: python finetune.py <model> <fraction> <seed>
├── analyze.py               # tables, plots, error samples
├── aggregate.py             # mean ± sd over seeds -> results/aggregate_tables.md
├── significance.py          # bootstrap CIs + McNemar tests (seed 42)
├── significance_s1.py       # the same tests on seed 1 (used in the paper)
├── seeds_stats.py           # multi-seed statistics
├── verify_labels.py         # label-mapping check against a reference dataset
├── run_extended.sh          # seeds | ablation | newmodels | stats
├── class_names.json         # id -> class name
├── label_annotation.csv, latest_labels.csv, label_key.csv   # manual label check
├── LICENSE
├── requirements.txt
└── results/                 # metrics, predictions, tables, plots
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/Sandilya-17/Final-project.git
cd Final-project

# 2. Create an environment and install packages
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download the data
python prepare_data.py

# 4. Run everything (baselines + fine-tuning + plots)
bash start.sh
```

Run a single experiment (model: `mbert | xlmr | indicbert | muril | telugubert`, fraction 0.1 to 1.0):

```bash
python finetune.py indicbert 1.0 42
```

Settings are read from environment variables: `LR`, `EPOCHS`, `BATCH`, `WARMUP`, `UNFREEZE_EMB=1`, `SKIP_DONE=1`. For example, `LR=5e-5 EPOCHS=10 python finetune.py indicbert 0.1 42`.

Recompute the statistics from the saved predictions:

```bash
python analyze.py; python aggregate.py; python significance.py; python seeds_stats.py
python significance_s1.py    # seed-1 McNemar tests
```

> Full training takes about 9 to 15 minutes per run on an M2 Mac. The complete study (all seeds and ablations) takes about a day.

## Limitations

- One dataset and one language, with short texts.
- The *neutral* class is noisy. The label mapping was checked on a small manual sample (90 sentences, one annotator, kappa 0.52 against the dataset labels), and an automatic check against `actsa-sc.te` disagreed.
- Hyperparameters were not tuned per model. Default settings were used for the main table.
- Word embeddings were frozen for the main experiments because of hardware memory limits.
- The data-size and ablation studies use 3 seeds; the main table uses 5. Differences among the top four models are within seed-to-seed variation.
- XLM-R is sensitive to the learning rate: at 3e-5 its full-data macro-F1 drops to 0.559 ± 0.177 (3 seeds), although no run falls below the collapse threshold.
- Several runs collapse to the majority class (see the table above). Reported means include them unless stated otherwise.

## Citation

If you use this work, please cite:

```bibtex
@misc{sandilya2026telugu,
  author = {Sandilya},
  title  = {Telugu Sentiment Classification with Multilingual Transformers},
  year   = {2026},
  url    = {https://github.com/Sandilya-17/Final-project}
}
```

## Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for IndicGLUE and IndicBERT
- [Hugging Face](https://huggingface.co/) Transformers and Datasets

## Contact

**Sandilya** · sandilya939174@gmail.com
