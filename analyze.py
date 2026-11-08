"""Builds the results tables, learning curve, confusion matrix and an error sample (seed 42, default settings).
For mean +/- std over seeds and the ablations, run aggregate.py.  Use DATASET=<name> for another dataset."""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
from common import ALL_MODELS, DATA_DIR, RES_DIR, DATASET, DEFAULTS, class_names, load_results

df = load_results().sort_values(["model", "frac", "lr", "epochs", "warmup", "seed"])
df.to_csv(f"{RES_DIR}/summary.csv", index=False)

is_default = ((df.lr == DEFAULTS["lr"]) & (df.epochs == DEFAULTS["epochs"]) & (df.seed == DEFAULTS["seed"])
              & (df.warmup == 0) & df.emb_frozen)
default, tuned = df[is_default], df[(df.seed == DEFAULTS["seed"]) & ~is_default]
print(f"== Default settings, seed 42 ({DATASET}) ==")
print(default[["model", "frac", "acc", "macro_f1"]].to_string(index=False))

k, maj_f1 = 3, None
if os.path.exists(f"{DATA_DIR}/test.csv"):   # data/ is not in git; run prepare_data.py to get it
    te = pd.read_csv(f"{DATA_DIR}/test.csv")
    maj, k = te.label.value_counts(), te.label.nunique()
    share = maj.iloc[0] / len(te)
    maj_f1 = 2 * share / (1 + share) / k      # F1 of the majority class, other classes score 0
    print(f"\nMajority-class reference: accuracy {share:.3f}, macro-F1 {maj_f1:.3f}")

if len(tuned):
    cols = [c for c in ["model", "frac", "lr", "epochs", "warmup", "emb_frozen", "val_macro_f1", "acc", "macro_f1"] if c in tuned]
    print("\n== Non-default runs, seed 42 (choose by val_macro_f1, then report test) ==")
    print(tuned[cols].to_string(index=False))

plt.figure(figsize=(6, 4))
for m in ALL_MODELS:
    d = default[default.model == m]
    if len(d): plt.plot(d.frac * 100, d.macro_f1, marker="o", label=m)
if maj_f1 is not None: plt.axhline(maj_f1, color="gray", ls=":", lw=1)
plt.xlabel("Training data (%)"); plt.ylabel("Macro-F1"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{RES_DIR}/learning_curve.png", dpi=200)

labels = class_names(k)
full = default[(default.frac == 1.0) & default.model.isin(ALL_MODELS)]
full = full[[os.path.exists(f"{RES_DIR}/{m}_1.0_preds.csv") for m in full.model]]
if len(full):
    best = full.sort_values("macro_f1").iloc[-1].model
    p = pd.read_csv(f"{RES_DIR}/{best}_1.0_preds.csv")
    print("\nBest model:", best)
    print(classification_report(p.true, p.pred, target_names=labels))
    ConfusionMatrixDisplay(confusion_matrix(p.true, p.pred), display_labels=labels).plot(cmap="Blues")
    plt.tight_layout(); plt.savefig(f"{RES_DIR}/confusion_matrix.png", dpi=200)
    p[p.true != p.pred].sample(min(50, (p.true != p.pred).sum()), random_state=1)\
        .to_csv(f"{RES_DIR}/errors_to_analyze.csv", index=False)
    print(f"Wrote {RES_DIR}/errors_to_analyze.csv")
