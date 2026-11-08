"""Builds the results tables, learning curve, confusion matrix and an error sample."""
import glob, json, os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

rows = [json.load(open(f)) for f in glob.glob("results/*.json")]
df = pd.DataFrame(rows)
for col, default in [("seed", 42), ("lr", 2e-5), ("epochs", 4)]:
    if col not in df: df[col] = default
    df[col] = df[col].fillna(default)
df = df.sort_values(["model", "frac", "lr", "epochs", "seed"])
df.to_csv("results/summary.csv", index=False)

default = df[(df.lr == 2e-5) & (df.epochs == 4) & (df.seed == 42)]
print("== Default settings, seed 42 ==")
print(default[["model", "frac", "acc", "macro_f1"]].to_string(index=False))

tuned = df[(df.lr != 2e-5) | (df.epochs != 4)]
if len(tuned):
    cols = [c for c in ["model", "frac", "lr", "epochs", "val_macro_f1", "acc", "macro_f1"] if c in tuned]
    print("\n== Tuned runs (choose by val_macro_f1, then report test) ==")
    print(tuned[cols].to_string(index=False))

plt.figure(figsize=(6, 4))
for m in ["mbert", "xlmr", "indicbert"]:
    d = default[default.model == m]
    if len(d): plt.plot(d.frac * 100, d.macro_f1, marker="o", label=m)
plt.xlabel("Training data (%)"); plt.ylabel("Macro-F1"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/learning_curve.png", dpi=200)

names = {"0": "C0", "1": "C1", "2": "C2"}
if os.path.exists("class_names.json"):
    names.update(json.load(open("class_names.json")))
labels = [names[str(i)] for i in range(3)]

full = default[(default.frac == 1.0) & default.model.isin(["mbert", "xlmr", "indicbert"])]
if len(full):
    best = full.sort_values("macro_f1").iloc[-1].model
    p = pd.read_csv(f"results/{best}_1.0_preds.csv")
    print("\nBest model:", best)
    print(classification_report(p.true, p.pred, target_names=labels))
    ConfusionMatrixDisplay(confusion_matrix(p.true, p.pred), display_labels=labels).plot(cmap="Blues")
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=200)
    p[p.true != p.pred].sample(min(50, (p.true != p.pred).sum()), random_state=1)\
        .to_csv("results/errors_to_analyze.csv", index=False)
    print("Wrote results/errors_to_analyze.csv")
