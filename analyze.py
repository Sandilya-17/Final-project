"""Builds results table, learning curve, confusion matrix, and wrong-prediction sample."""
import glob, json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

rows = [json.load(open(f)) for f in glob.glob("results/*.json")]
df = pd.DataFrame(rows).sort_values(["model", "frac"])
df.to_csv("results/summary.csv", index=False)
print(df.to_string(index=False))

plt.figure(figsize=(6, 4))
for m in ["mbert", "xlmr", "indicbert"]:
    d = df[(df.model == m) & (df.get("seed", 42).fillna(42) == 42)]
    if len(d): plt.plot(d.frac * 100, d.macro_f1, marker="o", label=m)
plt.xlabel("Training data (%)"); plt.ylabel("Macro-F1"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/learning_curve.png", dpi=200)

full = df[df.frac == 1.0]
full = full[full.model.isin(["mbert", "xlmr", "indicbert"])]
if len(full):
    best = full.sort_values("macro_f1").iloc[-1].model
    p = pd.read_csv(f"results/{best}_1.0_preds.csv")
    print("\nBest model:", best)
    print(classification_report(p.true, p.pred))
    ConfusionMatrixDisplay(confusion_matrix(p.true, p.pred)).plot(cmap="Blues")
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=200)
    p[p.true != p.pred].sample(min(50, (p.true != p.pred).sum()), random_state=1)\
        .to_csv("results/errors_to_analyze.csv", index=False)
    print("Wrote results/errors_to_analyze.csv - read these and group the error types.")
