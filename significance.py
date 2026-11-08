"""Bootstrap 95% CIs, pairwise McNemar tests, and error rate by sentence length (full-data, seed 42)."""
import os, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

MODELS = [m for m in ["mbert", "xlmr", "indicbert"] if os.path.exists(f"results/{m}_1.0_preds.csv")]
P = {m: pd.read_csv(f"results/{m}_1.0_preds.csv") for m in MODELS}
out = []
def say(s=""):
    print(s); out.append(s)

rng = np.random.default_rng(0)
say("Bootstrap 95% confidence intervals (1000 resamples of the test set)")
for m in MODELS:
    t, p = P[m].true.values, P[m].pred.values
    accs, f1s = [], []
    for _ in range(1000):
        i = rng.integers(0, len(t), len(t))
        accs.append((t[i] == p[i]).mean()); f1s.append(f1_score(t[i], p[i], average="macro"))
    say(f"{m:10s} acc {np.mean(t == p):.3f} [{np.percentile(accs, 2.5):.3f}, {np.percentile(accs, 97.5):.3f}]   "
        f"macro-F1 {f1_score(t, p, average='macro'):.3f} [{np.percentile(f1s, 2.5):.3f}, {np.percentile(f1s, 97.5):.3f}]")

say("\nPairwise McNemar exact test (accuracy, same test examples)")
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        oa, ob = (P[a].true == P[a].pred).values, (P[b].true == P[b].pred).values
        n01, n10 = int((oa & ~ob).sum()), int((~oa & ob).sum())
        pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        say(f"{a} vs {b}: only-{a} correct={n01}, only-{b} correct={n10}, p={pv:.4f}")

best = max(MODELS, key=lambda m: f1_score(P[m].true, P[m].pred, average="macro"))
d = P[best].copy()
d["words"] = d.text.astype(str).str.split().str.len()
d["bucket"] = pd.cut(d.words, [0, 8, 14, 1000], labels=["short (<=8 words)", "medium (9-14)", "long (15+)"])
say(f"\nError rate by sentence length, {best}")
g = d.groupby("bucket", observed=True).apply(lambda x: pd.Series({"n": len(x), "error_rate": (x.true != x.pred).mean()}))
say(g.round(3).to_string())
open("results/significance.txt", "w").write("\n".join(out))
