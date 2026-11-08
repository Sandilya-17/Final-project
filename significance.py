"""Bootstrap 95% CIs, pairwise McNemar tests (Bonferroni-corrected), and error rate by sentence length
(full-data, seed 42, default settings). Uses every model that has a predictions file."""
import os, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from common import ALL_MODELS, RES_DIR, DATASET

MODELS = [m for m in ALL_MODELS if os.path.exists(f"{RES_DIR}/{m}_1.0_preds.csv")]
P = {m: pd.read_csv(f"{RES_DIR}/{m}_1.0_preds.csv") for m in MODELS}
out = []
def say(s=""):
    print(s); out.append(s)

rng = np.random.default_rng(0)
say(f"Dataset: {DATASET}\nBootstrap 95% confidence intervals (1000 resamples of the test set)")
for m in MODELS:
    t, p = P[m].true.values, P[m].pred.values
    accs, f1s = [], []
    for _ in range(1000):
        i = rng.integers(0, len(t), len(t))
        accs.append((t[i] == p[i]).mean()); f1s.append(f1_score(t[i], p[i], average="macro"))
    say(f"{m:10s} acc {np.mean(t == p):.3f} [{np.percentile(accs, 2.5):.3f}, {np.percentile(accs, 97.5):.3f}]   "
        f"macro-F1 {f1_score(t, p, average='macro'):.3f} [{np.percentile(f1s, 2.5):.3f}, {np.percentile(f1s, 97.5):.3f}]")

n_pairs = len(MODELS) * (len(MODELS) - 1) // 2
alpha = 0.05 / max(n_pairs, 1)
say(f"\nPairwise McNemar exact test (accuracy, same test examples); {n_pairs} comparisons, "
    f"Bonferroni threshold 0.05/{n_pairs} = {alpha:.4f}  (* = significant after correction)")
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        oa, ob = (P[a].true == P[a].pred).values, (P[b].true == P[b].pred).values
        n01, n10 = int((oa & ~ob).sum()), int((~oa & ob).sum())
        pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        say(f"{a} vs {b}: only-{a} correct={n01}, only-{b} correct={n10}, p={pv:.4f}{' *' if pv < alpha else ''}")

best = max(MODELS, key=lambda m: f1_score(P[m].true, P[m].pred, average="macro"))
d = P[best].copy()
d["words"] = d.text.astype(str).str.split().str.len()
d["bucket"] = pd.cut(d.words, [0, 8, 14, 1000], labels=["short (<=8 words)", "medium (9-14)", "long (15+)"])
say(f"\nError rate by sentence length, {best}")
g = d.groupby("bucket", observed=True).apply(lambda x: pd.Series({"n": len(x), "error_rate": (x.true != x.pred).mean()}))
say(g.round(3).to_string())
open(f"{RES_DIR}/significance.txt", "w").write("\n".join(out))
