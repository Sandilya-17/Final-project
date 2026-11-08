"""Mean +/- std over seeds (full-data runs) and a McNemar test, IndicBERT vs XLM-R."""
import json, os, numpy as np, pandas as pd
from scipy.stats import binomtest

SEEDS = [42, 1, 2]
def tag(m, s): return f"{m}_1.0" + ("" if s == 42 else f"_s{s}")

print("Model        acc (mean +/- std)    macro-F1 (mean +/- std)   n_seeds")
for m in ["mbert", "xlmr", "indicbert"]:
    rs = [json.load(open(f"results/{tag(m, s)}.json")) for s in SEEDS if os.path.exists(f"results/{tag(m, s)}.json")]
    if not rs: continue
    a = np.array([r["acc"] for r in rs]); f = np.array([r["macro_f1"] for r in rs])
    print(f"{m:10s}   {a.mean():.3f} +/- {a.std(ddof=1) if len(a)>1 else 0:.3f}      {f.mean():.3f} +/- {f.std(ddof=1) if len(f)>1 else 0:.3f}          {len(rs)}")

print("\nMcNemar exact test, IndicBERT vs XLM-R (same test examples):")
for s in SEEDS:
    pa, pb = f"results/{tag('indicbert', s)}_preds.csv", f"results/{tag('xlmr', s)}_preds.csv"
    if not (os.path.exists(pa) and os.path.exists(pb)): continue
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    ok_a, ok_b = (a.true == a.pred).values, (b.true == b.pred).values
    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    print(f"seed {s}: IndicBERT-only correct={n01}, XLM-R-only correct={n10}, p={p:.3f}")
