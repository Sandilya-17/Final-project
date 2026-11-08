"""Per-seed comparison of the full-data runs (default settings): mean +/- std, how many seeds each model
wins, and a McNemar test per seed for every model pair. Seeds are detected from the result files."""
import glob, json, os, re, numpy as np, pandas as pd
from scipy.stats import binomtest
from common import ALL_MODELS, RES_DIR, DATASET

def tag(m, s): return f"{m}_1.0" + ("" if s == 42 else f"_s{s}")
SEEDS = sorted({42} | {int(x) for f in glob.glob(f"{RES_DIR}/*_1.0_s*.json")
                       for x in re.findall(r"_1\.0_s(\d+)\.json$", f)})
R = {m: {s: json.load(open(f"{RES_DIR}/{tag(m, s)}.json")) for s in SEEDS if os.path.exists(f"{RES_DIR}/{tag(m, s)}.json")}
     for m in ALL_MODELS}
R = {m: v for m, v in R.items() if v}
out = []
def say(s=""):
    print(s); out.append(s)

say(f"Dataset: {DATASET}\nModel         acc (mean +/- std)    macro-F1 (mean +/- std)   n_seeds   seeds")
for m, rs in R.items():
    a = np.array([r["acc"] for r in rs.values()]); f = np.array([r["macro_f1"] for r in rs.values()])
    sd = lambda x: x.std(ddof=1) if len(x) > 1 else 0
    say(f"{m:12s}  {a.mean():.3f} +/- {sd(a):.3f}      {f.mean():.3f} +/- {sd(f):.3f}          {len(rs)}         {sorted(rs)}")

say("\nPairs: seeds where A has the higher macro-F1, and exact McNemar test per seed (needs the *_preds.csv files)")
names = list(R)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        common = sorted(set(R[a]) & set(R[b]))
        if not common: continue
        wins = sum(R[a][s]["macro_f1"] > R[b][s]["macro_f1"] for s in common)
        diffs = [R[a][s]["macro_f1"] - R[b][s]["macro_f1"] for s in common]
        say(f"{a} vs {b}: {a} higher in {wins}/{len(common)} seeds, mean diff {np.mean(diffs):+.3f}")
        for s in common:
            pa, pb = f"{RES_DIR}/{tag(a, s)}_preds.csv", f"{RES_DIR}/{tag(b, s)}_preds.csv"
            if not (os.path.exists(pa) and os.path.exists(pb)): continue
            x, y = pd.read_csv(pa), pd.read_csv(pb)
            ok_a, ok_b = (x.true == x.pred).values, (y.true == y.pred).values
            n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            say(f"    seed {s}: only-{a} correct={n01}, only-{b} correct={n10}, p={p:.3f}")
open(f"{RES_DIR}/seeds_stats.txt", "w").write("\n".join(out))
