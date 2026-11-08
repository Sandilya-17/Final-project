import glob, re, unicodedata, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import ALL_MODELS, DATA_DIR, RES_DIR, DATASET

norm = lambda s: " ".join(unicodedata.normalize("NFC", str(s)).split())
tr, va, te = (pd.read_csv(f"{DATA_DIR}/{s}.csv") for s in ["train", "validation", "test"])
allr = pd.concat([tr, va, te]).assign(k=lambda d: d.text.map(norm))
g = allr.groupby("k").label.nunique()
conflict, train_keys = set(g[g > 1].index), set(tr.text.map(norm))

lines = [f"Dataset: {DATASET}. Cleaned test = test minus sentences found in train, with conflicting labels, or repeated in test."]
sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else float("nan")
for m in ALL_MODELS:
    files = [f for f in glob.glob(f"{RES_DIR}/{m}_1.0*_preds.csv") if re.fullmatch(rf"{m}_1\.0(_s\d+)?_preds\.csv", f.split("/")[-1])]
    full, clean, n_clean = [], [], None
    for f in sorted(files):
        p = pd.read_csv(f)
        k = p.text.map(norm)
        keep = ~k.isin(train_keys) & ~k.isin(conflict) & ~k.duplicated()
        full.append(f1_score(p.true, p.pred, average="macro"))
        clean.append(f1_score(p.true[keep], p.pred[keep], average="macro")); n_clean = int(keep.sum())
        n_full = len(p)
    if full:
        lines.append(f"{m:11s} seeds={len(full)}  full test (n={n_full}) macro-F1 {np.mean(full):.3f} ± {sd(full):.3f}   "
                     f"cleaned test (n={n_clean}) macro-F1 {np.mean(clean):.3f} ± {sd(clean):.3f}")
print("\n".join(lines)); open(f"{RES_DIR}/dedupe_eval.txt", "w").write("\n".join(lines) + "\n")
