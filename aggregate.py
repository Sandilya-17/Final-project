"""Mean +/- std over seeds for every setting, as paper-ready tables and plots.

  python aggregate.py                     # original dataset (results/)
  DATASET=indicsent python aggregate.py   # second dataset  (results_indicsent/)

Writes <results>/aggregate.csv, <results>/aggregate_tables.md, learning_curve_seeds.png,
lr_sensitivity_seeds.png. The std uses n-1 and is only a rough estimate for 3 seeds; min-max is in the csv.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ALL_MODELS, RES_DIR, DATASET, DEFAULTS, load_results

df = load_results()
df = df[df.model.isin(ALL_MODELS)]
if df.empty:
    raise SystemExit(f"No transformer results in {RES_DIR}/ yet.")
KEYS = ["model", "frac", "lr", "epochs", "warmup", "emb_frozen"]
sd = lambda x: x.std(ddof=1) if len(x) > 1 else np.nan
agg = (df.groupby(KEYS).agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", sd), f1_min=("macro_f1", "min"),
                            f1_max=("macro_f1", "max"), acc_mean=("acc", "mean"), acc_std=("acc", sd),
                            n_seeds=("macro_f1", "count"), seeds=("seed", lambda s: " ".join(str(int(v)) for v in sorted(s))))
       .reset_index())
agg.to_csv(f"{RES_DIR}/aggregate.csv", index=False)
models = [m for m in ALL_MODELS if m in set(agg.model)]

def get(m, frac=1.0, lr=DEFAULTS["lr"], epochs=DEFAULTS["epochs"], warmup=0.0, frozen=True):
    r = agg[(agg.model == m) & np.isclose(agg.frac, frac) & np.isclose(agg.lr, lr) & (agg.epochs == epochs)
            & np.isclose(agg.warmup, warmup) & (agg.emb_frozen == frozen)]
    return r.iloc[0] if len(r) else None

def cell(r, col="f1"):
    if r is None: return "n/a"
    mean, std = r[f"{col}_mean"], r[f"{col}_std"]
    return f"{mean:.3f} (n=1)" if r.n_seeds == 1 else f"{mean:.3f} ± {std:.3f} (n={int(r.n_seeds)})"

def md(title, header, rows):
    out = [f"\n### {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)

parts = [f"# Aggregate results ({DATASET}); macro-F1 mean ± std over seeds (n = number of seeds)"]
parts.append(md("Full data, default settings", ["Model", "Accuracy", "Macro-F1", "Per-seed range"],
    [[m, cell(get(m), "acc"), cell(get(m)), (f"{get(m).f1_min:.3f}–{get(m).f1_max:.3f}" if get(m) is not None else "n/a")]
     for m in models]))
fracs = [0.1, 0.25, 0.5, 1.0]
parts.append(md("Training-data size (default settings)", ["Model"] + [f"{int(f*100)}%" for f in fracs],
    [[m] + [cell(get(m, frac=f)) for f in fracs] for m in models]))
lrs = [2e-5, 3e-5, 5e-5]
parts.append(md("Learning rate, full data, 4 epochs, no warmup", ["Model"] + [f"{lr:g}" for lr in lrs],
    [[m] + [cell(get(m, lr=lr)) for lr in lrs] for m in models]))
parts.append(md("10% of the data: default vs longer training with a higher learning rate",
    ["Model", "2e-5, 4 ep", "5e-5, 10 ep"],
    [[m, cell(get(m, frac=0.1)), cell(get(m, frac=0.1, lr=5e-5, epochs=10))] for m in models]))
parts.append(md("Ablations, full data (does warmup prevent the 5e-5 collapse? do trainable embeddings matter?)",
    ["Model", "default", "warmup 0.1 @2e-5", "5e-5, no warmup", "5e-5, warmup 0.1", "embeddings trainable"],
    [[m, cell(get(m)), cell(get(m, warmup=0.1)), cell(get(m, lr=5e-5)), cell(get(m, lr=5e-5, warmup=0.1)),
      cell(get(m, frozen=False))] for m in models]))
open(f"{RES_DIR}/aggregate_tables.md", "w").write("\n".join(parts) + "\n")
print("\n".join(parts))

# plot 1: learning curve with error bars
plt.figure(figsize=(6.4, 4.2))
for m in models:
    pts = [(f * 100, r.f1_mean, 0 if np.isnan(r.f1_std) else r.f1_std) for f in fracs if (r := get(m, frac=f)) is not None]
    if pts:
        x, y, e = zip(*pts); plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=m)
plt.xlabel("Training data (%)"); plt.ylabel("Test macro-F1 (mean ± std over seeds)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(f"{RES_DIR}/learning_curve_seeds.png", dpi=200)

# plot 2: learning-rate sensitivity with error bars
plt.figure(figsize=(7, 4.2)); w = 0.8 / len(models)
for i, m in enumerate(models):
    rs = [get(m, lr=lr) for lr in lrs]
    plt.bar(np.arange(3) + i * w, [r.f1_mean if r is not None else 0 for r in rs], w,
            yerr=[0 if r is None or np.isnan(r.f1_std) else r.f1_std for r in rs], capsize=2, label=m)
plt.xticks(np.arange(3) + w * (len(models) - 1) / 2, [f"{lr:g}" for lr in lrs])
plt.xlabel("Learning rate"); plt.ylabel("Test macro-F1 (mean ± std)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RES_DIR}/lr_sensitivity_seeds.png", dpi=200)
print(f"\nWrote {RES_DIR}/aggregate.csv, aggregate_tables.md, learning_curve_seeds.png, lr_sensitivity_seeds.png")
