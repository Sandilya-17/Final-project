"""Shared settings for all scripts.

DATASET=inltkh (the default) keeps the original data/ and results/ folders, so existing
results and commands keep working. Any other name, e.g. DATASET=indicsent, uses
data_<name>/ and results_<name>/.
"""
import json
import os

DATASET = os.environ.get("DATASET", "inltkh")
DATA_DIR = "data" if DATASET == "inltkh" else f"data_{DATASET}"
RES_DIR = "results" if DATASET == "inltkh" else f"results_{DATASET}"

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "muril": "google/muril-base-cased",         # Indian-language BERT (Google)
    "telugubert": "l3cube-pune/telugu-bert",    # Telugu-only BERT (L3Cube)
}
ORIGINAL_MODELS = ["mbert", "xlmr", "indicbert"]
ALL_MODELS = list(MODELS)

# The settings used for every main table in the paper.
DEFAULTS = {"lr": 2e-5, "epochs": 4, "warmup": 0.0, "emb_frozen": True, "seed": 42}


def class_names(n):
    """Class names for ids 0..n-1 (from class_names.json if present)."""
    path = "class_names.json" if DATASET == "inltkh" else os.path.join(DATA_DIR, "class_names.json")
    names = {str(i): f"C{i}" for i in range(n)}
    if os.path.exists(path):
        names.update(json.load(open(path)))
    return [names[str(i)] for i in range(n)]


def load_results(glob_dir=None):
    """All result JSONs of the current dataset as a DataFrame with the settings columns filled in."""
    import glob
    import pandas as pd
    rows = [json.load(open(f)) for f in glob.glob(f"{glob_dir or RES_DIR}/*.json")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col, default in DEFAULTS.items():
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["lr"] = df["lr"].astype(float).round(8)
    df["warmup"] = df["warmup"].astype(float).round(4)
    df["emb_frozen"] = df["emb_frozen"].astype(bool)
    return df
