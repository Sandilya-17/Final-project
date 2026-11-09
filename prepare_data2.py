"""Prepares a second Telugu dataset (external validity) as data_<name>/{train,validation,test}.csv.

  python prepare_data2.py indicsentiment             # ai4bharat/IndicSentiment, Telugu (translated reviews, binary) -> data_indicsent/
  python prepare_data2.py csv my_file.csv mydata     # any CSV with columns 'text' and 'label'  -> data_mydata/

Then run everything on it with the DATASET variable, e.g.
  DATASET=indicsent python baseline.py && bash run_extended.sh dataset2
The split is stratified 70/10/20 with a fixed seed, because IndicSentiment ships without a training set.
"""
import json, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split


def save_splits(df, name, seed=0):
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df.text.astype(str).str.strip()
    df = df[df.text != ""].drop_duplicates("text")
    counts = df.label.astype(str).str.strip().str.lower().value_counts()   # drop missing / very rare labels
    bad = counts[(counts < 10) | counts.index.isin(["nan", "none", "null", ""])].index
    if len(bad):
        print("Dropping rare or invalid labels:", counts[bad].to_dict())
        df = df[~df.label.astype(str).str.strip().str.lower().isin(bad)]
    classes = sorted(df.label.astype(str).unique())
    df["label"] = df.label.astype(str).map({c: i for i, c in enumerate(classes)})
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=seed)
    val, test = train_test_split(rest, test_size=2 / 3, stratify=rest.label, random_state=seed)
    out = f"data_{name}"
    os.makedirs(out, exist_ok=True)
    for s, d in [("train", train), ("validation", val), ("test", test)]:
        d[["text", "label"]].to_csv(f"{out}/{s}.csv", index=False)
        print(s, len(d), d.label.value_counts().sort_index().to_dict())
    json.dump({str(i): c for i, c in enumerate(classes)}, open(f"{out}/class_names.json", "w"))
    print(f"Wrote {out}/ with classes {classes}")


def load_indicsentiment():
    from datasets import load_dataset
    ds, last = None, None
    for cfg in ["translation-te", "te"]:
        try:
            ds = load_dataset("ai4bharat/IndicSentiment", cfg, trust_remote_code=True)
            print("Loaded ai4bharat/IndicSentiment config", cfg, {s: len(ds[s]) for s in ds})
            break
        except Exception as e:
            last = e
            print(f"Failed config {cfg}: {str(e)[:200]}")
    if ds is None:
        raise SystemExit("Could not load IndicSentiment automatically. Download the Telugu data yourself and use\n"
                         "  python prepare_data2.py csv <file.csv> <name>\n" + str(last)[:300])
    df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)
    print("Columns:", list(df.columns))
    text_col = next((c for c in df.columns if "indic" in c.lower()), None) or next(
        (c for c in df.columns if c.lower() in ("text", "review")), None)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if not text_col or not label_col:
        raise SystemExit(f"Could not find the text/label columns in {list(df.columns)}; use the csv mode.")
    print(f"Using text column '{text_col}', label column '{label_col}'")
    return df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]].assign(
        label=lambda d: d.label.astype(str).str.strip().str.lower())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "indicsentiment"
    if mode == "indicsentiment":
        save_splits(load_indicsentiment(), "indicsent")
    elif mode == "csv" and len(sys.argv) >= 4:
        save_splits(pd.read_csv(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit(__doc__)
