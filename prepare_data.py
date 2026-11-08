"""Downloads the Telugu news classification dataset (IndicGLUE inltkh.te) to data/*.csv"""
import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
ds = None
for repo in ["indic_glue", "ai4bharat/indic_glue"]:
    try:
        ds = load_dataset(repo, "inltkh.te", trust_remote_code=True)
        print("Loaded from", repo)
        break
    except Exception as e:
        print(f"Failed {repo}: {str(e)[:200]}")

if ds is None:
    raise SystemExit(
        "\nCould not download automatically.\n"
        "Manual option: put your own train.csv, validation.csv, test.csv in data/ "
        "with columns 'text' and 'label' (integer labels), then continue."
    )

for split in ["train", "validation", "test"]:
    df = ds[split].to_pandas()
    df = df[["text", "label"]]
    df.to_csv(f"data/{split}.csv", index=False)
    print(split, len(df))
print("Labels:", ds["train"].features["label"].names)
print(df.head())
