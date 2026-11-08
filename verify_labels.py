"""Checks what the label ids 7 / 8 / 9 of IndicGLUE inltkh.te really mean.

The paper takes the class names (positive, negative, neutral) from the Hugging Face label list and could not
confirm them. This script matches the sentences of data/*.csv against a reference copy that has its own labels
and prints a table: inltkh id  x  reference label.

  python verify_labels.py                       # reference = IndicGLUE actsa-sc.te (same split sizes as inltkh.te)
  python verify_labels.py --ref-csv actsa.csv [--text-col text] [--label-col label]   # original ACTSA files

It also counts duplicate sentences inside inltkh.te (train/test overlap and conflicting labels).
Writes results/label_verification.txt. Needs a network connection for the default reference.
"""
import argparse, unicodedata
import pandas as pd


def norm(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).split())


def load_local(data_dir="data"):
    return {s: pd.read_csv(f"{data_dir}/{s}.csv") for s in ["train", "validation", "test"]}


def load_reference_hf():
    from datasets import load_dataset
    for repo in ["indic_glue", "ai4bharat/indic_glue"]:
        try:
            ds = load_dataset(repo, "actsa-sc.te", trust_remote_code=True)
            names = ds["train"].features["label"].names
            frames = []
            for s in ds:
                d = ds[s].to_pandas()[["text", "label"]]
                d["label"] = d["label"].map(lambda i: names[i])
                d["split"] = s
                frames.append(d)
            return pd.concat(frames, ignore_index=True), f"{repo}/actsa-sc.te (label names: {names})"
        except Exception as e:
            print(f"Failed {repo}: {str(e)[:150]}")
    raise SystemExit("Could not load actsa-sc.te. Give the original corpus with --ref-csv.")


def compare(local, ref):
    """local: dict split -> DataFrame(text,label); ref: DataFrame(text,label[,split]). Returns report text."""
    lines = []
    L = pd.concat([d.assign(split=s) for s, d in local.items()], ignore_index=True)
    L["key"], ref = L.text.map(norm), ref.assign(key=ref.text.map(norm))

    # same-position check (same sizes and order would mean the two files are the same corpus)
    if "split" in ref:
        for s, d in local.items():
            r = ref[ref.split == s]
            if len(r) == len(d):
                same = (d.text.map(norm).values == r.text.map(norm).values).mean()
                lines.append(f"split {s}: {len(d)} rows, identical sentence at the same position: {same:.1%}")
            else:
                lines.append(f"split {s}: local {len(d)} rows vs reference {len(r)} rows")

    # drop sentences whose label is not unique inside either table, then join on text
    def unique_labels(t):
        g = t.groupby("key").label.nunique()
        return t[t.key.isin(g[g == 1].index)].drop_duplicates("key")[["key", "label"]]
    j = unique_labels(L).merge(unique_labels(ref), on="key", suffixes=("_local", "_ref"))
    lines.append(f"\nSentences matched by text: {len(j)} of {L.key.nunique()} distinct local sentences")
    if len(j) == 0:
        lines.append("No overlap, so the labels cannot be verified with this reference.")
        return "\n".join(lines), None
    ct = pd.crosstab(j.label_local, j.label_ref)
    lines.append("\nCounts (rows: inltkh label id, columns: reference label)\n" + ct.to_string())
    lines.append("\nRow shares\n" + ct.div(ct.sum(1), axis=0).round(3).to_string())
    unmatched = sorted(set(L.label) - set(ct.index))
    for lid in ct.index:
        top = ct.loc[lid].idxmax()
        lines.append(f"id {lid}: {ct.loc[lid].max() / ct.loc[lid].sum():.1%} of matched sentences are '{top}' in the reference")
    if unmatched:
        lines.append(f"ids with no matched sentences: {unmatched}")
    return "\n".join(lines), ct


def duplicates(local):
    tr, va, te = (local[s].assign(key=local[s].text.map(norm)) for s in ["train", "validation", "test"])
    lines = ["\nDuplicate check inside inltkh.te"]
    lines.append(f"train sentences that also occur in test: {tr.key.isin(te.key).sum()} rows, "
                 f"{tr.key[tr.key.isin(te.key)].nunique()} distinct")
    lines.append(f"train sentences that also occur in validation: {tr.key.isin(va.key).sum()} rows")
    allr = pd.concat([tr, va, te])
    g = allr.groupby("key").label.nunique()
    lines.append(f"distinct sentences with conflicting labels: {(g > 1).sum()}")
    lines.append(f"duplicate rows overall: {allr.duplicated('key').sum()} of {len(allr)}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv"); ap.add_argument("--text-col", default="text"); ap.add_argument("--label-col", default="label")
    a = ap.parse_args()
    local = load_local()
    if a.ref_csv:
        sep = "\t" if a.ref_csv.endswith((".tsv", ".txt")) else ","
        ref = pd.read_csv(a.ref_csv, sep=sep).rename(columns={a.text_col: "text", a.label_col: "label"})[["text", "label"]]
        ref["label"], src = ref.label.astype(str), a.ref_csv
    else:
        ref, src = load_reference_hf()
        ref["label"] = ref.label.astype(str)
    local = {s: d.assign(label=d.label.astype(str)) for s, d in local.items()}
    body, _ = compare(local, ref)
    text = f"Reference: {src}\n{body}\n{duplicates(local)}\n"
    text += ("\nHow to read this: if each inltkh id lines up (almost) one-to-one with one reference label, the mapping is "
             "confirmed for that id. If the ids mix several reference labels, or the mapping differs from "
             "7=positive, 8=negative, 9=neutral, fix class_names.json and the paper's class names.\n")
    print(text)
    open("results/label_verification.txt", "w").write(text)
