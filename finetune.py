"""Usage: python finetune.py <mbert|xlmr|indicbert|muril|telugubert> <fraction> [seed]

Optional environment variables (all off by default, so the original protocol is unchanged):
  LR=5e-5 EPOCHS=10   learning rate / epochs         (tag gets _lr..._ep...)
  BATCH=4             per-device batch size; gradient accumulation is raised so the
                      effective batch size stays 16
  WARMUP=0.1          linear warmup over this share of all optimizer steps   (tag gets _wu0.1)
  UNFREEZE_EMB=1      also train the word-embedding layer                     (tag gets _unfrz)
  SKIP_DONE=1         exit immediately if the result file already exists
  DATASET=name        use data_<name>/ and results_<name>/  (default: inltkh -> data/, results/)
"""
import sys, os, json, math, numpy as np, pandas as pd, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from common import MODELS, DATASET, DATA_DIR, RES_DIR

name, frac = sys.argv[1], float(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
lr = float(os.environ.get("LR", "2e-5"))
epochs = int(os.environ.get("EPOCHS", "4"))
batch = int(os.environ.get("BATCH", "8"))
warmup = float(os.environ.get("WARMUP", "0"))
unfreeze = os.environ.get("UNFREEZE_EMB", "0") == "1"
grad_acc = max(1, 16 // batch)            # effective batch size 16 for BATCH = 16, 8, 4, 2, 1

tag = f"{name}_{frac}" + (f"_s{seed}" if seed != 42 else "")
if "LR" in os.environ or "EPOCHS" in os.environ:
    tag += f"_lr{os.environ.get('LR', '2e-5')}_ep{epochs}"
if warmup > 0:
    tag += f"_wu{warmup:g}"
if unfreeze:
    tag += "_unfrz"

os.makedirs(RES_DIR, exist_ok=True)
if os.environ.get("SKIP_DONE") == "1" and os.path.exists(f"{RES_DIR}/{tag}.json"):
    print(f"skip {DATASET}/{tag} (already done)")
    sys.exit(0)

from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

tr = pd.read_csv(f"{DATA_DIR}/train.csv").sample(frac=frac, random_state=seed)
va, te = pd.read_csv(f"{DATA_DIR}/validation.csv"), pd.read_csv(f"{DATA_DIR}/test.csv")
# remap labels to 0..K-1 (the Telugu IndicGLUE subset uses ids like 7,8,9)
classes = sorted(pd.read_csv(f"{DATA_DIR}/train.csv").label.unique())
lab2id = {l: i for i, l in enumerate(classes)}
for d in (tr, va, te):
    d["label"] = d["label"].map(lab2id)
n_labels = len(classes)
print("Dataset:", DATASET, "| Classes:", classes, "->", list(range(n_labels)))

tok = AutoTokenizer.from_pretrained(MODELS[name])
def enc(df):
    d = Dataset.from_pandas(df[["text", "label"]].astype({"text": str}).reset_index(drop=True))
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODELS[name], num_labels=n_labels)

if not unfreeze:   # original protocol: the word-embedding layer is frozen to save memory
    for prm in model.base_model.embeddings.parameters():
        prm.requires_grad = False

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}

total_steps = math.ceil(len(tr) / (batch * grad_acc)) * epochs
warmup_steps = int(round(warmup * total_steps))
print(f"optimizer steps: {total_steps}, warmup steps: {warmup_steps}, embeddings {'trainable' if unfreeze else 'frozen'}")

args = TrainingArguments(output_dir=f"{RES_DIR}/tmp_{tag}", num_train_epochs=epochs,
    per_device_train_batch_size=batch, per_device_eval_batch_size=16,
    gradient_accumulation_steps=grad_acc, learning_rate=lr, warmup_steps=warmup_steps,
    eval_strategy="epoch", save_strategy="no", seed=seed,
    fp16=torch.cuda.is_available(), report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=enc(tr), eval_dataset=enc(va),
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
out = trainer.predict(enc(te))
preds = np.argmax(out.predictions, axis=1)
val = trainer.evaluate()   # validation score, used to choose settings
res = {"model": name, "frac": frac, "seed": seed, "lr": lr, "epochs": epochs,
       "warmup": warmup, "emb_frozen": not unfreeze, "dataset": DATASET,
       "val_macro_f1": val["eval_macro_f1"], **metrics(out)}
print(res)
json.dump(res, open(f"{RES_DIR}/{tag}.json", "w"))
pd.DataFrame({"text": te.text, "true": te.label, "pred": preds}).to_csv(f"{RES_DIR}/{tag}_preds.csv", index=False)
