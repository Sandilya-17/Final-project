"""Usage: python finetune.py <mbert|xlmr|indicbert> <fraction> [seed]"""
import sys, os, json, numpy as np, pandas as pd, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
}
name, frac = sys.argv[1], float(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
tag = f"{name}_{frac}" + (f"_s{seed}" if seed != 42 else "")

tr = pd.read_csv("data/train.csv").sample(frac=frac, random_state=seed)
va, te = pd.read_csv("data/validation.csv"), pd.read_csv("data/test.csv")
# remap labels to 0..K-1 (this Telugu subset uses ids like 7,8,9)
classes = sorted(pd.read_csv("data/train.csv").label.unique())
lab2id = {l: i for i, l in enumerate(classes)}
for d in (tr, va, te):
    d["label"] = d["label"].map(lab2id)
n_labels = len(classes)
print("Classes:", classes, "->", list(range(n_labels)))

tok = AutoTokenizer.from_pretrained(MODELS[name])
def enc(df):
    d = Dataset.from_pandas(df[["text", "label"]].astype({"text": str}).reset_index(drop=True))
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODELS[name], num_labels=n_labels)

for prm in model.base_model.embeddings.parameters():
    prm.requires_grad = False

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(output_dir=f"results/tmp_{tag}", num_train_epochs=4,
    per_device_train_batch_size=int(os.environ.get('BATCH','8')), per_device_eval_batch_size=16, gradient_accumulation_steps=2, learning_rate=2e-5,
    eval_strategy="epoch", save_strategy="no", seed=seed,
    fp16=torch.cuda.is_available(), report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=enc(tr), eval_dataset=enc(va),
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
out = trainer.predict(enc(te))
preds = np.argmax(out.predictions, axis=1)
res = {"model": name, "frac": frac, "seed": seed, **metrics(out)}
print(res)
json.dump(res, open(f"results/{tag}.json", "w"))
pd.DataFrame({"text": te.text, "true": te.label, "pred": preds}).to_csv(f"results/{tag}_preds.csv", index=False)
