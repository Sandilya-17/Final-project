import json, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
Xtr, Xte = vec.fit_transform(tr.text.astype(str)), vec.transform(te.text.astype(str))

for name, clf in [("logreg", LogisticRegression(max_iter=1000)), ("linear_svm", LinearSVC())]:
    p = clf.fit(Xtr, tr.label).predict(Xte)
    res = {"model": name, "frac": 1.0,
           "acc": accuracy_score(te.label, p),
           "macro_f1": f1_score(te.label, p, average="macro")}
    print(res)
    json.dump(res, open(f"results/{name}_1.0.json", "w"))
