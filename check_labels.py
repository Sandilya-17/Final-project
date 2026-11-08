"""Prints label counts and 10 random training sentences per label, so you can decide what each class means."""
import pandas as pd
d = pd.read_csv("data/train.csv")
print(d.label.value_counts().sort_index(), "\n")
print("Label ids in the CSV are 7, 8, 9. Models use 0, 1, 2 in the same order (7->0, 8->1, 9->2).\n")
for i, l in enumerate(sorted(d.label.unique())):
    print(f"=== CSV label {l}  (class id {i}) ===")
    for t in d[d.label == l].text.sample(10, random_state=1):
        print("  ", t[:120])
    print()
print("Now create class_names.json, for example:")
print('  echo \'{"0":"<name for 7>","1":"<name for 8>","2":"<name for 9>"}\' > class_names.json')
