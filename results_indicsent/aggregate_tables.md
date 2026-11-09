# Aggregate results (indicsent); macro-F1 mean ± std over seeds (n = number of seeds)

### Full data, default settings

| Model | Accuracy | Macro-F1 | Per-seed range |
|---|---|---|---|
| mbert | 0.876 ± 0.011 (n=3) | 0.876 ± 0.011 (n=3) | 0.866–0.887 |
| xlmr | 0.898 ± 0.046 (n=3) | 0.897 ± 0.047 (n=3) | 0.843–0.926 |
| indicbert | 0.968 ± 0.002 (n=3) | 0.968 ± 0.003 (n=3) | 0.965–0.970 |
| muril | 0.944 ± 0.007 (n=3) | 0.944 ± 0.008 (n=3) | 0.935–0.948 |
| telugubert | 0.957 ± 0.011 (n=3) | 0.957 ± 0.011 (n=3) | 0.948–0.970 |

### Training-data size (default settings)

| Model | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| mbert | n/a | n/a | n/a | 0.876 ± 0.011 (n=3) |
| xlmr | n/a | n/a | n/a | 0.897 ± 0.047 (n=3) |
| indicbert | n/a | n/a | n/a | 0.968 ± 0.003 (n=3) |
| muril | n/a | n/a | n/a | 0.944 ± 0.008 (n=3) |
| telugubert | n/a | n/a | n/a | 0.957 ± 0.011 (n=3) |

### Learning rate, full data, 4 epochs, no warmup

| Model | 2e-05 | 3e-05 | 5e-05 |
|---|---|---|---|
| mbert | 0.876 ± 0.011 (n=3) | n/a | n/a |
| xlmr | 0.897 ± 0.047 (n=3) | n/a | n/a |
| indicbert | 0.968 ± 0.003 (n=3) | n/a | n/a |
| muril | 0.944 ± 0.008 (n=3) | n/a | n/a |
| telugubert | 0.957 ± 0.011 (n=3) | n/a | n/a |

### 10% of the data: default vs longer training with a higher learning rate

| Model | 2e-5, 4 ep | 5e-5, 10 ep |
|---|---|---|
| mbert | n/a | n/a |
| xlmr | n/a | n/a |
| indicbert | n/a | n/a |
| muril | n/a | n/a |
| telugubert | n/a | n/a |

### Ablations, full data (does warmup prevent the 5e-5 collapse? do trainable embeddings matter?)

| Model | default | warmup 0.1 @2e-5 | 5e-5, no warmup | 5e-5, warmup 0.1 | embeddings trainable |
|---|---|---|---|---|---|
| mbert | 0.876 ± 0.011 (n=3) | n/a | n/a | n/a | n/a |
| xlmr | 0.897 ± 0.047 (n=3) | n/a | n/a | n/a | n/a |
| indicbert | 0.968 ± 0.003 (n=3) | n/a | n/a | n/a | n/a |
| muril | 0.944 ± 0.008 (n=3) | n/a | n/a | n/a | n/a |
| telugubert | 0.957 ± 0.011 (n=3) | n/a | n/a | n/a | n/a |
