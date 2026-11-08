# Aggregate results (inltkh); macro-F1 mean ± std over seeds (n = number of seeds)

### Full data, default settings

| Model | Accuracy | Macro-F1 | Per-seed range |
|---|---|---|---|
| mbert | 0.601 ± 0.007 (n=3) | 0.583 ± 0.006 (n=3) | 0.579–0.591 |
| xlmr | 0.659 ± 0.013 (n=3) | 0.654 ± 0.013 (n=3) | 0.643–0.669 |
| indicbert | 0.680 ± 0.018 (n=3) | 0.679 ± 0.017 (n=3) | 0.666–0.699 |
| muril | 0.614 ± 0.125 (n=3) | 0.526 ± 0.271 (n=3) | 0.214–0.695 |
| telugubert | 0.691 ± 0.013 (n=3) | 0.689 ± 0.013 (n=3) | 0.682–0.704 |

### Training-data size (default settings)

| Model | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| mbert | 0.301 ± 0.124 (n=2) | 0.533 ± 0.006 (n=2) | 0.568 ± 0.034 (n=2) | 0.583 ± 0.006 (n=3) |
| xlmr | 0.267 ± 0.092 (n=3) | 0.546 ± 0.012 (n=3) | 0.636 ± 0.015 (n=3) | 0.654 ± 0.013 (n=3) |
| indicbert | 0.214 (n=1) | 0.577 (n=1) | 0.663 (n=1) | 0.679 ± 0.017 (n=3) |
| muril | n/a | n/a | n/a | 0.526 ± 0.271 (n=3) |
| telugubert | n/a | n/a | n/a | 0.689 ± 0.013 (n=3) |

### Learning rate, full data, 4 epochs, no warmup

| Model | 2e-05 | 3e-05 | 5e-05 |
|---|---|---|---|
| mbert | 0.583 ± 0.006 (n=3) | 0.575 ± 0.006 (n=2) | 0.377 ± 0.231 (n=2) |
| xlmr | 0.654 ± 0.013 (n=3) | 0.559 ± 0.177 (n=3) | 0.353 ± 0.242 (n=3) |
| indicbert | 0.679 ± 0.017 (n=3) | 0.675 (n=1) | 0.653 (n=1) |
| muril | 0.526 ± 0.271 (n=3) | n/a | n/a |
| telugubert | 0.689 ± 0.013 (n=3) | n/a | n/a |

### 10% of the data: default vs longer training with a higher learning rate

| Model | 2e-5, 4 ep | 5e-5, 10 ep |
|---|---|---|
| mbert | 0.301 ± 0.124 (n=2) | 0.461 (n=1) |
| xlmr | 0.267 ± 0.092 (n=3) | 0.364 ± 0.151 (n=3) |
| indicbert | 0.214 (n=1) | 0.624 (n=1) |
| muril | n/a | n/a |
| telugubert | n/a | n/a |

### Ablations, full data (does warmup prevent the 5e-5 collapse? do trainable embeddings matter?)

| Model | default | warmup 0.1 @2e-5 | 5e-5, no warmup | 5e-5, warmup 0.1 | embeddings trainable |
|---|---|---|---|---|---|
| mbert | 0.583 ± 0.006 (n=3) | 0.588 ± 0.011 (n=3) | 0.377 ± 0.231 (n=2) | 0.578 ± 0.037 (n=3) | n/a |
| xlmr | 0.654 ± 0.013 (n=3) | 0.621 ± 0.021 (n=3) | 0.353 ± 0.242 (n=3) | 0.214 ± 0.000 (n=3) | n/a |
| indicbert | 0.679 ± 0.017 (n=3) | 0.670 ± 0.001 (n=3) | 0.653 (n=1) | 0.665 ± 0.015 (n=3) | n/a |
| muril | 0.526 ± 0.271 (n=3) | n/a | n/a | n/a | n/a |
| telugubert | 0.689 ± 0.013 (n=3) | n/a | n/a | n/a | n/a |
