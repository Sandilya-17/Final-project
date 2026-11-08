# Telugu News Classification: Multilingual Transformers

## Run on Mac Terminal
cd telugu-nlp
bash start.sh          # does everything automatically (setup + data + all runs + plots)

Or step by step:
source venv/bin/activate
python prepare_data.py
python baseline.py
python finetune.py xlmr 1.0      # one model, one data fraction
python analyze.py

## Notes
- Apple Silicon (M1/M2/M3) uses the GPU automatically via MPS. Intel Macs run on CPU (very slow; use Google Colab for finetune.py).
- Expect roughly 1-4 hours total on Apple Silicon. You can stop and re-run run_all.sh; finished runs are skipped.
- If you get out-of-memory errors, lower per_device_train_batch_size to 8 in finetune.py.
- Outputs are in results/: summary.csv, learning_curve.png, confusion_matrix.png, errors_to_analyze.csv.
- For a stronger paper, repeat 1.0 runs with seeds: python finetune.py xlmr 1.0 1 (and 2, 3).
