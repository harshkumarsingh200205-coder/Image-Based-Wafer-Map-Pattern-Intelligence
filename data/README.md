# Wafer Map Datasets Directory

This directory stores raw, processed, and partitioned wafer map datasets.

## Subdirectories
- `raw/`: Place original wafer map datasets here (e.g. WM-811K `.pkl` or `.mat` files, or synthetic wafer datasets).
- `processed/`: Processed, cropped, and normalized wafer arrays.
- `splits/`: Stratified split manifests (`train.csv`, `val.csv`, `test.csv`) to guarantee zero data leakage across runs.

## Important Data Handling Rules
1. Never commit raw binary dataset archives to Git (`.gitignore` protects `data/raw/` and `data/processed/`).
2. Always fix random seeds (`seed=42`) when generating splits.
3. If multiple images originate from the same wafer lot, group them by `lot_id` during splitting to prevent data leakage.
