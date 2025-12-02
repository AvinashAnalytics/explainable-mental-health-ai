#!/usr/bin/env python3
"""
Compute bootstrap 95% CIs for accuracy and macro-F1 using per-model results CSVs
listed in outputs/batch_check/batch_check_report.json.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs' / 'batch_check' / 'batch_check_report.json'
N_BOOT = 1000
SEED = 42

if not REPORT.exists():
    print('Batch report not found at', REPORT)
    raise SystemExit(1)

with open(REPORT, 'r', encoding='utf-8') as fh:
    data = json.load(fh)

rng = np.random.default_rng(SEED)

for m in data.get('models', []):
    name = m.get('name')
    csv_path = ROOT / m.get('results_csv', f'outputs/batch_check/{name}_results.csv')
    if not csv_path.exists():
        print('Results CSV missing for', name, 'at', csv_path)
        continue

    df = pd.read_csv(csv_path)
    if 'true_label' not in df.columns and 'label' not in df.columns:
        # try columns with numeric labels
        if 'prediction_label' in df.columns:
            print(f"No true labels for {name}; skipping bootstrap.")
            continue
        else:
            print(f"No label columns found for {name}; skipping.")
            continue

    # prefer provided 'label' or 'true_label'
    if 'label' in df.columns:
        y_true_all = df['label'].astype(float)
    else:
        # try mapping textual labels first, then numeric conversion
        mapped = df['true_label'].map({'Control': 0, 'Depression': 1})
        if mapped.isna().all():
            # try numeric conversion
            mapped = pd.to_numeric(df['true_label'], errors='coerce')
        y_true_all = mapped.astype(float)

    y_pred_all = pd.to_numeric(df['prediction_label'], errors='coerce').astype(float)

    # Drop rows where true or pred is NaN (incomplete data)
    mask = (~y_true_all.isna()) & (~y_pred_all.isna())
    if mask.sum() == 0:
        print(f'No valid labeled rows for {name}; skipping bootstrap.')
        continue

    y_true = y_true_all[mask].astype(int).to_numpy()
    y_pred = y_pred_all[mask].astype(int).to_numpy()
    n = len(y_true)

    accs = []
    f1s = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0))
    acc_mean = float(np.mean(accs))
    acc_lo = float(np.percentile(accs, 2.5))
    acc_hi = float(np.percentile(accs, 97.5))
    f1_mean = float(np.mean(f1s))
    f1_lo = float(np.percentile(f1s, 2.5))
    f1_hi = float(np.percentile(f1s, 97.5))

    print('Model:', name)
    print(' Samples:', n)
    print(f' Accuracy mean: {acc_mean:.4f}, 95% CI: [{acc_lo:.4f}, {acc_hi:.4f}]')
    print(f' Macro-F1 mean: {f1_mean:.4f}, 95% CI: [{f1_lo:.4f}, {f1_hi:.4f}]')
    print('-'*60)
