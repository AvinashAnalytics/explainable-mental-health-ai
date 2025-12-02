#!/usr/bin/env python3
"""
Create a curated 'best' test set and evaluate it across trained models.
Strategy: pick from disagreement, low-confidence, misclassified, stratified.
Writes outputs to outputs/best_test_sets/ and outputs/best_test_sets/eval_results.json
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
BATCH_REPORT = ROOT / 'outputs' / 'batch_check' / 'batch_check_report.json'
OUT_DIR = ROOT / 'outputs' / 'best_test_sets'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
PICK_PER_STRAT = 50  # number of samples per strategy
TOTAL_TARGET = 200
SELECTED_MODEL = None  # if None, use first model in report
RANDOM_SEED = 42

if not BATCH_REPORT.exists():
    print('No batch report found at', BATCH_REPORT)
    raise SystemExit(1)

with open(BATCH_REPORT, 'r', encoding='utf-8') as fh:
    report = json.load(fh)

models = report.get('models', [])
if not models:
    print('No models listed in batch report')
    raise SystemExit(1)

# Load all result CSVs and merge by id
dfs = {}
for m in models:
    name = m.get('name')
    csvp = ROOT / m.get('results_csv', '')
    if csvp.exists():
        df = pd.read_csv(csvp)
        # normalize columns: ensure id, text, prediction_label, confidence, true_label or label
        if 'id' not in df.columns:
            df['id'] = range(1, len(df)+1)
        dfs[name] = df
    else:
        print('Warning: results CSV missing for', name)

# Choose base df: prefer a df that has text and id
base_name = next(iter(dfs.keys()))
base_df = dfs[base_name].copy()

# Merge predictions from all models
merged = base_df[['id','text']].copy()
for name, df in dfs.items():
    # Ensure id column exists
    temp = df[['id','prediction_label','confidence']].copy()
    temp = temp.rename(columns={'prediction_label': f'pred_{name}', 'confidence': f'conf_{name}'})
    merged = merged.merge(temp, on='id', how='left')

# If there is a label column in base df, preserve it
if 'label' in base_df.columns:
    merged['true_label'] = base_df['label']
elif 'true_label' in base_df.columns:
    merged['true_label'] = base_df['true_label'].map({'Control':0,'Depression':1})

# Strategy: disagreement across models
pred_cols = [c for c in merged.columns if c.startswith('pred_')]
if pred_cols:
    merged['disagreement'] = merged[pred_cols].apply(lambda row: len(set([v for v in row.values if pd.notna(v)])), axis=1)
else:
    merged['disagreement'] = 1

# Low confidence for selected model (choose first model if none)
if SELECTED_MODEL is None:
    SELECTED_MODEL = models[0].get('name')
sel_conf_col = f'conf_{SELECTED_MODEL}'
if sel_conf_col not in merged.columns:
    merged[sel_conf_col] = np.nan
# Convert confidence to numeric; earlier code had confidence in percent (0..100)
merged[sel_conf_col] = pd.to_numeric(merged[sel_conf_col], errors='coerce')
merged['abs50'] = (merged[sel_conf_col] - 50.0).abs()

# Misclassified by selected model (requires true_label)
if 'true_label' in merged.columns:
    merged['pred_sel'] = merged.get(f'pred_{SELECTED_MODEL}', np.nan)
    merged['misclassified'] = merged.apply(lambda r: 1 if (pd.notna(r['pred_sel']) and pd.notna(r['true_label']) and int(r['pred_sel']) != int(r['true_label'])) else 0, axis=1)
else:
    merged['misclassified'] = 0

# Stratified: we'll sample evenly across true_label if available
# Build candidate pools per strategy
candidates = []

# Disagreement top
cand_dis = merged.sort_values('disagreement', ascending=False).head(PICK_PER_STRAT)
candidates.append(cand_dis)

# Low confidence top (closest to 50)
cand_lowconf = merged.sort_values('abs50', ascending=True).head(PICK_PER_STRAT)
candidates.append(cand_lowconf)

# Misclassified
if merged['misclassified'].sum() > 0:
    cand_mis = merged[merged['misclassified']==1].head(PICK_PER_STRAT)
else:
    cand_mis = merged.head(0)
candidates.append(cand_mis)

# Stratified
if 'true_label' in merged.columns:
    classes = merged['true_label'].dropna().unique().tolist()
    parts = []
    if len(classes) > 0:
        per_class = max(1, PICK_PER_STRAT // max(1, len(classes)))
        for cls in classes:
            pool = merged[merged['true_label']==cls]
            if len(pool) == 0:
                continue
            parts.append(pool.sample(n=min(per_class, len(pool)), random_state=RANDOM_SEED))
    if parts:
        cand_strat = pd.concat(parts)
    else:
        cand_strat = merged.sample(n=min(PICK_PER_STRAT, len(merged)), random_state=RANDOM_SEED)
else:
    cand_strat = merged.sample(n=min(PICK_PER_STRAT, len(merged)), random_state=RANDOM_SEED)
candidates.append(cand_strat)

# Combine unique ids while preserving priority order
selected_ids = []
for c in candidates:
    for _id in c['id'].tolist():
        if _id not in selected_ids:
            selected_ids.append(_id)
        if len(selected_ids) >= TOTAL_TARGET:
            break
    if len(selected_ids) >= TOTAL_TARGET:
        break

# If still short, fill with random samples
if len(selected_ids) < TOTAL_TARGET:
    remaining = [i for i in merged['id'].tolist() if i not in selected_ids]
    rng = np.random.default_rng(RANDOM_SEED)
    fill = list(rng.choice(remaining, size=min(len(remaining), TOTAL_TARGET - len(selected_ids)), replace=False))
    selected_ids.extend(fill)

selected_df = merged.set_index('id').loc[selected_ids].reset_index()
selected_df = selected_df[['id','text']]
out_csv = OUT_DIR / f'best_test_set_{TOTAL_TARGET}.csv'
selected_df.to_csv(out_csv, index=False)
print('Wrote test set to', out_csv)

# Evaluate across models
results = {}
for m in models:
    name = m.get('name')
    model_path = Path(m.get('model_path', ''))
    if not model_path.exists():
        print('Model path missing for', name, 'skipping')
        continue
    print('Evaluating model', name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()

        texts = selected_df['text'].astype(str).tolist()
        batch_size = 16
        preds = []
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, truncation=True, padding=True, return_tensors='pt').to(device)
                outs = model(**inputs)
                logits = outs.logits if hasattr(outs, 'logits') else outs
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                preds.extend(np.argmax(p, axis=1).tolist())
                probs.extend(p.tolist())

        # If true labels exist in base df, compute metrics
        y_true = None
        if 'true_label' in merged.columns:
            # map selected ids to true labels
            id_to_label = merged.set_index('id')['true_label'].to_dict()
            y_true_raw = [id_to_label.get(i, -1) for i in selected_df['id'].tolist()]
            # convert NaN to -1 and ensure ints where possible
            y_true = []
            for v in y_true_raw:
                try:
                    if pd.isna(v):
                        y_true.append(-1)
                    else:
                        y_true.append(int(v))
                except Exception:
                    y_true.append(-1)
            # filter out -1
            mask = [yt != -1 for yt in y_true]
            if any(mask):
                y_true_f = [y for y,m in zip(y_true,mask) if m]
                y_pred_f = [p for p,m in zip(preds,mask) if m]
                acc = accuracy_score(y_true_f, y_pred_f)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true_f, y_pred_f, average='binary', zero_division=0)
                cm = confusion_matrix(y_true_f, y_pred_f).tolist()
            else:
                acc, prec, rec, f1, cm = None, None, None, None, None
        else:
            acc, prec, rec, f1, cm = None, None, None, None, None

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'n_samples': len(selected_df)
        }
    except Exception as e:
           import traceback
           tb = traceback.format_exc()
           print('Evaluation failed for', name, e)
           print(tb)
           results[name] = {'error': str(e), 'traceback': tb}

out_json = OUT_DIR / 'best_test_set_eval.json'
with open(out_json, 'w', encoding='utf-8') as fh:
    json.dump({'test_set': str(out_csv), 'results': results}, fh, indent=2)
print('Wrote evaluation results to', out_json)

# Also write a human-readable summary
print('\nSummary:')
for name, r in results.items():
    if 'error' in r:
        print(name, 'ERROR:', r['error'])
    else:
        print(name, 'Acc:', r['accuracy'], 'F1:', r['f1'])
