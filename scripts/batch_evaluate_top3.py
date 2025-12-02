#!/usr/bin/env python3
"""
Batch evaluate top-3 models on data/temp.csv and compute token attributions

Saves per-model CSV results to `outputs/` and attributions to `outputs/attributions/`.
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ensure src is referenceable; load token_attribution module directly to avoid package-level imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load token_attribution as a standalone module to avoid `src` package import side-effects
from importlib.machinery import SourceFileLoader
ta_path = str(ROOT / "src" / "explainability" / "token_attribution.py")
token_attr_mod = SourceFileLoader("token_attribution", ta_path).load_module()
explain_tokens_with_ig = token_attr_mod.explain_tokens_with_ig


MODELS = [
    "models/trained/distilbert_20251126_234714",
    "models/trained/distilbert_cpu_20251127_002141",
    "models/trained/smoke_test_final",
]

DATA_PATH = Path("data/temp.csv")
OUTPUT_DIR = Path("outputs/batch_check")
ATTR_DIR = OUTPUT_DIR / "attributions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ATTR_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path):
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError('Input CSV must contain a `text` column')
    return df


def run_model_on_df(model_path: str, df: pd.DataFrame, max_attrib: int = 20, n_steps: int = 20):
    name = Path(model_path).name
    print(f"\n=== Evaluating model: {name} ({model_path}) ===")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    rows = []
    attributions = []

    for i, row in df.iterrows():
        text = str(row['text'])
        inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).cpu().item())
            conf = float(probs[0, pred].cpu().item())
            prob_control = float(probs[0, 0].cpu().item())
            prob_depression = float(probs[0, 1].cpu().item())

        row_out = {
            'id': row.get('id', i+1),
            'text': text,
            'prediction_label': pred,
            'prediction': 'Depression' if pred == 1 else 'Control',
            'confidence': conf,
            'prob_control': prob_control,
            'prob_depression': prob_depression,
        }

        if 'label' in row and not pd.isna(row['label']):
            row_out['true_label'] = int(row['label'])

        rows.append(row_out)

        # Compute attribution for first max_attrib rows
        if len(attributions) < max_attrib:
            try:
                toks = explain_tokens_with_ig(model, tokenizer, text, pred, device=device, n_steps=n_steps)
            except Exception as e:
                print(f"Attribution error for row {i}: {e}")
                toks = []
            attributions.append({'id': row_out['id'], 'tokens': toks})

    results_df = pd.DataFrame(rows)

    # Metrics if labels present
    metrics = {}
    if 'true_label' in results_df.columns:
        y_true = results_df['true_label'].astype(int).tolist()
        y_pred = results_df['prediction_label'].astype(int).tolist()
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'total_samples': len(df),
        }

    # Save outputs
    csv_path = OUTPUT_DIR / f"{name}_results.csv"
    results_df.to_csv(csv_path, index=False)

    attr_path = ATTR_DIR / f"{name}_attributions.jsonl"
    with open(attr_path, 'w', encoding='utf-8') as fh:
        for item in attributions:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Quick summary
    print(f"Saved results to {csv_path}")
    print(f"Saved attributions (first {max_attrib}) to {attr_path}")
    if metrics:
        print("Metrics:")
        print(json.dumps(metrics, indent=2))

    return dict(name=name, model_path=model_path, metrics=metrics, results_csv=str(csv_path), attributions=str(attr_path))


def main():
    df = load_data(DATA_PATH)

    summary = []
    for m in MODELS:
        try:
            res = run_model_on_df(m, df, max_attrib=20, n_steps=20)
            summary.append(res)
        except Exception as e:
            print(f"Error running model {m}: {e}")

    # Save combined report
    report_path = OUTPUT_DIR / "batch_check_report.json"
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump({'models': summary}, fh, indent=2)

    print(f"\nCombined report saved to {report_path}")


if __name__ == '__main__':
    main()
