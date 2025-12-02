#!/usr/bin/env python3
"""
Recompute Integrated Gradients attributions (n_steps=50) for all models
listed in `outputs/batch_check/batch_check_report.json` and save JSONL files.

This script mirrors the IG loader approach used elsewhere and overwrites
the existing attribution files with higher-fidelity results.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parents[1]
BATCH_REPORT = ROOT / "outputs" / "batch_check" / "batch_check_report.json"
ATTR_DIR = ROOT / "outputs" / "batch_check" / "attributions"

N_STEPS = 50
NUM_SAMPLES = 20

def load_token_attr_module():
    ta_path = str(ROOT / "src" / "explainability" / "token_attribution.py")
    return SourceFileLoader("token_attribution", ta_path).load_module()

def main():
    if not BATCH_REPORT.exists():
        print("Batch report not found:", BATCH_REPORT)
        return

    with open(BATCH_REPORT, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    models = data.get('models', [])
    if not models:
        print('No models in batch report')
        return

    ta_mod = load_token_attr_module()
    explain = ta_mod.explain_tokens_with_ig

    for m in models:
        model_path = Path(m['model_path'])
        name = m['name']
        csv_path = ROOT / m.get('results_csv', f'outputs/batch_check/{name}_results.csv')
        out_path = ATTR_DIR / f'{name}_attributions.jsonl'

        if not csv_path.exists():
            print('Results CSV missing for', name, '; skipping')
            continue
        if not model_path.exists():
            print('Model path missing for', name, '; skipping')
            continue

        print(f'Recomputing attributions for {name} (n_steps={N_STEPS})')

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()

        import pandas as pd
        df = pd.read_csv(csv_path)
        samples = df.head(NUM_SAMPLES)

        out_lines = []
        for _, row in samples.iterrows():
            sid = int(row['id'])
            text = str(row['text'])
            pred = int(row['prediction_label'])
            try:
                toks = explain(model=model, tokenizer=tokenizer, text=text, prediction=pred, device=device, n_steps=N_STEPS)
            except Exception as e:
                print(f'Error for {name} id {sid}: {e}')
                toks = []
            out_lines.append({'id': sid, 'tokens': toks})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as fh:
            for item in out_lines:
                fh.write(json.dumps(item, ensure_ascii=False) + '\n')

        print('Wrote', out_path)

if __name__ == '__main__':
    main()
