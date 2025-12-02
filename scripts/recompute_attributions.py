#!/usr/bin/env python3
"""
Recompute Integrated Gradients attributions with higher fidelity (n_steps=50)
for a single model and overwrite the saved attributions JSONL.

Usage: run from repo root. Adjust MODEL_PATH and NUM_SAMPLES as needed.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained" / "distilbert_20251126_234714"
RESULTS_CSV = ROOT / "outputs" / "batch_check" / "distilbert_20251126_234714_results.csv"
ATTR_OUT = ROOT / "outputs" / "batch_check" / "attributions" / "distilbert_20251126_234714_attributions.jsonl"

N_STEPS = 50
NUM_SAMPLES = 20

def load_token_attr_module():
    ta_path = str(ROOT / "src" / "explainability" / "token_attribution.py")
    return SourceFileLoader("token_attribution", ta_path).load_module()

def main():
    if not MODEL_PATH.exists():
        print("Model path not found:", MODEL_PATH)
        return
    if not RESULTS_CSV.exists():
        print("Results CSV not found:", RESULTS_CSV)
        return

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    ta_mod = load_token_attr_module()
    explain = ta_mod.explain_tokens_with_ig

    import pandas as pd
    df = pd.read_csv(RESULTS_CSV)

    samples = df.head(NUM_SAMPLES)

    out_lines = []
    for _, row in samples.iterrows():
        sid = int(row['id'])
        text = str(row['text'])
        # Load prediction from CSV (prediction_label)
        pred = int(row['prediction_label'])
        try:
            toks = explain(model=model, tokenizer=tokenizer, text=text, prediction=pred, device=device, n_steps=N_STEPS)
        except Exception as e:
            print(f"Error computing attribution for id {sid}: {e}")
            toks = []
        out_lines.append({'id': sid, 'tokens': toks})

    ATTR_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTR_OUT, 'w', encoding='utf-8') as fh:
        for item in out_lines:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Wrote attributions to", ATTR_OUT)

if __name__ == '__main__':
    main()
