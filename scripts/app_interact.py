#!/usr/bin/env python3
"""
Generate HTML samples showing token highlighting using saved attributions.

Reads: outputs/batch_check/*_results.csv and outputs/batch_check/attributions/*_attributions.jsonl
Writes: outputs/batch_check/ui_samples.html
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "batch_check"
ATTR_DIR = OUTPUT_DIR / "attributions"

def load_results_csv(csv_path: Path):
    return pd.read_csv(csv_path)

def load_attributions_jsonl(jsonl_path: Path):
    data = {}
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            item = json.loads(line)
            data[item['id']] = item['tokens']
    return data

def highlight_text(text: str, token_dicts: list) -> str:
    # Build a lowercase map of best token per word
    word_map = {t['word'].lower(): t for t in token_dicts}

    highlighted = text
    # Sort tokens by length to avoid partial replacements
    sorted_tokens = sorted(token_dicts, key=lambda x: len(x['word']), reverse=True)

    for t in sorted_tokens:
        word = t['word']
        level = t.get('level', 'low')
        score = t.get('score', 0.0)
        lvl = str(level).lower()
        if lvl == 'high':
            bg = '#ff4444'; color='white'
        elif lvl == 'medium':
            bg = '#ffaa00'; color='white'
        else:
            bg = '#44dd44'; color='white'

        # case-insensitive replace (first occurrence)
        idx = highlighted.lower().find(word.lower())
        if idx != -1:
            orig = highlighted[idx:idx+len(word)]
            repl = f'<span style="background-color: {bg}; color: {color}; padding:1px 4px; border-radius:3px;" title="Importance: {score:.3f}">{orig}</span>'
            highlighted = highlighted[:idx] + repl + highlighted[idx+len(word):]

    return highlighted

def main():
    # Pick a model to show samples for
    model_name = 'distilbert_20251126_234714'
    results_csv = OUTPUT_DIR / f'{model_name}_results.csv'
    attr_jsonl = ATTR_DIR / f'{model_name}_attributions.jsonl'

    if not results_csv.exists() or not attr_jsonl.exists():
        print('Required files not found for', model_name)
        return

    df = load_results_csv(results_csv)
    atts = load_attributions_jsonl(attr_jsonl)

    # Generate HTML for first 5
    out_path = OUTPUT_DIR / 'ui_samples.html'
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('<html><head><meta charset="utf-8"><title>UI Samples</title></head><body>')
        fh.write(f'<h2>Model: {model_name} - Sample Highlights</h2>')
        for i, row in df.head(5).iterrows():
            sid = int(row['id'])
            text = row['text']
            tokens = atts.get(sid, [])
            highlighted = highlight_text(text, tokens)
            fh.write('<div style="margin:20px;padding:12px;border:1px solid #ddd;border-radius:8px;">')
            fh.write(f'<h4>Sample ID: {sid} | True: {row.get("true_label","N/A")} | Pred: {row.get("prediction")}</h4>')
            fh.write(f'<p style="font-size:1.1rem;line-height:1.6">{highlighted}</p>')
            fh.write('<details><summary>Top tokens</summary><pre>')
            fh.write(json.dumps(tokens, indent=2))
            fh.write('</pre></details>')
            fh.write('</div>')
        fh.write('</body></html>')

    print('Wrote UI samples to', out_path)

if __name__ == '__main__':
    main()
