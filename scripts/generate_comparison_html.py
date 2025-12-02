#!/usr/bin/env python3
"""
Generate side-by-side comparison HTML for multiple models using saved attributions.

Reads: outputs/batch_check/*_results.csv and outputs/batch_check/attributions/*_attributions.jsonl
Writes: outputs/batch_check/comparison_ui.html
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "batch_check"
ATTR_DIR = OUTPUT_DIR / "attributions"
REPORT = OUTPUT_DIR / "batch_check_report.json"

NUM_SAMPLES = 5

def load_attributions(jsonl_path: Path):
    d = {}
    if not jsonl_path.exists():
        return d
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            item = json.loads(line)
            d[item['id']] = item['tokens']
    return d

def highlight_text(text: str, token_dicts: list) -> str:
    highlighted = text
    sorted_tokens = sorted(token_dicts, key=lambda x: len(x.get('word','')), reverse=True)
    for t in sorted_tokens:
        word = t.get('word','')
        lvl = str(t.get('level','')).lower()
        score = t.get('score',0.0)
        if lvl == 'high': bg='#ff4444'
        elif lvl == 'medium': bg='#ffaa00'
        else: bg='#44dd44'
        idx = highlighted.lower().find(word.lower())
        if idx != -1 and len(word)>0:
            orig = highlighted[idx:idx+len(word)]
            repl = f'<span style="background-color:{bg};color:white;padding:1px 4px;border-radius:3px;" title="{score:.3f}">{orig}</span>'
            highlighted = highlighted[:idx] + repl + highlighted[idx+len(word):]
    return highlighted

def main():
    if not REPORT.exists():
        print('Batch report not found:', REPORT)
        return
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    models = report.get('models', [])
    if not models:
        print('No models found')
        return

    # Choose a sample set from first model's CSV
    first_csv = OUTPUT_DIR / models[0]['results_csv'].split('outputs\\batch_check\\')[-1] if models[0].get('results_csv') else OUTPUT_DIR / f"{models[0]['name']}_results.csv"
    if not first_csv.exists():
        print('First results CSV missing:', first_csv)
        return

    df = pd.read_csv(first_csv).head(NUM_SAMPLES)

    out_html = OUTPUT_DIR / 'comparison_ui.html'
    with open(out_html, 'w', encoding='utf-8') as fh:
        fh.write('<html><head><meta charset="utf-8"><title>Model Comparison</title></head><body>')
        fh.write('<h2>Side-by-side attribution comparison</h2>')

        # Header: model names
        fh.write('<div style="display:flex;gap:12px;align-items:flex-start;">')
        for m in models:
            fh.write(f'<div style="flex:1;padding:8px;border:1px solid #ddd;background:#f9f9f9;border-radius:6px;text-align:center;">{m["name"]}</div>')
        fh.write('</div>')

        for _, row in df.iterrows():
            sid = int(row['id'])
            fh.write(f'<h3>Sample ID: {sid} | True: {row.get("true_label","N/A")}</h3>')
            fh.write('<div style="display:flex;gap:12px;align-items:flex-start;">')
            for m in models:
                name = m['name']
                # Load attribution for model
                attr_file = ATTR_DIR / f'{name}_attributions.jsonl'
                atts = load_attributions(attr_file)
                tokens = atts.get(sid, [])
                text = str(row['text'])
                highlighted = highlight_text(text, tokens)
                fh.write(f'<div style="flex:1;padding:12px;border:1px solid #eee;border-radius:6px;">')
                fh.write(f'<div style="font-size:0.9rem;color:#666;margin-bottom:6px"><strong>{name}</strong></div>')
                fh.write(f'<p style="line-height:1.6">{highlighted}</p>')
                fh.write('</div>')
            fh.write('</div><hr/>')

        fh.write('</body></html>')

    print('Wrote comparison to', out_html)

if __name__ == '__main__':
    main()
