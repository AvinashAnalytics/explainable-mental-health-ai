"""
Inspect tokenized dataset shapes to diagnose training error.
"""
import importlib.util
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
trainer_path = os.path.join(repo_root, 'train_depression_classifier.py')
spec = importlib.util.spec_from_file_location('train_depression_classifier', trainer_path)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

from transformers import AutoTokenizer

data_path = 'data/dreaddit_sample.csv'
print('Loading data...')
df = trainer.load_data(data_path)
print('Samples:', len(df))

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
train_ds, test_ds, test_df = trainer.prepare_datasets(df, test_size=0.2, tokenizer=tokenizer, max_length=128, seed=42)

print('Train features:', train_ds.features)
print('Example 0 keys:', list(train_ds[0].keys()))
for k, v in train_ds[0].items():
    t = type(v)
    try:
        length = len(v)
    except Exception:
        length = 'n/a'
    print(f" - {k}: type={t}, len={length}")

print('\nInspect a batch (first 2 examples)')
for i in range(min(2, len(train_ds))):
    ex = train_ds[i]
    print(f'Example {i}:')
    for k, v in ex.items():
        print('  ', k, type(v), (len(v) if hasattr(v, '__len__') else 'n/a'))

print('\nDone')
