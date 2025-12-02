"""
Smoke test: load dataset via project loader and run tokenization.
This avoids full training and validates imports, cleaning, and tokenization.
"""
from transformers import AutoTokenizer
import importlib.util
import os


def load_trainer_module():
    """Load train_depression_classifier.py as a module by file path."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    trainer_path = os.path.join(repo_root, 'train_depression_classifier.py')
    spec = importlib.util.spec_from_file_location('train_depression_classifier', trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


if __name__ == '__main__':
    trainer = load_trainer_module()
    load_data = trainer.load_data
    prepare_datasets = trainer.prepare_datasets

    data_path = 'data/dreaddit_sample.csv'
    print(f'Loading data from {data_path}')
    df = load_data(data_path)
    print(f'Loaded samples: {len(df)}')

    tokenizer_name = 'distilbert-base-uncased'
    print(f'Loading tokenizer: {tokenizer_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print('Preparing tokenized datasets (max_length=128)')
    try:
        train_ds, test_ds, test_df = prepare_datasets(df, test_size=0.2, tokenizer=tokenizer, max_length=128, seed=42)
    except ValueError as e:
        print('Stratified split failed (small sample). Falling back to non-stratified split and manual tokenization.')
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)
        # Manual tokenization using datasets.Dataset
        from datasets import Dataset

        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        test_ds = Dataset.from_pandas(test_df[['text', 'label']])
        train_ds = train_ds.map(tokenize_function, batched=True)
        test_ds = test_ds.map(tokenize_function, batched=True)

    print(f'Train dataset size: {len(train_ds)}')
    print(f'Test dataset size: {len(test_ds)}')
    # Show a tokenized example
    sample = train_ds[0]
    keys = list(sample.keys())[:6]
    print('Sample keys:', keys)
    for k in keys:
        print(k, type(sample[k]))
    print('Token ids (first 20):', sample.get('input_ids')[:20])
