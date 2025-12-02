"""
Train Depression Detection Classifier

Usage:
    python train_depression_classifier.py --model roberta-base --epochs 3 --batch-size 16
    python train_depression_classifier.py --model distilbert-base-uncased --epochs 5
    python train_depression_classifier.py --model bert-base-uncased --lr 3e-5

What this script does:
1. Loads depression dataset from data/dreaddit-train.csv
2. Fine-tunes transformer model (RoBERTa/BERT/DistilBERT)
3. Evaluates on test set
4. Saves model checkpoint
5. Generates evaluation report
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train depression detection classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='roberta-base',
                        choices=['roberta-base', 'distilbert-base-uncased', 'bert-base-uncased'],
                        help='Pretrained model to fine-tune')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/dreaddit-train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set split ratio')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation set ratio (fraction of train)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for regularization')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum sequence length')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/trained',
                        help='Directory to save trained model')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name (default: model_timestamp)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps for effective larger batch size')
    parser.add_argument('--min-text-length', type=int, default=10,
                        help='Minimum number of characters in text to keep')
    parser.add_argument('--early-stopping-patience', type=int, default=2,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--class-weights', action='store_true',
                        help='Compute and use class weights to address class imbalance')
    
    return parser.parse_args()


def load_data(data_path: str, min_text_length: int = 10):
    """Load and validate dataset."""
    logger.info(f"Loading data from {data_path} (via src.data.loaders)")

    # Delegate CSV loading and cleaning to the centralized loader in src/
    try:
        from src.data.loaders import load_generic_csv
    except (ImportError, ModuleNotFoundError):
        # Fallback to local CSV read if src loader is not importable
        logger.warning("Could not import src.data.loaders — falling back to local loader")
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found: {data_path}")
            sys.exit(1)
        df = pd.read_csv(data_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            logger.error("Dataset must have 'text' and 'label' columns")
            sys.exit(1)
        df['text'] = df['text'].fillna('').astype(str)
        df = df[df['text'].str.len() >= min_text_length].reset_index(drop=True)
        # Validate and normalize labels explicitly (do not silently coerce to 0)
        label_mapping = {'depression': 1, 'depressed': 1, 'control': 0, 'none': 0, '0': 0, '1': 1}
        df['raw_label'] = df['label']
        df['label'] = df['label'].apply(lambda x: str(x).strip().lower() if pd.notnull(x) else x)
        df['label'] = df['label'].map(label_mapping)
        # for remaining unmapped, try numeric conversion
        numeric_mask = df['label'].isna() & df['raw_label'].notna()
        if numeric_mask.any():
            df.loc[numeric_mask, 'label'] = pd.to_numeric(df.loc[numeric_mask, 'raw_label'], errors='coerce')
        invalid_mask = ~df['label'].isin([0, 1])
        if invalid_mask.any():
            logger.warning(f"Dropping {int(invalid_mask.sum())} samples with invalid labels: "
                           f"{df.loc[invalid_mask, 'raw_label'].unique().tolist()}")
            df = df[~invalid_mask].reset_index(drop=True)
        df['label'] = df['label'].astype(int)
        logger.info(f"Loaded {len(df)} samples (fallback)")
        return df

    dataset = load_generic_csv(data_path, text_column='text', label_column='label', source_name='dreaddit', clean=True)
    df = dataset.to_dataframe()

    if len(df) == 0:
        logger.error(f"No valid samples found after cleaning: {data_path}")
        sys.exit(1)

    # Validate and normalize labels (explicit mapping and drop invalids)
    label_mapping = {'depression': 1, 'depressed': 1, 'control': 0, 'none': 0, '0': 0, '1': 1}
    df['raw_label'] = df['label']
    df['label'] = df['label'].apply(lambda x: str(x).strip().lower() if pd.notnull(x) else x)
    df['label'] = df['label'].map(label_mapping)
    numeric_mask = df['label'].isna() & df['raw_label'].notna()
    if numeric_mask.any():
        df.loc[numeric_mask, 'label'] = pd.to_numeric(df.loc[numeric_mask, 'raw_label'], errors='coerce')
    invalid_mask = ~df['label'].isin([0, 1])
    if invalid_mask.any():
        logger.warning(f"Dropping {int(invalid_mask.sum())} samples with invalid labels: "
                       f"{df.loc[invalid_mask, 'raw_label'].unique().tolist()}")
        df = df[~invalid_mask].reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    # Ensure text is string and has minimal length
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.len() >= min_text_length].reset_index(drop=True)

    # Safe logging: avoid division by zero when computing percentages
    total = len(df)
    logger.info(f"Loaded {total} samples")
    if total > 0:
        dep_count = int((df['label'] == 1).sum())
        ctrl_count = int((df['label'] == 0).sum())
        logger.info(f"  Depression (1): {dep_count} ({dep_count/total*100:.1f}%)")
        logger.info(f"  Control (0): {ctrl_count} ({ctrl_count/total*100:.1f}%)")
    else:
        logger.info("  Depression (1): 0 (0.0%)")
        logger.info("  Control (0): 0 (0.0%)")

    return df


def prepare_datasets(df, test_size, tokenizer, max_length, seed, val_size: float = 0.1):
    """Split and tokenize dataset."""
    logger.info(f"Splitting data (test_size={test_size})")
    # First split out test set
    try:
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label'])
    except ValueError:
        logger.warning("Stratified test split failed; falling back to non-stratified test split.")
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=None)

    # Now split train_val into train and val
    try:
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=seed, stratify=train_val_df['label'])
    except ValueError:
        logger.warning("Stratified val split failed; falling back to non-stratified val split.")
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=seed, stratify=None)
    
    logger.info(f"  Training samples: {len(train_df)}")
    logger.info(f"  Validation samples: {len(val_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    
    # Create datasets (avoid preserving pandas index which creates __index_level_0__)
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']], preserve_index=False)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    # HuggingFace Trainer expects label column named 'labels'
    try:
        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')
        test_dataset = test_dataset.rename_column('label', 'labels')
    except Exception:
        logger.debug('Could not rename label column; proceeding assuming correct label key present')
    # Remove non-tensor columns that can break PyTorch collation (e.g., raw text, index columns)
    def _clean_dataset(ds):
        remove_cols = [c for c in ds.column_names if c in ('text', '__index_level_0__', 'source')]
        if remove_cols:
            try:
                ds = ds.remove_columns(remove_cols)
            except Exception:
                logger.debug(f'Could not remove columns {remove_cols} from dataset')
        return ds

    train_dataset = _clean_dataset(train_dataset)
    val_dataset = _clean_dataset(val_dataset)
    test_dataset = _clean_dataset(test_dataset)

    return train_dataset, val_dataset, test_dataset, test_df


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    # Normalize predictions to 1D array of class indices
    try:
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        predictions = np.asarray(predictions)

        if predictions.ndim == 1:
            preds = (predictions > 0.5).astype(int)
        elif predictions.ndim == 2:
            preds = np.argmax(predictions, axis=1)
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    except Exception as e:
        logger.error(f"Failed to normalize predictions: type={type(predictions)}, shape={getattr(predictions, 'shape', None)}")
        raise ValueError(f"Failed to process predictions for metrics: {e}") from e

    acc = accuracy_score(labels, preds)
    # Binary metrics for positive class (Depression=1)
    try:
        p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
    except ValueError:
        # If binary cannot be computed (e.g., single class present), fallback to zeros
        p_pos = r_pos = f1_pos = 0.0
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'precision_pos': float(p_pos),
        'recall_pos': float(r_pos),
        'f1_pos': float(f1_pos),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f_macro)
    }


def train_model(args):
    """Main training function."""
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data (with minimum text length)
    df = load_data(args.data_path, min_text_length=args.min_text_length)

    # Derive number of labels from data
    num_labels = int(df['label'].nunique()) if 'label' in df.columns else 2
    logger.info(f"Detected {num_labels} unique label(s)")

    # Load model and tokenizer
    logger.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels
    )

    logger.info(f"  Parameters: {model.num_parameters():,}")

    # Prepare datasets (train/val/test)
    train_dataset, val_dataset, test_dataset, test_df = prepare_datasets(
        df, args.test_size, tokenizer, args.max_length, args.seed, val_size=args.val_size
    )
    
    # Setup output directory
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split('-')[0]  # roberta, bert, distilbert
        run_name = f"{model_short}_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # FP16 / GPU capability check
    use_fp16 = False
    if device == 'cuda':
        try:
            cap = torch.cuda.get_device_capability()
            use_fp16 = (cap[0] >= 7)
            if not use_fp16:
                logger.warning(f"GPU compute capability {cap} < 7.0, disabling fp16")
        except Exception:
            logger.debug('Could not determine GPU compute capability; leaving fp16 off')

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        seed=args.seed,
        fp16=use_fp16,
        report_to="none"  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    # Optionally compute class weights and use a WeightedTrainer
    class_weights_tensor = None
    if args.class_weights:
        try:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(df['label'])
            if len(classes) > 1:
                weights = compute_class_weight('balanced', classes=classes, y=df['label'])
                # Map weights into tensor ordered by label value
                weights_ordered = [float(weights[list(classes).tolist().index(int(c))]) for c in classes]
                class_weights_tensor = torch.tensor(weights_ordered, dtype=torch.float)
                logger.info(f"Computed class weights: {class_weights_tensor.tolist()}")
            else:
                logger.warning('Only one class present in data; skipping class weight computation')
        except Exception as e:
            logger.warning(f'Failed to compute class weights: {e}')

    if args.class_weights and class_weights_tensor is not None:
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights.to(self.model.device) if (class_weights is not None) else None

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                logits = outputs.logits
                if self.class_weights is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
            class_weights=class_weights_tensor
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        )
    
    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    
    train_result = trainer.train()
    
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"  Time: {train_result.metrics['train_runtime']:.2f}s")
    logger.info(f"  Samples/sec: {train_result.metrics['train_samples_per_second']:.2f}")
    logger.info("=" * 70)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    # Evaluate on validation (used during training) and then test set later
    eval_results = trainer.evaluate()

    logger.info("Validation Results:")
    logger.info(f"  Accuracy:  {eval_results.get('eval_accuracy', 0.0):.4f}")
    logger.info(f"  Precision (pos): {eval_results.get('eval_precision_pos', 0.0):.4f}")
    logger.info(f"  Recall (pos):    {eval_results.get('eval_recall_pos', 0.0):.4f}")
    logger.info(f"  F1 (pos):        {eval_results.get('eval_f1_pos', 0.0):.4f}")
    
    # Detailed predictions
    # Final evaluation on the held-out test set
    predictions = trainer.predict(test_dataset)
    # Normalize prediction output (handle tuple/list logits)
    preds_raw = predictions.predictions
    if isinstance(preds_raw, (tuple, list)):
        preds_raw = preds_raw[0]
    preds_raw = np.asarray(preds_raw)
    if preds_raw.ndim == 1:
        preds = (preds_raw > 0.5).astype(int)
    else:
        preds = np.argmax(preds_raw, axis=1)
    labels = predictions.label_ids
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Save training config for reproducibility
    try:
        config = {
            'model_name': args.model,
            'max_length': args.max_length,
            'label_mapping': {0: 'Control', 1: 'Depression'},
            'training_args': vars(args),
            'num_labels': num_labels
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {os.path.join(output_dir, 'config.json')}")
    except Exception as e:
        logger.warning(f"Failed to save config.json: {e}")
    
    # Save evaluation report
    # Build classification report robustly for cases where only a subset of classes are present
    try:
        labels_unique = sorted(list(set(labels.tolist())))
    except Exception:
        labels_unique = sorted(list(set(labels)))

    target_name_map = {0: 'Control', 1: 'Depression'}
    target_names_present = [target_name_map.get(l, str(l)) for l in labels_unique]

    report = {
        'model': args.model,
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'max_length': args.max_length
        },
        'data': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'data_path': args.data_path
        },
        'metrics': {
            'validation_accuracy': float(eval_results.get('eval_accuracy', 0.0)),
            'validation_precision_pos': float(eval_results.get('eval_precision_pos', 0.0)),
            'validation_recall_pos': float(eval_results.get('eval_recall_pos', 0.0)),
            'validation_f1_pos': float(eval_results.get('eval_f1_pos', 0.0))
        },
        'classification_report': classification_report(
            labels, preds, labels=labels_unique, target_names=target_names_present, output_dict=True
        )
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {report_path}")
    
    logger.info("=" * 70)
    logger.info(f"✅ Training complete! Model saved to: {output_dir}")
    logger.info("=" * 70)
    
    return output_dir, eval_results


if __name__ == '__main__':
    args = parse_args()
    
    logger.info("Depression Detection Classifier Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("=" * 70)
    
    try:
        output_dir, results = train_model(args)
        logger.info("\n🎉 Training successful!")
        logger.info(f"\nTo use the trained model:")
        logger.info(f"  python predict_depression.py --model {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
