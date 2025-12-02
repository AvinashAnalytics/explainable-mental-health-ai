"""
GPU-Optimized Depression Detection Classifier Training

Usage:
    python train_depression_classifier_gpu.py --model roberta-base --batch-size 32
    python train_depression_classifier_gpu.py --auto-config  # Auto-detect best settings
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GPU UTILITIES
# ============================================================================

def get_gpu_info() -> Dict:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return {'available': False, 'device': 'cpu'}
    
    gpu_info = {
        'available': True,
        'device': 'cuda',
        'count': torch.cuda.device_count(),
        'gpus': []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info['gpus'].append({
            'id': i,
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count
        })
    
    return gpu_info


def get_optimal_config(gpu_info: Dict) -> Dict:
    """Auto-configure based on GPU capabilities."""
    
    if not gpu_info['available']:
        logger.warning("No GPU detected! Using CPU configuration.")
        return {
            'model': 'distilbert-base-uncased',
            'batch_size': 4,
            'max_length': 128,
            'fp16': False,
            'gradient_accumulation_steps': 4,
            'dataloader_num_workers': 0
        }
    
    # Get primary GPU memory
    primary_gpu = gpu_info['gpus'][0]
    vram_gb = primary_gpu['total_memory_gb']
    compute_cap = float(primary_gpu['compute_capability'])
    
    logger.info(f"Detected GPU: {primary_gpu['name']} ({vram_gb:.1f}GB VRAM)")
    
    # Configuration based on VRAM
    if vram_gb >= 40:  # A100, H100
        config = {
            'model': 'roberta-large',
            'batch_size': 64,
            'max_length': 512,
            'fp16': True,
            'gradient_accumulation_steps': 1,
            'dataloader_num_workers': 4
        }
    elif vram_gb >= 20:  # RTX 3090, 4090, A5000
        config = {
            'model': 'roberta-base',
            'batch_size': 32,
            'max_length': 512,
            'fp16': True,
            'gradient_accumulation_steps': 1,
            'dataloader_num_workers': 4
        }
    elif vram_gb >= 10:  # RTX 3080, 4070
        config = {
            'model': 'roberta-base',
            'batch_size': 16,
            'max_length': 384,
            'fp16': True,
            'gradient_accumulation_steps': 2,
            'dataloader_num_workers': 2
        }
    elif vram_gb >= 6:  # RTX 3060, 4060
        config = {
            'model': 'distilbert-base-uncased',
            'batch_size': 8,
            'max_length': 256,
            'fp16': True,
            'gradient_accumulation_steps': 4,
            'dataloader_num_workers': 2
        }
    else:  # Older/smaller GPUs
        config = {
            'model': 'distilbert-base-uncased',
            'batch_size': 4,
            'max_length': 128,
            'fp16': compute_cap >= 7.0,
            'gradient_accumulation_steps': 8,
            'dataloader_num_workers': 2
        }
    
    # Check FP16 compatibility
    if compute_cap < 7.0:
        logger.warning(f"GPU compute capability {compute_cap} < 7.0, disabling FP16")
        config['fp16'] = False
    
    return config


def clear_gpu_memory():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def monitor_gpu_memory() -> str:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # Compute free memory more accurately
    free = total - (allocated + reserved)

    return (
        f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, "
        f"{free:.1f}GB free, {total:.1f}GB total"
    )


def find_max_batch_size(
    model, tokenizer, sample_texts, max_length: int, device: str, candidates=None
) -> int:
    """Try candidate batch sizes and return the largest that fits on device.

    This runs a forward pass (no grad) on a small synthetic batch to detect OOMs.
    """
    if candidates is None:
        # prefer larger first
        candidates = [64, 32, 16, 8, 4, 2, 1]

    model.eval()
    for bs in candidates:
        try:
            texts = (sample_texts * ((bs // len(sample_texts)) + 1))[:bs]
            inputs = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = model(**inputs)
            # success
            return bs
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"Batch size {bs} OOM; trying smaller size")
                clear_gpu_memory()
                continue
            else:
                # re-raise unexpected errors
                raise

    return 1


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data(data_path: str, min_text_length: int = 10) -> pd.DataFrame:
    """Load and validate dataset with proper error handling."""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        logger.error("Dataset must have 'text' and 'label' columns")
        sys.exit(1)
    
    original_count = len(df)
    
    # Clean text column
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.len() >= min_text_length].reset_index(drop=True)
    
    # Validate and convert labels
    label_mapping = {
        'depression': 1, 'depressed': 1, '1': 1, 1: 1,
        'control': 0, 'normal': 0, '0': 0, 0: 0
    }
    
    def map_label(x):
        if isinstance(x, str):
            return label_mapping.get(x.lower().strip(), None)
        return label_mapping.get(x, None)
    
    df['label'] = df['label'].apply(map_label)
    
    # Remove invalid labels
    invalid_mask = df['label'].isna()
    if invalid_mask.any():
        logger.warning(f"Dropping {invalid_mask.sum()} samples with invalid labels")
        df = df[~invalid_mask].reset_index(drop=True)
    
    df['label'] = df['label'].astype(int)
    
    # Log statistics
    final_count = len(df)
    logger.info(f"Loaded {final_count} samples (dropped {original_count - final_count})")
    
    if final_count > 0:
        class_counts = df['label'].value_counts()
        for label, count in class_counts.items():
            label_name = 'Depression' if label == 1 else 'Control'
            logger.info(f"  {label_name}: {count} ({count/final_count*100:.1f}%)")
    
    return df


def prepare_datasets(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    tokenizer,
    max_length: int,
    seed: int
) -> Tuple[Dataset, Dataset, Dataset, pd.DataFrame]:
    """Split data into train/val/test and tokenize."""
    
    logger.info(f"Splitting data (test={test_size}, val={val_size})")
    
    # First split: train+val vs test (try stratified; fallback to random)
    try:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df['label']
        )

        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=seed, stratify=train_val_df['label']
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using random split.")
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=seed
        )
    
    logger.info(f"  Training: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    # Create datasets
    def create_dataset(df_subset):
        dataset = Dataset.from_pandas(
            df_subset[['text', 'label']], 
            preserve_index=False
        )
        return dataset
    
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Dynamic padding with DataCollator
            max_length=max_length
        )
    
    logger.info("Tokenizing datasets...")
    
    # Tokenize with batched processing
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text']
    )
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text']
    )
    
    # Rename label column
    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')
    
    return train_dataset, val_dataset, test_dataset, test_df


# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for binary classification."""
    predictions, labels = eval_pred
    
    # Handle various prediction formats
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    
    predictions = np.asarray(predictions)
    
    if predictions.ndim == 1:
        preds = (predictions > 0.5).astype(int)
    elif predictions.ndim == 2:
        preds = np.argmax(predictions, axis=1)
    else:
        raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
    
    # Binary metrics (for depression class)
    precision_dep, recall_dep, f1_dep, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    
    # Macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    
    accuracy = accuracy_score(labels, preds)
    
    return {
        'accuracy': accuracy,
        'f1': f1_dep,  # Primary metric: depression F1
        'precision': precision_dep,
        'recall': recall_dep,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }


# ============================================================================
# CUSTOM TRAINER WITH GPU OPTIMIZATIONS
# ============================================================================

class GPUOptimizedTrainer(Trainer):
    """Trainer with additional GPU optimizations."""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), 
            labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GPU-Optimized Depression Detection Classifier Training'
    )
    
    # Auto-configuration
    parser.add_argument('--auto-config', action='store_true',
                        help='Auto-configure based on GPU capabilities')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=None,
                        choices=['roberta-base', 'roberta-large', 
                                'distilbert-base-uncased', 'bert-base-uncased'],
                        help='Pretrained model to fine-tune')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/dreaddit-train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set split ratio')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation set split ratio')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for regularization')
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximum sequence length')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate scheduler')
    
    # GPU arguments
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None,
                        help='Gradient accumulation steps')
    parser.add_argument('--fp16', action='store_true', default=None,
                        help='Use FP16 mixed precision')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 even if supported')
    parser.add_argument('--bf16', action='store_true', default=None,
                        help='Enable BF16 (if supported by hardware)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable model.gradient_checkpointing to save memory')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save-total-limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--quantize', action='store_true', help='Apply dynamic quantization for CPU inference after training')
    parser.add_argument('--dataloader-num-workers', type=int, default=None,
                        help='Number of dataloader workers')
    parser.add_argument('--batch-finder', action='store_true',
                        help='Run a quick batch-size finder to pick the largest fit on GPU')
    parser.add_argument('--distributed', action='store_true',
                        help='Flag indicating distributed training; use torchrun to launch')
    
    # Class balancing
    parser.add_argument('--class-weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/trained',
                        help='Directory to save trained model')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                        help='Early stopping patience')
    
    return parser.parse_args()


def train_model(args):
    """Main training function with GPU optimizations."""
    global torch
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    logger.info("=" * 70)
    logger.info("GPU INFORMATION")
    logger.info("=" * 70)
    
    if gpu_info['available']:
        for gpu in gpu_info['gpus']:
            logger.info(f"GPU {gpu['id']}: {gpu['name']}")
            logger.info(f"  Memory: {gpu['total_memory_gb']:.1f} GB")
            logger.info(f"  Compute Capability: {gpu['compute_capability']}")
    else:
        logger.warning("No GPU available! Training will be slow.")
    
    # Auto-configure if requested or if parameters not specified
    if args.auto_config or args.model is None:
        optimal_config = get_optimal_config(gpu_info)
        logger.info("\nAuto-configured settings:")
        for key, value in optimal_config.items():
            logger.info(f"  {key}: {value}")
        
        # Apply optimal config for unspecified parameters
        if args.model is None:
            args.model = optimal_config['model']
        if args.batch_size is None:
            args.batch_size = optimal_config['batch_size']
        if args.max_length is None:
            args.max_length = optimal_config['max_length']
        if args.gradient_accumulation_steps is None:
            args.gradient_accumulation_steps = optimal_config['gradient_accumulation_steps']
        if args.fp16 is None and not args.no_fp16:
            args.fp16 = optimal_config['fp16']
        if args.dataloader_num_workers is None:
            args.dataloader_num_workers = optimal_config['dataloader_num_workers']
    
    # Set defaults for any remaining None values
    args.batch_size = args.batch_size or 16
    args.max_length = args.max_length or 256
    args.gradient_accumulation_steps = args.gradient_accumulation_steps or 1
    args.fp16 = args.fp16 if args.fp16 is not None else False
    args.dataloader_num_workers = args.dataloader_num_workers or 0
    
    if args.no_fp16:
        args.fp16 = False
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device setup
    device = 'cuda' if gpu_info['available'] else 'cpu'
    
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"FP16: {args.fp16}")
    logger.info(f"BF16: {args.bf16}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Load data
    df = load_data(args.data_path)
    
    # Load tokenizer and model
    logger.info(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    num_labels = df['label'].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels
    )

    # Optional gradient checkpointing to reduce memory (at cost of speed)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info('Enabled gradient checkpointing on the model')
        except Exception:
            logger.warning('Model does not support gradient checkpointing')

    # Multi-GPU note
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Multiple GPUs detected: {torch.cuda.device_count()} GPUs available.")
        logger.info("To take full advantage, launch with torchrun or accelerate for multi-GPU training.")

    # BF16 safety check
    if args.bf16:
        try:
            # torch.cuda.is_bf16_supported is available on newer CUDA/torch builds
            bf16_ok = getattr(torch.cuda, 'is_bf16_supported', lambda : False)()
        except Exception:
            bf16_ok = False
        if not bf16_ok:
            logger.warning('BF16 requested but not supported on this device; disabling bf16')
            args.bf16 = False
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Move model to GPU
    model = model.to(device)
    logger.info(monitor_gpu_memory())
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, test_df = prepare_datasets(
        df, args.test_size, args.val_size, tokenizer, args.max_length, args.seed
    )

    # Optionally run batch-size finder to select largest batch that fits
    if getattr(args, 'batch_finder', False):
        try:
            sample_texts = []
            if 'text' in test_df.columns:
                sample_texts = test_df['text'].dropna().astype(str).tolist()[:4]
            if not sample_texts:
                sample_texts = df['text'].dropna().astype(str).tolist()[:4]
            logger.info("Running batch-size finder (quick memory probe)")
            best_bs = find_max_batch_size(model, tokenizer, sample_texts, args.max_length, device)
            args.batch_size = best_bs
            logger.info(f"Batch finder selected batch size: {best_bs}")
        except Exception as e:
            logger.warning(f"Batch finder failed: {e}")

    # Distributed flag guidance
    if getattr(args, 'distributed', False):
        logger.info("Distributed training requested. Ensure you launched with torchrun or accelerate.")
    
    # Data collator for dynamic padding (more efficient)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Calculate class weights if requested
    class_weights = None
    if args.class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            'balanced',
            classes=np.unique(df['label']),
            y=df['label']
        )
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        logger.info(f"Class weights: {weights}")
    
    # Setup output directory
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split('-')[0]
        gpu_name = gpu_info['gpus'][0]['name'].replace(' ', '_') if gpu_info['available'] else 'cpu'
        run_name = f"{model_short}_{gpu_name}_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\nOutput directory: {output_dir}")
    
    # Training arguments with GPU optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimizer
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # GPU optimizations
        fp16=args.fp16,
        bf16=args.bf16 if args.bf16 is not None else False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True if device == 'cuda' else False,
        save_total_limit=args.save_total_limit,
        
        # Logging
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        report_to="none",
        
        # Other
        seed=args.seed,
        remove_unused_columns=True,
    )
    
    # Initialize trainer
    trainer = GPUOptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ]
    )
    
    # Train
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    clear_gpu_memory()
    logger.info(monitor_gpu_memory())
    
    # Support resuming from checkpoint if provided
    resume_checkpoint = args.resume_from_checkpoint if getattr(args, 'resume_from_checkpoint', None) else None
    if resume_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    # If resume-from-checkpoint requested, Trainer.train should be called with resume argument
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    runtime = train_result.metrics.get('train_runtime', 0.0)
    samples_per_sec = train_result.metrics.get('train_samples_per_second', 0.0)
    logger.info(f"Training time: {runtime:.2f}s")
    logger.info(f"Samples/sec: {samples_per_sec:.2f}")
    logger.info(monitor_gpu_memory())
    
    # Evaluate on validation set
    logger.info("\nValidation Results:")
    val_results = trainer.evaluate()
    for key, value in val_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    # Evaluate on test set
    logger.info("\nTest Results:")
    test_results = trainer.evaluate(test_dataset)
    for key, value in test_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    # Get predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    preds_raw = predictions.predictions
    if isinstance(preds_raw, (tuple, list)):
        preds_raw = preds_raw[0]
    preds = np.argmax(preds_raw, axis=1)
    labels = predictions.label_ids
    
    # Confusion matrix
    # Ensure confusion matrix has shape (2,2) even if one label is missing
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    logger.info("\nConfusion Matrix:")
    logger.info(f"              Predicted")
    logger.info(f"              Control  Depression")
    logger.info(f"Actual Control    {cm[0][0]:5d}     {cm[0][1]:5d}")
    logger.info(f"Actual Depression {cm[1][0]:5d}     {cm[1][1]:5d}")
    
    # Save model
    logger.info(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Optional dynamic quantization for CPU inference
    if getattr(args, 'quantize', False):
        try:
            logger.info('Applying dynamic quantization for CPU inference (INT8)')
            import torch.quantization
            qmodel = torch.quantization.quantize_dynamic(
                model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
            )
            qpath = os.path.join(output_dir, 'quantized_model.pth')
            torch.save(qmodel.state_dict(), qpath)
            logger.info(f'Quantized model state saved to {qpath}')
        except Exception as e:
            logger.warning(f'Quantization failed: {e}')
    
    # Save comprehensive report
    report = {
        'model': args.model,
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'gpu_info': gpu_info,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'max_length': args.max_length,
            'warmup_ratio': args.warmup_ratio,
            'fp16': args.fp16,
            'class_weights': args.class_weights
        },
        'data': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'data_path': args.data_path
        },
        'training_metrics': {
            'runtime_seconds': train_result.metrics.get('train_runtime', 0.0),
            'samples_per_second': train_result.metrics.get('train_samples_per_second', 0.0)
        },
        'validation_metrics': {
            key.replace('eval_', ''): value 
            for key, value in val_results.items() 
            if isinstance(value, (int, float))
        },
        'test_metrics': {
            key.replace('eval_', ''): value 
            for key, value in test_results.items() 
            if isinstance(value, (int, float))
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            labels, preds,
            labels=[0, 1],
            target_names=['Control', 'Depression'],
            output_dict=True,
            zero_division=0
        )
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save training config for reproducibility
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    logger.info(f"Training report saved to {report_path}")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    logger.info("=" * 70)
    logger.info(f"✅ Training complete! Model saved to: {output_dir}")
    logger.info("=" * 70)
    
    return output_dir, test_results


if __name__ == '__main__':
    args = parse_args()
    try:
        output_dir, results = train_model(args)

        logger.info("\n🎉 Training successful!")
        logger.info(f"\nTo use the trained model:")
        logger.info(f"  python predict_depression.py --model {output_dir}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("\n❌ GPU OUT OF MEMORY!")
            logger.error("Solutions:")
            logger.error("  --batch-size 8")
            logger.error("  --max-length 128")
            logger.error("  --gradient-checkpointing")
            logger.error("  --model distilbert-base-uncased")
            logger.error("  --gradient-accumulation-steps 4")
        else:
            logger.exception("Training failed")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
