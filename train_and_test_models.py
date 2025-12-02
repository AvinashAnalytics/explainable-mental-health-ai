"""
Comprehensive Model Training and Testing Script
Trains models on small dataset and tests all components
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, 'src')

print("=" * 80)
print("🚀 COMPREHENSIVE MODEL TRAINING & TESTING")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Use real dataset with sampled subset for faster training on CPU
DATASET_PATH = "data/merged_real_dataset.csv"
OUTPUT_DIR = "models/trained"
TEST_SIZE = 0.2
RANDOM_SEED = 42
SAMPLE_SIZE = 1000  # Use 1000 samples for reasonable training time on CPU

MODELS_TO_TRAIN = [
    {
        "name": "DistilBERT",
        "model_name": "distilbert-base-uncased",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "description": "Lightweight BERT variant (66M params)"
    },
    {
        "name": "DistilRoBERTa-Emotion",
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "description": "Pre-trained on emotion data"
    },
    {
        "name": "MentalBERT",
        "model_name": "mental/mental-bert-base-uncased",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "description": "BERT fine-tuned on mental health data"
    },
    {
        "name": "RoBERTa-Base",
        "model_name": "roberta-base",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "description": "RoBERTa base model (125M params)"
    },
    {
        "name": "BERT-Base",
        "model_name": "bert-base-uncased",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "description": "Original BERT base (110M params)"
    }
]

print(f"\n📊 Dataset: {DATASET_PATH}")
print(f"📁 Output: {OUTPUT_DIR}")
print(f"🎯 Test Split: {TEST_SIZE * 100}%")
print(f"🎲 Random Seed: {RANDOM_SEED}")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✓ Loaded {len(df)} samples")
    
    # Sample for faster training
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"✓ Sampled {SAMPLE_SIZE} samples for faster training")
    
    # Check required columns
    if 'text' not in df.columns:
        raise ValueError("Dataset must have 'text' column")
    
    # Labels should already exist in merged_real_dataset
    if 'label' not in df.columns:
        raise ValueError("Dataset must have 'label' column")
    
    # Show distribution
    label_counts = df['label'].value_counts()
    print(f"\n📊 Label Distribution:")
    print(f"   Class 0 (Control): {label_counts[0]} samples ({label_counts[0]/len(df)*100:.1f}%)")
    print(f"   Class 1 (Depression): {label_counts[1]} samples ({label_counts[1]/len(df)*100:.1f}%)")
    
    print(f"\n✓ Using {len(df)} samples for training")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: SPLITTING DATA")
print("=" * 80)

from sklearn.model_selection import train_test_split

try:
    train_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=df['label']
    )
    
    print(f"✓ Train set: {len(train_df)} samples")
    print(f"✓ Test set: {len(test_df)} samples")
    
    # Save splits
    os.makedirs("data/splits", exist_ok=True)
    train_df.to_csv("data/splits/train.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)
    print(f"✓ Saved splits to data/splits/")
    
except Exception as e:
    print(f"✗ Error splitting data: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: TRAIN MODELS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: TRAINING MODELS")
print("=" * 80)

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'],
        'f1': f1_metric.compute(predictions=predictions, references=labels, average='binary')['f1'],
        'precision': precision_metric.compute(predictions=predictions, references=labels, average='binary')['precision'],
        'recall': recall_metric.compute(predictions=predictions, references=labels, average='binary')['recall']
    }

results = {}

for idx, model_config in enumerate(MODELS_TO_TRAIN, 1):
    print(f"\n{'─' * 80}")
    print(f"MODEL {idx}/{len(MODELS_TO_TRAIN)}: {model_config['name']}")
    print(f"{'─' * 80}")
    
    model_name = model_config['model_name']
    output_path = os.path.join(OUTPUT_DIR, model_config['name'].lower().replace(' ', '_'))
    
    try:
        # Load tokenizer and model
        print(f"\n📥 Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        
        # Tokenize datasets
        print(f"\n🔤 Tokenizing data...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=256
            )
        
        # Convert to HF Dataset
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        print(f"✓ Tokenization complete")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=model_config['epochs'],
            per_device_train_batch_size=model_config['batch_size'],
            per_device_eval_batch_size=model_config['batch_size'],
            learning_rate=model_config['learning_rate'],
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"{output_path}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=RANDOM_SEED,
            report_to="none",
            save_total_limit=1,
            no_cuda=not torch.cuda.is_available()
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Train
        print(f"\n🚀 Training {model_config['name']}...")
        print(f"   Epochs: {model_config['epochs']}")
        print(f"   Batch Size: {model_config['batch_size']}")
        print(f"   Learning Rate: {model_config['learning_rate']}")
        print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        if not torch.cuda.is_available():
            print(f"\n⚠️ Training on CPU - This will take ~15-20 minutes")
            print(f"   Estimated time: {len(train_df) // model_config['batch_size'] * model_config['epochs'] * 3 // 60} minutes")
        
        start_time = datetime.now()
        
        train_result = trainer.train()
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Training complete in {training_time/60:.1f} minutes")
        
        # Evaluate
        print(f"\n📊 Evaluating on test set...")
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"✓ Model saved to {output_path}")
        
        # Store results
        results[model_config['name']] = {
            'model_name': model_name,
            'description': model_config['description'],
            'training_time_minutes': training_time / 60,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'accuracy': eval_result['eval_accuracy'],
            'f1_score': eval_result['eval_f1'],
            'precision': eval_result['eval_precision'],
            'recall': eval_result['eval_recall'],
            'train_loss': train_result.training_loss,
            'eval_loss': eval_result['eval_loss'],
            'output_path': output_path
        }
        
        print(f"\n📈 Results:")
        print(f"   Accuracy:  {eval_result['eval_accuracy']:.4f}")
        print(f"   F1 Score:  {eval_result['eval_f1']:.4f}")
        print(f"   Precision: {eval_result['eval_precision']:.4f}")
        print(f"   Recall:    {eval_result['eval_recall']:.4f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        break
    except Exception as e:
        print(f"\n✗ Error training {model_config['name']}: {e}")
        results[model_config['name']] = {
            'error': str(e),
            'status': 'failed'
        }
        continue

# ============================================================================
# STEP 4: COMPREHENSIVE TESTING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: COMPREHENSIVE MODEL TESTING")
print("=" * 80)

# Test sample texts
test_texts = [
    "I feel hopeless and worthless. Nothing brings me joy anymore.",
    "Life is great! I'm so happy and excited about the future.",
    "I can't sleep at night. Everything feels overwhelming and pointless.",
    "Just finished a great workout. Feeling energized and positive!",
    "I think about death a lot. Maybe everyone would be better off without me.",
    "Enjoying a wonderful day with friends and family. So grateful!"
]

print(f"\n🧪 Testing on {len(test_texts)} sample texts...")

for model_name, result in results.items():
    if 'error' in result:
        print(f"\n⚠️ Skipping {model_name} (training failed)")
        continue
    
    print(f"\n{'─' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'─' * 80}")
    
    try:
        # Load model
        output_path = result['output_path']
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        model = AutoModelForSequenceClassification.from_pretrained(output_path)
        model.eval()
        
        predictions = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_label].item()
            
            predictions.append({
                'text': text[:60] + "...",
                'prediction': 'Depression' if pred_label == 1 else 'Control',
                'confidence': confidence
            })
            
            print(f"\n📝 Text: {text[:60]}...")
            print(f"   Prediction: {predictions[-1]['prediction']}")
            print(f"   Confidence: {confidence:.2%}")
        
        # Store predictions
        result['sample_predictions'] = predictions
        
    except Exception as e:
        print(f"✗ Error testing {model_name}: {e}")

# ============================================================================
# STEP 5: GENERATE REPORT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: GENERATING REPORT")
print("=" * 80)

# Create report
report = {
    'timestamp': datetime.now().isoformat(),
    'dataset': DATASET_PATH,
    'total_samples': len(df),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'models_trained': len([r for r in results.values() if 'error' not in r]),
    'models_failed': len([r for r in results.values() if 'error' in r]),
    'results': results
}

# Save report
os.makedirs("outputs", exist_ok=True)
report_path = f"outputs/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to {report_path}")

# Print summary
print("\n" + "=" * 80)
print("📊 TRAINING SUMMARY")
print("=" * 80)

print(f"\n✓ Successfully trained: {report['models_trained']}/{len(MODELS_TO_TRAIN)} models")
print(f"✗ Failed: {report['models_failed']}/{len(MODELS_TO_TRAIN)} models")

print(f"\n🏆 Model Rankings (by F1 Score):")
sorted_results = sorted(
    [(name, r) for name, r in results.items() if 'f1_score' in r],
    key=lambda x: x[1]['f1_score'],
    reverse=True
)

for rank, (name, result) in enumerate(sorted_results, 1):
    print(f"\n{rank}. {name}")
    print(f"   F1 Score:  {result['f1_score']:.4f}")
    print(f"   Accuracy:  {result['accuracy']:.4f}")
    print(f"   Precision: {result['precision']:.4f}")
    print(f"   Recall:    {result['recall']:.4f}")
    print(f"   Training Time: {result['training_time_minutes']:.1f} minutes")

# Create CSV summary
summary_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': r.get('accuracy', 0),
        'F1 Score': r.get('f1_score', 0),
        'Precision': r.get('precision', 0),
        'Recall': r.get('recall', 0),
        'Training Time (min)': r.get('training_time_minutes', 0),
        'Status': 'Success' if 'error' not in r else 'Failed'
    }
    for name, r in results.items()
])

summary_path = "outputs/training_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary CSV saved to {summary_path}")

print("\n" + "=" * 80)
print("✅ ALL TRAINING AND TESTING COMPLETE!")
print("=" * 80)

print(f"\n📁 Output files:")
print(f"   - Models: {OUTPUT_DIR}/")
print(f"   - Report: {report_path}")
print(f"   - Summary: {summary_path}")
print(f"   - Splits: data/splits/")

print("\n🚀 Next steps:")
print("   1. Review training report for metrics")
print("   2. Test models in enhanced Streamlit app")
print("   3. Compare with rule-based and LLM methods")
print("   4. Use best model for production")

print("\n" + "=" * 80)
