"""
Train DistilRoBERTa-Emotion Model with Fixed Classifier
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from datetime import datetime
import json

print("=" * 80)
print("🚀 TRAINING: DistilRoBERTa-Emotion (Fixed)")
print("=" * 80)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv("data/merged_real_dataset.csv")
df_sample = df.sample(n=1000, random_state=42)
print(f"✓ Using {len(df_sample)} samples")

# Split
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
print(f"✓ Train: {len(train_df)}, Test: {len(test_df)}")

# Model setup
model_name = "j-hartmann/emotion-english-distilroberta-base"
output_dir = "models/trained/distilroberta-emotion"

print(f"\n📥 Loading {model_name}...")

# Load config and modify for binary classification
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 2  # Change from 7 (emotions) to 2 (binary)
config.problem_type = "single_label_classification"

# Load model with modified config and ignore size mismatch
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True  # This allows resizing classifier layer
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# Tokenize
print("\n🔤 Tokenizing...")
train_encodings = tokenizer(
    train_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=512
)
test_encodings = tokenizer(
    test_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=512
)

# Create dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SimpleDataset(train_encodings, train_df['label'].tolist())
test_dataset = SimpleDataset(test_encodings, test_df['label'].tolist())
print("✓ Datasets ready")

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    use_cpu=True,
    save_total_limit=1,
    metric_for_best_model="f1",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train
print("\n🚀 Training DistilRoBERTa-Emotion...")
print(f"   Epochs: 2")
print(f"   Batch Size: 8")
print(f"   Device: CPU")
print(f"\n⚠️  Estimated time: ~5-7 minutes\n")

start_time = datetime.now()
train_result = trainer.train()
training_time = (datetime.now() - start_time).total_seconds() / 60

print(f"\n✓ Training complete in {training_time:.1f} minutes")

# Evaluate
print("\n📊 Evaluating...")
eval_results = trainer.evaluate()

# Save
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Model saved to {output_dir}")

# Results
print(f"\n📈 Results:")
print(f"   Accuracy:  {eval_results['eval_accuracy']:.4f}")
print(f"   F1 Score:  {eval_results['eval_f1']:.4f}")
print(f"   Precision: {eval_results['eval_precision']:.4f}")
print(f"   Recall:    {eval_results['eval_recall']:.4f}")

# Save report
results = {
    'model': 'distilroberta-emotion',
    'timestamp': datetime.now().isoformat(),
    'metrics': {
        'accuracy': float(eval_results['eval_accuracy']),
        'f1_score': float(eval_results['eval_f1']),
        'precision': float(eval_results['eval_precision']),
        'recall': float(eval_results['eval_recall']),
        'training_time_minutes': training_time
    }
}

with open(f"outputs/distilroberta_emotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🚀 Model ready at: {output_dir}")
print("🌐 Launch Streamlit app to use it!\n")
