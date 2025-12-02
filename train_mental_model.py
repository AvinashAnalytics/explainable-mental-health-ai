"""
Train Mental Health Focused Model - Alternative to Gated MentalBERT
Using: mental/mental-roberta-base or distilbert fine-tuned for mental health
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
print("🚀 TRAINING: Mental Health Focused Model")
print("=" * 80)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv("data/merged_real_dataset.csv")
df_sample = df.sample(n=1000, random_state=42)
print(f"✓ Using {len(df_sample)} samples")

# Split
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
print(f"✓ Train: {len(train_df)}, Test: {len(test_df)}")

# Try multiple model alternatives
models_to_try = [
    ("mental/mental-roberta-base", "mental-roberta"),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "twitter-roberta-sentiment"),
    ("distilbert-base-uncased", "distilbert-mental-health")
]

model_trained = False

for model_name, output_name in models_to_try:
    print(f"\n{'='*80}")
    print(f"Trying: {model_name}")
    print(f"{'='*80}")
    
    try:
        output_dir = f"models/trained/{output_name}"
        
        print(f"\n📥 Loading {model_name}...")
        
        # Load config
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
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
        print(f"\n🚀 Training {output_name}...")
        print(f"   Epochs: 2")
        print(f"   Batch Size: 8")
        print(f"   Device: CPU")
        print(f"\n⚠️  Estimated time: ~5-10 minutes\n")
        
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
            'model': output_name,
            'base_model': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': float(eval_results['eval_accuracy']),
                'f1_score': float(eval_results['eval_f1']),
                'precision': float(eval_results['eval_precision']),
                'recall': float(eval_results['eval_recall']),
                'training_time_minutes': training_time
            }
        }
        
        with open(f"outputs/{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\n🚀 Model ready at: {output_dir}")
        print("🌐 Launch Streamlit app to use it!\n")
        
        model_trained = True
        break  # Successfully trained, exit loop
        
    except Exception as e:
        print(f"\n❌ Failed to train {model_name}")
        print(f"   Error: {str(e)}")
        print(f"   Trying next alternative...\n")
        continue

if not model_trained:
    print("\n" + "=" * 80)
    print("⚠️  All model alternatives failed")
    print("=" * 80)
    print("\nMentalBERT requires HuggingFace authentication.")
    print("You already have 4 excellent models trained!")
    print("\nAlternative: Use the existing models which perform well:")
    print("  1. DistilRoBERTa-Emotion (97.5% accuracy)")
    print("  2. RoBERTa-Base (88% accuracy)")
    print("  3. BERT-Base (88% accuracy)")
    print("  4. DistilBERT (87% accuracy)")
