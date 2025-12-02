"""
Train Additional Models - One at a Time
"""
import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

print("=" * 80)
print("🚀 TRAINING ADDITIONAL MODELS")
print("=" * 80)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv("data/merged_real_dataset.csv")
print(f"✓ Total samples: {len(df)}")

# Sample for faster training
df_sample = df.sample(n=1000, random_state=42)
print(f"✓ Using {len(df_sample)} samples")

# Split
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42, stratify=df_sample['label'])
print(f"✓ Train: {len(train_df)}, Test: {len(test_df)}")

# Models to train (excluding already trained DistilBERT)
MODELS = [
    ("RoBERTa-Base", "roberta-base"),
    ("BERT-Base", "bert-base-uncased"),
]

# Train each model
for model_name, model_id in MODELS:
    print(f"\n{'='*80}")
    print(f"🤖 Training: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Create output directory
        output_dir = f"models/trained/{model_name.lower().replace('-', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        print(f"📥 Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {total_params/1e6:.1f}M parameters")
        
        # Tokenize
        print("🔤 Tokenizing...")
        train_encodings = tokenizer(
            train_df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        test_encodings = tokenizer(
            test_df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
        
        train_dataset = SimpleDataset(train_encodings, train_df['label'].tolist())
        test_dataset = SimpleDataset(test_encodings, test_df['label'].tolist())
        
        print(f"✓ Datasets ready")
        
        # Training arguments - optimized for CPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Just 1 epoch for speed
            per_device_train_batch_size=4,  # Smaller batch
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            use_cpu=True,
            disable_tqdm=False,
            report_to="none"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer
        )
        
        # Train
        print(f"\n🚀 Training {model_name}...")
        print(f"   ⏱️  Estimated time: ~15-20 minutes on CPU")
        print(f"   ⚠️  This will take a while - please wait...\n")
        
        start_time = datetime.now()
        trainer.train()
        train_time = (datetime.now() - start_time).total_seconds() / 60
        
        print(f"\n✅ Training complete in {train_time:.1f} minutes!")
        
        # Evaluate
        print("📊 Evaluating...")
        results = trainer.evaluate()
        
        # Save
        print(f"💾 Saving to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save metrics
        metrics = {
            "model_name": model_name,
            "model_id": model_id,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_time_minutes": train_time,
            "eval_loss": results.get("eval_loss", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ {model_name} training complete!")
        print(f"   📊 Eval Loss: {results.get('eval_loss', 0):.4f}")
        print(f"   ⏱️  Time: {train_time:.1f} minutes")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted for {model_name}")
        print("Moving to next model...")
        continue
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        continue

print("\n" + "="*80)
print("✅ ALL MODELS TRAINED!")
print("="*80)
print("\n📁 Models saved in: models/trained/")
print("🚀 Launch the Streamlit app to use them!")
