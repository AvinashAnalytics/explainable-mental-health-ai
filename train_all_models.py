"""
Train all three models (DistilBERT, MentalBERT, DepRoBERTa) on merged_real_dataset.csv
"""

import os
import sys
import subprocess
from datetime import datetime

print("=" * 80)
print("🚀 TRAINING ALL MODELS ON REAL MERGED DATASET")
print("=" * 80)

# Dataset info
dataset_path = "data/merged_real_dataset.csv"
print(f"\n📊 Dataset: {dataset_path}")
print(f"   Samples: 22,074 (12,825 control, 9,249 depression)")

# Training configuration
training_config = [
    {
        "name": "DistilBERT (Emotion Proxy)",
        "model": "distilbert-base-uncased",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "priority": 1
    },
    {
        "name": "MentalBERT",
        "model": "mental/mental-bert-base-uncased",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "priority": 2
    },
    {
        "name": "DepRoBERTa",
        "model": "mental/mental-roberta-base",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "priority": 3
    }
]

print("\n📋 Models to train:")
for i, config in enumerate(training_config, 1):
    print(f"   {i}. {config['name']}")
    print(f"      Model: {config['model']}")
    print(f"      Epochs: {config['epochs']}, Batch: {config['batch_size']}, LR: {config['learning_rate']}")

# Train each model
results = []
for i, config in enumerate(training_config, 1):
    print(f"\n{'='*80}")
    print(f"🔥 TRAINING MODEL {i}/3: {config['name']}")
    print(f"{'='*80}")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    
    # Build command
    cmd = [
        "python",
        "train_depression_classifier.py",
        "--model", config['model'],
        "--data", dataset_path,
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--lr", str(config['learning_rate']),
        "--output-dir", f"models/trained"
    ]
    
    print(f"\n💻 Command: {' '.join(cmd)}")
    print(f"\n{'─'*80}\n")
    
    try:
        # Run training
        start_time = datetime.now()
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        results.append({
            "model": config['name'],
            "status": "✅ SUCCESS",
            "duration": f"{duration:.1f} min",
            "timestamp": end_time.strftime("%Y%m%d_%H%M%S")
        })
        
        print(f"\n✅ {config['name']} training completed in {duration:.1f} minutes")
        
    except subprocess.CalledProcessError as e:
        results.append({
            "model": config['name'],
            "status": "❌ FAILED",
            "duration": "N/A",
            "error": str(e)
        })
        print(f"\n❌ {config['name']} training failed: {e}")
        print("Continuing with next model...")
    
    except Exception as e:
        results.append({
            "model": config['name'],
            "status": "❌ ERROR",
            "duration": "N/A",
            "error": str(e)
        })
        print(f"\n❌ Unexpected error: {e}")
        print("Continuing with next model...")

# Summary
print(f"\n{'='*80}")
print("📊 TRAINING SUMMARY")
print(f"{'='*80}\n")

for i, result in enumerate(results, 1):
    print(f"{i}. {result['model']}")
    print(f"   Status: {result['status']}")
    print(f"   Duration: {result['duration']}")
    if 'timestamp' in result:
        print(f"   Saved: models/trained/*_{result['timestamp']}/")
    if 'error' in result:
        print(f"   Error: {result['error']}")
    print()

# Count successes
success_count = sum(1 for r in results if r['status'] == "✅ SUCCESS")
print(f"{'='*80}")
print(f"🎯 Results: {success_count}/{len(results)} models trained successfully")
print(f"{'='*80}")

if success_count == len(results):
    print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("\n📝 Next steps:")
    print("   1. Run predictions: python predict_depression.py --model models/trained/distilbert_*")
    print("   2. Compare models: python compare_models.py --models models/trained/* --test-data data/merged_real_dataset.csv")
    print("   3. Test models: python test_model_comparison.py")
elif success_count > 0:
    print(f"\n⚠️  {success_count}/{len(results)} models trained. Check errors above.")
else:
    print("\n❌ No models trained successfully. Check errors above.")

print(f"\n⏰ Total time: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*80}\n")
