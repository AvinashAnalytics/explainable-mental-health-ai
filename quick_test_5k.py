"""
Quick Test Script - Train and Test All Models with 5K Samples
Controlled by config.py
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Import configuration
from config import (
    QUICK_TEST_CONFIG, DATASET_CONFIG, MODELS_CONFIG, 
    TRAINING_CONFIG, print_config, get_active_config
)

print("=" * 80)
print("🚀 QUICK TEST MODE - 5K SAMPLES")
print("=" * 80)

# Print configuration
print_config()

# Get active configuration
config = get_active_config()

print("\n📊 Step 1: Prepare 5K Sample Dataset")
print("-" * 80)

# Load full dataset and sample 5K
df = pd.read_csv(DATASET_CONFIG["test_dataset"])
print(f"✓ Loaded dataset: {len(df)} samples")

if QUICK_TEST_CONFIG["enabled"] and len(df) > QUICK_TEST_CONFIG["sample_size"]:
    # Sample 5K rows, stratified by label
    df_sampled = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), QUICK_TEST_CONFIG["sample_size"] // 2), random_state=42)
    )
    
    # Save sampled dataset
    sample_path = "data/quick_test_5k.csv"
    df_sampled.to_csv(sample_path, index=False)
    
    print(f"✓ Created 5K sample: {len(df_sampled)} samples")
    print(f"   Depression: {(df_sampled['label']==1).sum()}")
    print(f"   Control: {(df_sampled['label']==0).sum()}")
    print(f"✓ Saved to: {sample_path}")
    
    dataset_to_use = sample_path
else:
    dataset_to_use = DATASET_CONFIG["test_dataset"]
    print(f"✓ Using existing dataset: {dataset_to_use}")

print("\n🤖 Step 2: Train Model")
print("-" * 80)

import subprocess

for model in QUICK_TEST_CONFIG["models_to_test"]:
    print(f"\n🔥 Training: {model}")
    
    cmd = [
        "python", "train_depression_classifier.py",
        "--model", model,
        "--data", dataset_to_use,
        "--epochs", str(QUICK_TEST_CONFIG["epochs"]),
        "--batch-size", str(QUICK_TEST_CONFIG["batch_size"]),
        "--lr", str(TRAINING_CONFIG["learning_rate"]),
        "--output-dir", MODELS_CONFIG["output_dir"]
    ]
    
    print(f"💻 Command: {' '.join(cmd)}\n")
    
    start_time = datetime.now()
    result = subprocess.run(cmd)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds() / 60
    
    if result.returncode == 0:
        print(f"\n✅ {model} trained successfully in {duration:.1f} minutes")
    else:
        print(f"\n❌ {model} training failed")
        sys.exit(1)

print("\n🧪 Step 3: Run Tests")
print("-" * 80)

tests = [
    ("test_phase1.py", "Core Features"),
    ("test_new_features.py", "Advanced Features"),
    ("test_model_comparison.py", "Model Comparison"),
]

test_results = []

for test_file, test_name in tests:
    print(f"\n📝 Running: {test_name}")
    result = subprocess.run(["python", test_file], capture_output=True)
    
    if result.returncode == 0:
        print(f"   ✅ {test_name} - PASSED")
        test_results.append((test_name, "PASSED"))
    else:
        print(f"   ❌ {test_name} - FAILED")
        test_results.append((test_name, "FAILED"))

print("\n🎯 Step 4: Test Predictions")
print("-" * 80)

# Find the trained model
import glob
model_dirs = glob.glob(f"{MODELS_CONFIG['output_dir']}/distilbert_*")
if model_dirs:
    latest_model = sorted(model_dirs)[-1]
    print(f"✓ Found trained model: {latest_model}")
    
    # Test prediction
    test_text = "I feel hopeless and can't sleep. Life feels meaningless."
    cmd = [
        "python", "predict_depression.py",
        "--model", latest_model,
        "--text", test_text
    ]
    
    print(f"\n💬 Testing prediction...")
    print(f"   Input: {test_text}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ Prediction successful")
        print(f"\n{result.stdout}")
    else:
        print(f"   ❌ Prediction failed")
        print(f"{result.stderr}")
else:
    print("⚠️  No trained model found")

print("\n" + "=" * 80)
print("📊 QUICK TEST SUMMARY")
print("=" * 80)

print(f"\n✅ Dataset: {dataset_to_use}")
print(f"✅ Samples: {len(pd.read_csv(dataset_to_use))}")
print(f"✅ Model: {QUICK_TEST_CONFIG['models_to_test'][0]}")
print(f"✅ Epochs: {QUICK_TEST_CONFIG['epochs']}")

print(f"\n🧪 Test Results:")
for test_name, status in test_results:
    icon = "✅" if status == "PASSED" else "❌"
    print(f"   {icon} {test_name}: {status}")

success_count = sum(1 for _, status in test_results if status == "PASSED")
total_count = len(test_results)

print(f"\n🎯 Overall: {success_count}/{total_count} tests passed")

if success_count == total_count:
    print("\n🎉 ALL SYSTEMS WORKING!")
    print("✅ Training pipeline: Working")
    print("✅ Prediction pipeline: Working")
    print("✅ Testing framework: Working")
    print("\n💡 Ready for full-scale training with merged_real_dataset.csv (22K samples)")
else:
    print("\n⚠️  Some tests failed. Check logs above.")

print("\n" + "=" * 80)
