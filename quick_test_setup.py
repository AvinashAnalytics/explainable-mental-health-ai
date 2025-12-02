"""
Quick Model Test - Verify training setup without full training
"""

import sys
import os
sys.path.insert(0, 'src')

print("=" * 70)
print("🔍 QUICK MODEL SETUP VERIFICATION")
print("=" * 70)

# Test 1: Check dependencies
print("\n[1/6] Checking dependencies...")
try:
    import torch
    import transformers
    import datasets
    import evaluate
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    print("✓ All required packages installed")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 2: Load dataset
print("\n[2/6] Loading dataset...")
try:
    import pandas as pd
    df = pd.read_csv("data/dreaddit_sample.csv")
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Create labels
print("\n[3/6] Creating labels...")
try:
    if 'label' not in df.columns:
        df['label'] = df['subreddit'].apply(
            lambda x: 1 if x in ['depression', 'SuicideWatch', 'anxiety'] else 0
        )
    label_counts = df['label'].value_counts()
    print(f"✓ Labels created")
    print(f"  Class 0: {label_counts.get(0, 0)}")
    print(f"  Class 1: {label_counts.get(1, 0)}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 4: Load model (small test)
print("\n[4/6] Testing model loading...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "distilbert-base-uncased"
    print(f"  Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {params/1e6:.1f}M parameters")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 5: Test tokenization
print("\n[5/6] Testing tokenization...")
try:
    test_text = "I feel sad and hopeless"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)
    print(f"✓ Tokenization works")
    print(f"  Input shape: {inputs['input_ids'].shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 6: Test inference
print("\n[6/6] Testing inference...")
try:
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    
    print(f"✓ Inference works")
    print(f"  Prediction: {pred} (confidence: {conf:.2%})")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
print("=" * 70)

print("\n📊 Training Estimates (500 samples, CPU):")
print("  - DistilBERT (2 epochs): ~15-20 minutes")
print("  - DistilRoBERTa (2 epochs): ~15-20 minutes")
print("  - Total: ~30-40 minutes")

print("\n🚀 To start training, run:")
print("  python train_and_test_models.py")

print("\n" + "=" * 70)
