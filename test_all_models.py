"""
Test all 5 trained models to verify predictions are working correctly
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path

# Test cases
test_texts = [
    ("I am feeling great today and everything is wonderful!", "CONTROL"),
    ("Life is amazing, I'm so happy and excited about the future!", "CONTROL"),
    ("The weather is nice today.", "CONTROL"),
    ("I feel empty inside, nothing matters anymore, I can't go on", "DEPRESSION"),
    ("Every day is painful, I have no energy, no hope, constant sadness", "DEPRESSION"),
]

# Available models
models_dir = Path("models/trained")
available_models = []

print("=" * 100)
print("TESTING ALL TRAINED MODELS")
print("=" * 100)

# Check which models exist
for model_folder in models_dir.iterdir():
    if model_folder.is_dir():
        config_file = model_folder / "config.json"
        if config_file.exists():
            available_models.append(model_folder.name)

print(f"\n✅ Found {len(available_models)} trained models:")
for i, model_name in enumerate(available_models, 1):
    print(f"   {i}. {model_name}")

# Test each model
for model_idx, model_name in enumerate(available_models, 1):
    print("\n" + "=" * 100)
    print(f"MODEL {model_idx}/{len(available_models)}: {model_name}")
    print("=" * 100)
    
    try:
        # Load model
        model_path = models_dir / model_name
        print(f"\n📂 Loading from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   - Model type: {model.config.model_type}")
        print(f"   - Num labels: {model.config.num_labels}")
        
        # Test predictions
        print(f"\n🧪 Running {len(test_texts)} test cases...")
        print("-" * 100)
        
        correct_predictions = 0
        total_tests = len(test_texts)
        
        for test_idx, (text, expected_label) in enumerate(test_texts, 1):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()
            
            prob_control = probs[0][0].item()
            prob_depression = probs[0][1].item()
            
            predicted_label = "DEPRESSION" if prediction == 1 else "CONTROL"
            is_correct = predicted_label == expected_label
            
            if is_correct:
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print(f"\nTest {test_idx}/{total_tests}: {status}")
            print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"   Expected: {expected_label}")
            print(f"   Predicted: {predicted_label} ({confidence:.1%} confidence)")
            print(f"   Probabilities: Control={prob_control:.4f}, Depression={prob_depression:.4f}")
        
        # Summary for this model
        accuracy = (correct_predictions / total_tests) * 100
        print("\n" + "-" * 100)
        print(f"📊 MODEL SUMMARY: {model_name}")
        print(f"   Accuracy: {correct_predictions}/{total_tests} = {accuracy:.1f}%")
        
        if accuracy == 100:
            print(f"   ✅ PERFECT - All predictions correct!")
        elif accuracy >= 80:
            print(f"   ✅ GOOD - Most predictions correct")
        else:
            print(f"   ⚠️  WARNING - Many incorrect predictions")
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()

# Final summary
print("\n" + "=" * 100)
print("FINAL SUMMARY - ALL MODELS")
print("=" * 100)
print(f"\n✅ Tested {len(available_models)} models")
print(f"✅ Each model tested with {len(test_texts)} test cases")
print(f"   - {sum(1 for _, label in test_texts if label == 'CONTROL')} CONTROL texts (positive)")
print(f"   - {sum(1 for _, label in test_texts if label == 'DEPRESSION')} DEPRESSION texts (negative)")

print("\n" + "=" * 100)
print("CONCLUSION:")
print("=" * 100)
print("If all models show 100% or high accuracy:")
print("  ✅ Models are working correctly")
print("  ✅ Can distinguish between positive and negative texts")
print("  ✅ No issue with 'always predicting depression'")
print("\nIf models show low accuracy:")
print("  ⚠️  Models may need retraining")
print("  ⚠️  Check training data quality")
print("=" * 100)
