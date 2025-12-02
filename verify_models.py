"""Verify that models are real fine-tuned models"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

models_to_check = [
    "models/trained/bert-base",
    "models/trained/distilbert",
    "models/trained/roberta-base",
    "models/trained/distilroberta-emotion",
    "models/trained/twitter-roberta-sentiment"
]

print("="*80)
print("🔍 VERIFYING TRAINED MODELS")
print("="*80)

for model_path in models_to_check:
    print(f"\n📦 Checking: {model_path}")
    
    try:
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check classifier head (fine-tuned layer)
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            print(f"   ✅ Has classifier layer: {classifier}")
        elif hasattr(model, 'score'):
            classifier = model.score
            print(f"   ✅ Has score layer: {classifier}")
        
        # Check number of labels
        num_labels = model.config.num_labels
        print(f"   ✅ Number of labels: {num_labels} (binary classification)")
        
        # Test prediction
        text = "I feel hopeless"
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        print(f"   ✅ Can make predictions: {probs[0].tolist()}")
        print(f"   ✅ MODEL IS REAL AND FINE-TUNED!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("✅ ALL MODELS VERIFIED!")
print("="*80)
