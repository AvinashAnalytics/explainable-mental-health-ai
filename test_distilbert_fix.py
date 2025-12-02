"""Test DistilBERT token attribution after fix"""
import sys
sys.path.insert(0, 'src')

from explainability.token_attribution import TokenAttributionExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("="*80)
print("TESTING DISTILBERT TOKEN ATTRIBUTION FIX")
print("="*80)

# Load model
model_path = "models/trained/distilbert"
print(f"\nLoading model from: {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create explainer
print("Creating TokenAttributionExplainer...")
explainer = TokenAttributionExplainer(model, tokenizer)

# Test cases
test_cases = [
    ("I feel empty and hopeless, nothing matters anymore", "Depression text"),
    ("I am so happy today, everything is wonderful", "Control text"),
    ("I'm tired but trying my best", "Mixed text")
]

for text, description in test_cases:
    print("\n" + "="*80)
    print(f"TEST: {description}")
    print("="*80)
    print(f"Text: '{text}'")
    
    try:
        # Get prediction
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        print(f"\nPrediction: {'DEPRESSION' if prediction == 1 else 'CONTROL'} ({confidence:.1%})")
        
        # Get token attributions
        print("\nComputing token attributions...")
        token_explanations = explainer.explain_text(text, prediction)
        
        if token_explanations and len(token_explanations) > 0:
            print(f"✓ SUCCESS! Got {len(token_explanations)} token attributions")
            
            # Show top 5 important tokens
            print("\nTop 5 important tokens:")
            for i, token_dict in enumerate(token_explanations[:5], 1):
                word = token_dict['word']
                score = token_dict['score']
                level = token_dict['level'].upper()
                print(f"  {i}. '{word}': {score:.3f} [{level}]")
            
            # Count importance levels
            high = sum(1 for t in token_explanations if t['level'] == 'high')
            medium = sum(1 for t in token_explanations if t['level'] == 'medium')
            low = sum(1 for t in token_explanations if t['level'] == 'low')
            print(f"\nImportance distribution: {high} high, {medium} medium, {low} low")
            
        else:
            print("✗ FAILED - No attributions returned")
            
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
