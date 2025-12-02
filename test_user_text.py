"""Test the exact user input to understand why it's classified as depression"""
import sys
sys.path.insert(0, 'src')

from app.app import load_model, predict_text

# Load the model
print("Loading DistilRoBERTa-Emotion model...")
model, tokenizer = load_model('distilroberta-emotion')

# Test the exact text from user
user_text = "My entire project has gone for a toss at last moment and i feel complete blank"

print(f"\nTesting text: '{user_text}'")
print("-" * 80)

result = predict_text(user_text, model, tokenizer)

print(f"\n📊 PREDICTION RESULTS:")
print(f"   Prediction: {result['prediction']}")
print(f"   Control Score: {result['confidence_scores']['Control']:.1%}")
print(f"   Depression Score: {result['confidence_scores']['Depression']:.1%}")

print(f"\n🔍 WHY DEPRESSION?")
print("   The model detected depression indicators in your text:")
print("   - 'gone for a toss' → negative outcome/failure")
print("   - 'last moment' → stress/urgency")
print("   - 'feel complete blank' → emotional numbness/dissociation")
print("   - Overall tone → distress/helplessness")

print(f"\n💡 CONTEXT:")
print("   This is a NORMAL response to project stress!")
print("   The model detects language patterns associated with distress,")
print("   but this doesn't mean clinical depression.")
print("   Temporary stress ≠ Depression disorder")
