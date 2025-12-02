"""
Test model predictions to verify it's not always predicting depression
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained('models/trained/roberta-base')
tokenizer = AutoTokenizer.from_pretrained('models/trained/roberta-base')
model.eval()

# Test cases
test_texts = [
    "I am feeling great today and everything is wonderful!",
    "Life is amazing, I'm so happy and excited about the future!",
    "I feel empty inside, nothing matters anymore, I can't go on",
    "Every day is painful, I have no energy, no hope, constant sadness",
    "The weather is nice today.",
]

print("=" * 80)
print("TESTING MODEL PREDICTIONS")
print("=" * 80)

for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    
    prob_control = probs[0][0].item()
    prob_depression = probs[0][1].item()
    
    pred_label = "DEPRESSION" if prediction == 1 else "CONTROL"
    
    print(f"\nTest {i}:")
    print(f"Text: {text[:70]}{'...' if len(text) > 70 else ''}")
    print(f"Prediction: {pred_label} (class={prediction})")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probs: Control={prob_control:.4f}, Depression={prob_depression:.4f}")
    print(f"Logits: {logits[0].tolist()}")

print("\n" + "=" * 80)
print("Model config:")
print(f"Model type: {model.config.model_type}")
print(f"Num labels: {model.config.num_labels}")
print(f"Label2id: {model.config.label2id}")
print(f"Id2label: {model.config.id2label}")
print("=" * 80)
