"""Check DistilBERT transformer signature"""
import sys
sys.path.insert(0, 'src')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import inspect

model_path = "models/trained/distilbert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("="*80)
print("TRANSFORMER FORWARD SIGNATURE")
print("="*80)
sig = inspect.signature(model.distilbert.transformer.forward)
print(f"Signature: {sig}")
print("\nParameters:")
for param_name, param in sig.parameters.items():
    print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")

print("\n" + "="*80)
print("TESTING CORRECT FORWARD PASS")
print("="*80)

text = "I feel hopeless"
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    # Get embeddings
    embeddings = model.distilbert.embeddings(inputs['input_ids'])
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test transformer with different argument combinations
    print("\nTest 1: Only embeddings")
    try:
        output = model.distilbert.transformer(embeddings)
        print(f"✓ Success! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTest 2: With attn_mask positional")
    try:
        output = model.distilbert.transformer(embeddings, inputs['attention_mask'])
        print(f"✓ Success! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTest 3: With attn_mask keyword")
    try:
        output = model.distilbert.transformer(embeddings, attn_mask=inputs['attention_mask'])
        print(f"✓ Success! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTest 4: With head_mask")
    try:
        output = model.distilbert.transformer(embeddings, head_mask=None)
        print(f"✓ Success! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "="*80)
print("COMPLETE FORWARD PASS WITH PRE-CLASSIFIER")
print("="*80)

with torch.no_grad():
    # Correct sequence
    embeddings = model.distilbert.embeddings(inputs['input_ids'])
    print(f"1. Embeddings: {embeddings.shape}")
    
    transformer_output = model.distilbert.transformer(embeddings, inputs['attention_mask'])
    print(f"2. Transformer: {transformer_output[0].shape}")
    
    cls_token = transformer_output[0][:, 0, :]
    print(f"3. CLS token: {cls_token.shape}")
    
    pre_clf = model.pre_classifier(cls_token)
    print(f"4. Pre-classifier: {pre_clf.shape}")
    
    pre_clf = torch.relu(pre_clf)
    print(f"5. After ReLU: {pre_clf.shape}")
    
    pre_clf = model.dropout(pre_clf)
    print(f"6. After dropout: {pre_clf.shape}")
    
    logits = model.classifier(pre_clf)
    print(f"7. Final logits: {logits.shape}")
    
    print(f"\n✓ Complete! Logits: {logits}")
