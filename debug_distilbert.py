"""Debug DistilBERT architecture to fix token attribution"""
import sys
sys.path.insert(0, 'src')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading DistilBERT model...")
model_path = "models/trained/distilbert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
print(f"Model type: {type(model).__name__}")
print(f"\nModel components:")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")

print("\n" + "="*80)
print("DISTILBERT STRUCTURE")
print("="*80)
if hasattr(model, 'distilbert'):
    print("✓ Has 'distilbert' attribute")
    for name, module in model.distilbert.named_children():
        print(f"  - distilbert.{name}: {type(module).__name__}")
    
    print("\n" + "="*80)
    print("TRANSFORMER LAYERS")
    print("="*80)
    if hasattr(model.distilbert, 'transformer'):
        print("✓ Has 'transformer' attribute")
        for name, module in model.distilbert.transformer.named_children():
            print(f"  - transformer.{name}: {type(module).__name__}")

print("\n" + "="*80)
print("CLASSIFIER STRUCTURE")
print("="*80)
if hasattr(model, 'classifier'):
    print(f"Classifier type: {type(model.classifier).__name__}")
    print(f"Classifier: {model.classifier}")

print("\n" + "="*80)
print("PRE-CLASSIFIER (if exists)")
print("="*80)
if hasattr(model, 'pre_classifier'):
    print(f"Pre-classifier type: {type(model.pre_classifier).__name__}")
    print(f"Pre-classifier: {model.pre_classifier}")

print("\n" + "="*80)
print("TEST FORWARD PASS")
print("="*80)
text = "I feel hopeless"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
print(f"Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    # Test full model
    outputs = model(**inputs)
    print(f"✓ Full model output shape: {outputs.logits.shape}")
    
    # Test with embeddings
    embeddings = model.distilbert.embeddings(inputs['input_ids'])
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Test transformer
    transformer_output = model.distilbert.transformer(
        embeddings,
        attention_mask=inputs['attention_mask']
    )
    print(f"✓ Transformer output shape: {transformer_output[0].shape}")
    
    # Test CLS token extraction
    cls_token = transformer_output[0][:, 0, :]
    print(f"✓ CLS token shape: {cls_token.shape}")
    
    # Test pre-classifier (if exists)
    if hasattr(model, 'pre_classifier'):
        pre_clf_output = model.pre_classifier(cls_token)
        print(f"✓ Pre-classifier output shape: {pre_clf_output.shape}")
        
        # Test dropout
        if hasattr(model, 'dropout'):
            dropout_output = model.dropout(pre_clf_output)
            print(f"✓ After dropout shape: {dropout_output.shape}")
        else:
            dropout_output = pre_clf_output
        
        # Test classifier
        logits = model.classifier(dropout_output)
        print(f"✓ Final logits shape: {logits.shape}")
    else:
        # Direct classifier
        logits = model.classifier(cls_token)
        print(f"✓ Final logits shape: {logits.shape}")

print("\n" + "="*80)
print("RECOMMENDED FIX")
print("="*80)
print("DistilBERT has a pre_classifier layer that must be used!")
print("Sequence: embeddings → transformer → CLS[0] → pre_classifier → dropout → classifier")
