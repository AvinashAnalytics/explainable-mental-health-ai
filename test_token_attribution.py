"""
Test Token Attribution with Integrated Gradients

This script tests the new faithful token attribution system and demonstrates
the improvements over attention-based explanations.
"""

import sys
sys.path.append('.')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.explainability.token_attribution import explain_tokens_with_ig
import json


def test_token_attribution():
    """Test token attribution on high-risk and neutral texts."""
    
    print("="*80)
    print("TESTING INTEGRATED GRADIENTS TOKEN ATTRIBUTION")
    print("="*80)
    
    # Load a model (use any trained model from your models/ directory)
    print("\n1. Loading model...")
    try:
        model_path = "models/distilbert_depression_model"  # Adjust path as needed
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print(f"✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        print("Using placeholder model for demonstration...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        model.eval()
    
    # Test case 1: High-risk text
    print("\n" + "="*80)
    print("TEST CASE 1: High-Risk Depression Text")
    print("="*80)
    
    high_risk_text = "I hate myself, nothing helps, I can't sleep. Everything is hopeless and I feel worthless."
    print(f"\nText: \"{high_risk_text}\"")
    
    # Get prediction
    inputs = tokenizer(high_risk_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
        confidence = probs[0][prediction].item()
    
    print(f"Prediction: {'Depression' if prediction == 1 else 'Control'} (confidence: {confidence:.2%})")
    
    # Get token attributions
    print("\nComputing Integrated Gradients attributions...")
    token_explanations = explain_tokens_with_ig(
        model=model,
        tokenizer=tokenizer,
        text=high_risk_text,
        prediction=prediction,
        device='cpu',
        n_steps=50
    )
    
    print(f"\nTop 10 Important Tokens:")
    print(f"{'Token':<20} {'Score':<10} {'Level':<10}")
    print("-" * 40)
    for t in token_explanations[:10]:
        emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[t['level']]
        print(f"{t['word']:<20} {t['score']:>8.3f}  {emoji} {t['level']}")
    
    # Check distribution
    high_count = sum(1 for t in token_explanations if t['level'] == 'high')
    medium_count = sum(1 for t in token_explanations if t['level'] == 'medium')
    low_count = sum(1 for t in token_explanations if t['level'] == 'low')
    
    print(f"\nDistribution: 🔴 High: {high_count}, 🟡 Medium: {medium_count}, 🟢 Low: {low_count}")
    
    # Verify emotionally loaded words are marked high
    high_risk_words = ['hate', 'nothing', 'hopeless', 'worthless', 'can\'t']
    detected_high = [t['word'] for t in token_explanations if t['level'] == 'high']
    print(f"\nExpected high-risk words: {high_risk_words}")
    print(f"Detected high-importance: {detected_high[:5]}")
    
    # Test case 2: Neutral text
    print("\n" + "="*80)
    print("TEST CASE 2: Neutral/Control Text")
    print("="*80)
    
    neutral_text = "I made pasta today, it turned out pretty good. Looking forward to the weekend."
    print(f"\nText: \"{neutral_text}\"")
    
    # Get prediction
    inputs = tokenizer(neutral_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
        confidence = probs[0][prediction].item()
    
    print(f"Prediction: {'Depression' if prediction == 1 else 'Control'} (confidence: {confidence:.2%})")
    
    # Get token attributions
    print("\nComputing Integrated Gradients attributions...")
    token_explanations = explain_tokens_with_ig(
        model=model,
        tokenizer=tokenizer,
        text=neutral_text,
        prediction=prediction,
        device='cpu',
        n_steps=50
    )
    
    print(f"\nTop 10 Important Tokens:")
    print(f"{'Token':<20} {'Score':<10} {'Level':<10}")
    print("-" * 40)
    for t in token_explanations[:10]:
        emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[t['level']]
        print(f"{t['word']:<20} {t['score']:>8.3f}  {emoji} {t['level']}")
    
    # Check distribution
    high_count = sum(1 for t in token_explanations if t['level'] == 'high')
    medium_count = sum(1 for t in token_explanations if t['level'] == 'medium')
    low_count = sum(1 for t in token_explanations if t['level'] == 'low')
    
    print(f"\nDistribution: 🔴 High: {high_count}, 🟡 Medium: {medium_count}, 🟢 Low: {low_count}")
    print(f"✓ Neutral text should have mostly green/yellow tokens")
    
    # Test case 3: Edge case - very short text
    print("\n" + "="*80)
    print("TEST CASE 3: Edge Case - Short Text")
    print("="*80)
    
    short_text = "Help."
    print(f"\nText: \"{short_text}\"")
    
    inputs = tokenizer(short_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
    
    token_explanations = explain_tokens_with_ig(
        model=model,
        tokenizer=tokenizer,
        text=short_text,
        prediction=prediction,
        device='cpu',
        n_steps=50
    )
    
    print(f"Tokens extracted: {len(token_explanations)}")
    if token_explanations:
        for t in token_explanations:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[t['level']]
            print(f"  {emoji} {t['word']}: {t['score']:.3f}")
    
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    # Check that scores are properly normalized and distributed
    print("\n✓ Checking score normalization...")
    all_scores = [t['score'] for t in token_explanations]
    if all_scores:
        print(f"  Score range: [{min(all_scores):.3f}, {max(all_scores):.3f}]")
        print(f"  ✓ Scores should be in [0, 1] range")
    
    print("\n✓ Checking level distribution...")
    print(f"  Different importance levels detected: ", end="")
    levels = set(t['level'] for t in token_explanations)
    print(f"{levels}")
    if len(levels) > 1:
        print(f"  ✓ PASS: Multiple importance levels detected (not all same color!)")
    else:
        print(f"  ⚠ WARNING: All tokens have same level (might indicate issue)")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nExpected outcomes:")
    print("✓ High-risk text: Words like 'hate', 'nothing', 'hopeless' should be 🔴 RED")
    print("✓ Neutral text: Most words should be 🟡 YELLOW or 🟢 GREEN")
    print("✓ Score distribution: NOT all tokens with same score/color")
    print("✓ Scores normalized: All in [0, 1] range")
    print("\nIf you see varied colors (not all yellow), the fix is working! ✓")


if __name__ == "__main__":
    test_token_attribution()
