"""
Automated Test Script for Mental Health AI System
Tests all 22 sample posts and generates a report
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Test posts organized by category
test_posts = {
    "High-Risk Depression": [
        "I hate myself so much, everything I do ends in failure.",
        "Nothing feels worth it anymore. I feel empty every single day.",
        "I can't find any reason to keep going like this.",
        "Every day feels heavier than the last. I'm tired of pretending."
    ],
    "Moderate Distress": [
        "I feel exhausted all the time and I can't focus on anything.",
        "My motivation is completely gone, I don't enjoy the things I used to.",
        "I'm struggling a lot lately, but I'm not sure what's wrong with me."
    ],
    "Low-Risk Sadness": [
        "I'm feeling a bit down today, but I think I'll be okay.",
        "It's been a rough week, just trying to push through everything.",
        "I feel lonely sometimes, but I try to distract myself."
    ],
    "Neutral Control": [
        "Just finished my homework, time to watch a movie.",
        "I'm going for a walk later, the weather is nice today.",
        "Made pasta today, turned out pretty good!"
    ],
    "Ambiguous Cases": [
        "I'm done with everything… who even cares anymore.",
        "I failed my test again. What is even the point?",
        "People keep telling me to be strong, but I don't feel strong."
    ],
    "Sarcastic/Hidden": [
        "Yeah sure, my life is 'amazing'. Totally loving it.",
        "Guess I'm just the family disappointment, lol.",
        "I'm fine."
    ],
    "Crisis Posts": [
        "I don't want to live anymore.",
        "I'm thinking of ending everything tonight.",
        "No one would miss me if I disappeared."
    ]
}

print("=" * 80)
print("🧪 MENTAL HEALTH AI - AUTOMATED TEST SUITE")
print("=" * 80)
print(f"\nTotal Test Posts: {sum(len(posts) for posts in test_posts.values())}")
print(f"Categories: {len(test_posts)}\n")

# Import required modules
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("✅ PyTorch and Transformers loaded successfully")
except Exception as e:
    print(f"❌ Error loading dependencies: {e}")
    sys.exit(1)

# Test model loading
model_paths = [
    "models/distilbert",
    "models/bert-base",
    "models/roberta-base",
    "models/twitter-roberta",
    "models/distilroberta-emotion"
]

print("\n" + "=" * 80)
print("📦 MODEL LOADING TEST")
print("=" * 80)

loaded_models = []
for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            model_name = os.path.basename(model_path)
            loaded_models.append((model_name, tokenizer, model))
            print(f"✅ {model_name}: Loaded successfully")
        else:
            print(f"⚠️  {model_path}: Not found")
    except Exception as e:
        print(f"❌ {model_path}: Error - {str(e)[:50]}")

print(f"\n✅ Successfully loaded {len(loaded_models)}/{len(model_paths)} models")

# Test predictions on sample posts
if loaded_models:
    print("\n" + "=" * 80)
    print("🎯 PREDICTION TEST (Using first available model)")
    print("=" * 80)
    
    model_name, tokenizer, model = loaded_models[0]
    print(f"Testing with: {model_name}\n")
    
    for category, posts in test_posts.items():
        print(f"\n📋 Category: {category}")
        print("-" * 80)
        
        for i, post in enumerate(posts, 1):
            # Truncate long posts for display
            display_post = post if len(post) <= 50 else post[:47] + "..."
            print(f"\n{i}. \"{display_post}\"")
            
            try:
                # Tokenize and predict
                inputs = tokenizer(post, return_tensors="pt", truncation=True, 
                                 max_length=512, padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    prediction = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][prediction].item() * 100
                
                label = "Depression-Risk" if prediction == 1 else "Control"
                
                # Color coding
                if category == "Crisis Posts":
                    status = "🚨 CRISIS"
                elif confidence > 80:
                    status = "✅ High Confidence"
                elif confidence > 60:
                    status = "🟡 Moderate Confidence"
                else:
                    status = "⚠️  Low Confidence"
                
                print(f"   Prediction: {label} ({confidence:.1f}%) - {status}")
                
                # Check for crisis keywords
                crisis_keywords = ["kill", "die", "suicide", "end", "live anymore", "ending everything"]
                has_crisis = any(keyword in post.lower() for keyword in crisis_keywords)
                if has_crisis:
                    print(f"   🚨 CRISIS KEYWORDS DETECTED - Safety module should trigger!")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:60]}")

# Test crisis detection
print("\n" + "=" * 80)
print("🚨 CRISIS DETECTION TEST")
print("=" * 80)

crisis_patterns = [
    "suicide", "kill myself", "end my life", "want to die",
    "ending everything", "live anymore", "disappear", "harm myself"
]

print("\nChecking crisis keyword detection:")
for post in test_posts["Crisis Posts"]:
    detected = [kw for kw in crisis_patterns if kw in post.lower()]
    if detected:
        print(f"✅ \"{post[:50]}...\" → Detected: {', '.join(detected)}")
    else:
        print(f"⚠️  \"{post[:50]}...\" → No keywords detected")

# Summary
print("\n" + "=" * 80)
print("📊 TEST SUMMARY")
print("=" * 80)
print(f"✅ Models Loaded: {len(loaded_models)}/{len(model_paths)}")
print(f"✅ Test Posts: {sum(len(posts) for posts in test_posts.values())}")
print(f"✅ Categories Tested: {len(test_posts)}")
print("\n🎯 NEXT STEP: Open http://localhost:8501 and test manually")
print("   Use the posts from TEST_RESULTS.md for comprehensive testing")
print("=" * 80)
