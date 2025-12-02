"""
Quick Start - Analyze text immediately without installing ML libraries.

This uses the lightweight rule-based method (DSM-5/PHQ-9 keywords).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainability.rule_explainer import explain_prediction
from src.safety.ethical_guard import SafetyGuard


def quick_analyze(text: str):
    """Instant analysis using rule-based method."""
    
    print("\n" + "="*80)
    print("🧠 Mental Health Quick Analysis")
    print("="*80)
    print(f"\nInput Text:\n\"{text}\"\n")
    print("-"*80)
    
    # Rule-based analysis
    result = explain_prediction(text)
    
    # Apply safety layer
    guard = SafetyGuard()
    safe_result = guard.process({**result, 'input_text': text})
    
    # Display results
    print(f"\n📊 ASSESSMENT:")
    print(f"  Severity Level: {safe_result['severity'].upper()}")
    print(f"  Symptoms Detected: {safe_result['symptom_count']}/9 DSM-5 criteria")
    
    if safe_result['symptom_count'] >= 5:
        print(f"  ⚠️  Meets threshold for Major Depressive Disorder (5+ symptoms)")
    
    print(f"\n💡 EXPLANATION:")
    print(f"  {safe_result['explanation']}")
    
    if safe_result['detected_symptoms']:
        print(f"\n🔍 DETECTED SYMPTOMS:")
        for symptom in safe_result['detected_symptoms']:
            print(f"  • {symptom['symptom_label']}")
            print(f"    - Evidence: \"{symptom['matched_keyword']}\"")
            print(f"    - DSM-5: {symptom['dsm_criteria']}")
            print(f"    - PHQ-9: {symptom['phq9_question']}")
    
    # Crisis warning
    if safe_result.get('requires_crisis_intervention'):
        print(f"\n{'⚠️ '*20}")
        print(f"⚠️  CRISIS INDICATORS DETECTED")
        print(f"⚠️  This text may indicate immediate risk of self-harm")
        print(f"{'⚠️ '*20}")
        print(f"\n📞 EMERGENCY RESOURCES:")
        for resource in safe_result.get('crisis_resources', []):
            print(f"  {resource}")
    
    # Disclaimer
    print(f"\n📋 DISCLAIMER:")
    print(f"  {safe_result.get('disclaimer', 'This is not medical advice.')}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Interactive or command-line mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Mental Health Text Analysis')
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--file', type=str, help='Read text from file')
    args = parser.parse_args()
    
    # Get text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Interactive mode
        print("\n🧠 Mental Health Quick Analysis")
        print("="*80)
        print("\nEnter text to analyze (press Enter twice to finish):\n")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        text = '\n'.join(lines)
    
    if not text.strip():
        print("Error: No text provided")
        return
    
    # Analyze
    quick_analyze(text)


if __name__ == '__main__':
    main()
