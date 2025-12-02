"""
Quick demonstration of the mental health analysis system.

Run this to verify the installation and see the system in action.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import MentalHealthAnalyzer


def demo_analysis():
    """Demonstrate different analysis methods."""
    
    print("=" * 80)
    print("🧠 Mental Health Analysis System - Demo")
    print("=" * 80)
    print()
    
    # Sample texts for testing
    test_cases = [
        {
            "name": "Moderate Depression Indicators",
            "text": "I can't sleep anymore and nothing brings me joy. I feel worthless and completely exhausted all the time."
        },
        {
            "name": "Crisis Indicators (High Risk)",
            "text": "I don't want to live anymore. I have a plan to end it all. Nobody would miss me."
        },
        {
            "name": "No Significant Symptoms",
            "text": "Just finished a great workout! Feeling energized and looking forward to the weekend."
        }
    ]
    
    # Initialize analyzer
    print("Initializing analyzer...")
    try:
        analyzer = MentalHealthAnalyzer()
        print("✓ Analyzer initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}\n")
        return
    
    # Test each case
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test Case {i}: {case['name']}")
        print(f"{'─' * 80}")
        print(f"Input: \"{case['text']}\"\n")
        
        try:
            # Run analysis (hybrid method)
            result = analyzer.analyze(case['text'], method='hybrid')
            
            # Display rule-based results
            if 'rule_based' in result.get('analyses', {}):
                rule = result['analyses']['rule_based']
                print(f"📋 Rule-Based Analysis:")
                print(f"  Severity: {rule.get('severity', 'unknown')}")
                print(f"  Symptoms Detected: {rule.get('symptom_count', 0)}")
                
                if rule.get('detected_symptoms'):
                    print(f"  DSM-5 Symptoms:")
                    for symptom in rule['detected_symptoms'][:3]:  # Show first 3
                        print(f"    - {symptom['symptom_label']}: '{symptom['matched_keyword']}'")
                print()
            
            # Display LLM results
            if 'llm' in result.get('analyses', {}):
                llm = result['analyses']['llm']
                print(f"🤖 LLM Analysis:")
                print(f"  Assessment: {llm.get('depression_likelihood', 'N/A')}")
                if 'explanation' in llm:
                    print(f"  Explanation: {llm['explanation']}")
                print()
            
            # Combined assessment
            if 'combined_assessment' in result:
                combined = result['combined_assessment']
                print(f"🔬 Combined Assessment:")
                print(f"  Severity: {combined.get('severity', 'unknown')}")
                print(f"  Confidence: {combined.get('confidence', 'unknown')}")
                print(f"  Explanation: {combined.get('explanation', 'N/A')}")
                print()
            
            # Safety alerts
            if result.get('requires_crisis_intervention'):
                print(f"⚠️  CRISIS ALERT: Immediate intervention may be required")
                if 'crisis_resources' in result:
                    print(f"📞 Crisis Resources:")
                    for resource in result['crisis_resources'][:2]:
                        print(f"  {resource}")
                print()
            
            print(f"✓ Analysis completed")
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
    
    print(f"\n{'=' * 80}")
    print("Demo completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run full analysis: python scripts/inference.py \"your text here\"")
    print("2. Train a model: see scripts/train_classical.py")
    print("3. Configure LLM: set OPENAI_API_KEY environment variable")
    print("4. Read README.md for full documentation")
    print()


if __name__ == '__main__':
    demo_analysis()
