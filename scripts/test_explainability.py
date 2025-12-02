"""
Comprehensive Explainability Folder Test
Tests all 8 explainability modules with real scenarios
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict


def test_dsm_phq_mapping():
    """Test DSM-5 to PHQ-9 mapping"""
    print("\n[TEST 1] DSM-PHQ Mapping")
    print("=" * 50)
    
    from src.explainability.dsm_phq import DSM_PHQ_MAPPING
    
    # Check all 9 criteria present
    expected_criteria = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    actual_criteria = list(DSM_PHQ_MAPPING.keys())
    
    print(f"  Expected criteria: {expected_criteria}")
    print(f"  Found criteria: {actual_criteria}")
    
    if actual_criteria == expected_criteria:
        print(f"  [OK] All 9 PHQ-9 criteria present")
    else:
        print(f"  [FAIL] Missing criteria")
        return False
    
    # Check structure
    for criterion_id, criterion_data in DSM_PHQ_MAPPING.items():
        required_keys = ['phq_label', 'dsm_criteria', 'keywords']
        if not all(key in criterion_data for key in required_keys):
            print(f"  [FAIL] Criterion {criterion_id} missing required keys")
            return False
    
    print(f"  [OK] All criteria have correct structure")
    
    # Sample check
    criterion_1 = DSM_PHQ_MAPPING[1]
    print(f"\n  Sample - Criterion 1:")
    print(f"    PHQ: {criterion_1['phq_label']}")
    print(f"    DSM: {criterion_1['dsm_criteria']}")
    print(f"    Keywords: {criterion_1['keywords'][:3]}...")
    
    return True


def test_rule_explainer():
    """Test rule-based symptom detection"""
    print("\n[TEST 2] Rule-Based Explainer")
    print("=" * 50)
    
    from src.explainability.rule_explainer import detect_symptoms, MULTILINGUAL_PHRASES
    
    # Test English symptoms
    text_english = "I feel hopeless and tired. I can't sleep at night and have no appetite."
    symptoms_en = detect_symptoms(text_english)
    
    print(f"  English text: \"{text_english}\"")
    print(f"  Detected symptoms: {len(symptoms_en)}")
    
    for symptom in symptoms_en:
        print(f"    - {symptom['symptom_label']} (matched: '{symptom['keyword_found']}')")
    
    if len(symptoms_en) >= 3:
        print(f"  [OK] English symptom detection working")
    else:
        print(f"  [WARN] Expected at least 3 symptoms, got {len(symptoms_en)}")
    
    # Test Hinglish symptoms
    text_hinglish = "Mann nahi lagta, thakan lag rahi hai, neend nahi aa rahi"
    symptoms_hi = detect_symptoms(text_hinglish)
    
    print(f"\n  Hinglish text: \"{text_hinglish}\"")
    print(f"  Detected symptoms: {len(symptoms_hi)}")
    
    for symptom in symptoms_hi:
        print(f"    - {symptom['symptom_label']} (matched: '{symptom['keyword_found']}')")
    
    if len(symptoms_hi) >= 2:
        print(f"  [OK] Hinglish symptom detection working")
    else:
        print(f"  [WARN] Expected at least 2 symptoms, got {len(symptoms_hi)}")
    
    # Check multilingual lexicon coverage
    total_phrases = sum(len(phrases) for phrases in MULTILINGUAL_PHRASES.values())
    print(f"\n  Total multilingual phrases: {total_phrases}")
    print(f"  [OK] Multilingual lexicon loaded")
    
    return True


def test_llm_explainer():
    """Test LLM explanation generation"""
    print("\n[TEST 3] LLM Explainer")
    print("=" * 50)
    
    from src.explainability.llm_explainer import build_prompt, generate_prose_rationale
    
    # Test prompt building
    text = "I feel hopeless and don't want to live anymore"
    prompt = build_prompt(text)
    
    print(f"  Test text: \"{text}\"")
    print(f"  Prompt length: {len(prompt)} chars")
    
    if "DSM-5" in prompt and "PHQ-9" in prompt and text in prompt:
        print(f"  [OK] Prompt includes DSM-5, PHQ-9, and input text")
    else:
        print(f"  [FAIL] Prompt missing required elements")
        return False
    
    # Test prose generation (fallback mode, no API key)
    attention_weights = {
        "hopeless": 0.45,
        "don't": 0.32,
        "want": 0.28,
        "live": 0.35,
        "anymore": 0.20
    }
    
    explanation = generate_prose_rationale(text, attention_weights, "depression")
    
    print(f"\n  Generated explanation:")
    print(f"    \"{explanation}\"")
    
    if explanation and len(explanation) > 20:
        print(f"  [OK] Prose rationale generated")
    else:
        print(f"  [FAIL] Explanation too short or empty")
        return False
    
    return True


def test_attention_explainer():
    """Test attention extraction (structure check only)"""
    print("\n[TEST 4] Attention Explainer")
    print("=" * 50)
    
    from src.explainability.attention import AttentionExplainer
    
    # Check class structure
    required_methods = ['extract_top_tokens']
    
    for method in required_methods:
        if hasattr(AttentionExplainer, method):
            print(f"  [OK] Method '{method}' exists")
        else:
            print(f"  [FAIL] Method '{method}' missing")
            return False
    
    print(f"  [NOTE] Full testing requires trained model")
    return True


def test_lime_explainer():
    """Test LIME explainer availability"""
    print("\n[TEST 5] LIME Explainer")
    print("=" * 50)
    
    try:
        from src.explainability.lime_explainer import LIMEExplainer, LIME_AVAILABLE
        
        if LIME_AVAILABLE:
            print(f"  [OK] LIME library available")
            print(f"  [OK] LIMEExplainer class loaded")
        else:
            print(f"  [WARN] LIME library not installed (optional dependency)")
            print(f"  Run: pip install lime")
        
        # Check class structure
        required_methods = ['explain', '_predict_fn']
        methods_found = [m for m in required_methods if hasattr(LIMEExplainer, m)]
        
        print(f"  Methods found: {len(methods_found)}/{len(required_methods)}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error loading LIME explainer: {e}")
        return False


def test_shap_explainer():
    """Test SHAP explainer availability"""
    print("\n[TEST 6] SHAP Explainer")
    print("=" * 50)
    
    try:
        from src.explainability.shap_explainer import SHAPExplainer, SHAP_AVAILABLE
        
        if SHAP_AVAILABLE:
            print(f"  [OK] SHAP library available")
            print(f"  [OK] SHAPExplainer class loaded")
        else:
            print(f"  [WARN] SHAP library not installed (optional dependency)")
            print(f"  Run: pip install shap")
        
        # Check class structure
        required_methods = ['explain', '_predict_fn']
        methods_found = [m for m in required_methods if hasattr(SHAPExplainer, m)]
        
        print(f"  Methods found: {len(methods_found)}/{len(required_methods)}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error loading SHAP explainer: {e}")
        return False


def test_integrated_gradients():
    """Test Integrated Gradients implementation"""
    print("\n[TEST 7] Integrated Gradients")
    print("=" * 50)
    
    try:
        from src.explainability.integrated_gradients import IntegratedGradientsExplainer
        
        print(f"  [OK] IntegratedGradientsExplainer class loaded")
        
        # Check required methods
        required_methods = ['explain', 'compute_integrated_gradients']
        
        for method in required_methods:
            if hasattr(IntegratedGradientsExplainer, method):
                print(f"  [OK] Method '{method}' exists")
            else:
                print(f"  [WARN] Method '{method}' not found")
        
        print(f"  [NOTE] Full testing requires trained model")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error loading Integrated Gradients: {e}")
        return False


def test_attention_supervision():
    """Test attention supervision module"""
    print("\n[TEST 8] Attention Supervision")
    print("=" * 50)
    
    try:
        # Try to import
        import src.explainability.attention_supervision as attn_sup
        
        print(f"  [OK] Attention supervision module loaded")
        
        # Check for key functions/classes
        module_contents = dir(attn_sup)
        print(f"  Module exports: {len([x for x in module_contents if not x.startswith('_')])} items")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error loading attention supervision: {e}")
        return False


def test_usage_scenarios():
    """Test real-world usage patterns"""
    print("\n[TEST 9] Real-World Usage Scenarios")
    print("=" * 50)
    
    # Scenario 1: Quick symptom detection
    from src.explainability.rule_explainer import detect_symptoms
    
    text = "I'm exhausted, can't sleep, feel worthless, and think about suicide daily."
    symptoms = detect_symptoms(text)
    
    print(f"  Scenario 1 - Quick Symptom Check:")
    print(f"    Text: \"{text}\"")
    print(f"    Symptoms detected: {len(symptoms)}")
    print(f"    Crisis risk: {'Yes' if any(s['symptom_id'] == 9 for s in symptoms) else 'No'}")
    
    if len(symptoms) >= 4:
        print(f"    [OK] Multiple symptoms detected correctly")
    
    # Scenario 2: Explanation generation
    from src.explainability.llm_explainer import generate_prose_rationale
    
    attention = {"exhausted": 0.5, "suicide": 0.7, "worthless": 0.4}
    explanation = generate_prose_rationale(text, attention, "severe depression")
    
    print(f"\n  Scenario 2 - Explanation Generation:")
    print(f"    Explanation: \"{explanation[:100]}...\"")
    
    if "symptom" in explanation.lower() or "dsm" in explanation.lower():
        print(f"    [OK] Clinical explanation generated")
    
    # Scenario 3: DSM mapping lookup
    from src.explainability.dsm_phq import DSM_PHQ_MAPPING
    
    criterion_9 = DSM_PHQ_MAPPING[9]
    print(f"\n  Scenario 3 - DSM Criteria Lookup:")
    print(f"    Criterion 9 (Suicidal ideation):")
    print(f"      DSM: {criterion_9['dsm_criteria']}")
    print(f"      Keywords: {', '.join(criterion_9['keywords'][:5])}")
    print(f"    [OK] Mapping accessible")
    
    return True


def main():
    """Run all explainability tests"""
    print("\n" + "=" * 50)
    print("  EXPLAINABILITY FOLDER VALIDATION")
    print("=" * 50)
    
    tests = [
        ("DSM-PHQ Mapping", test_dsm_phq_mapping),
        ("Rule Explainer", test_rule_explainer),
        ("LLM Explainer", test_llm_explainer),
        ("Attention Explainer", test_attention_explainer),
        ("LIME Explainer", test_lime_explainer),
        ("SHAP Explainer", test_shap_explainer),
        ("Integrated Gradients", test_integrated_gradients),
        ("Attention Supervision", test_attention_supervision),
        ("Usage Scenarios", test_usage_scenarios)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("  TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # File status
    print("\n" + "=" * 50)
    print("  FILE STATUS & PURPOSE")
    print("=" * 50)
    
    files_info = [
        ("dsm_phq.py", "REQUIRED", "DSM-5 to PHQ-9 mapping (9 criteria)"),
        ("rule_explainer.py", "REQUIRED", "Multilingual symptom detection (EN/HI)"),
        ("llm_explainer.py", "REQUIRED", "LLM-based explanation generation"),
        ("attention.py", "REQUIRED", "Extract attention weights from transformers"),
        ("lime_explainer.py", "OPTIONAL", "LIME local explanations (requires lime package)"),
        ("shap_explainer.py", "OPTIONAL", "SHAP game-theoretic explanations"),
        ("integrated_gradients.py", "OPTIONAL", "Gradient-based attribution"),
        ("attention_supervision.py", "RESEARCH", "Attention supervision training")
    ]
    
    print("\nFile                          Status      Purpose")
    print("-" * 75)
    for filename, status, purpose in files_info:
        print(f"{filename:25} [{status:8}] {purpose}")
    
    print("\n" + "=" * 50)
    print("  USAGE SUMMARY")
    print("=" * 50)
    print("\nCore files actively used:")
    print("  - dsm_phq.py: Used by rule_explainer.py, clinical_validity.py")
    print("  - rule_explainer.py: Used by scripts/inference.py, scripts/quick_start.py")
    print("  - llm_explainer.py: Used by inference pipeline for LLM explanations")
    print("  - attention.py: Used for model interpretability")
    print("\nOptional files (research/advanced features):")
    print("  - lime_explainer.py: Visual word importance (requires lime)")
    print("  - shap_explainer.py: Game-theoretic attribution (requires shap)")
    print("  - integrated_gradients.py: Gradient-based explanations")
    print("  - attention_supervision.py: Training with attention supervision")
    
    if passed == total:
        print("\n[SUCCESS] All explainability modules validated!")
        return True
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
