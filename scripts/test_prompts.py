"""
Test Prompt System
Tests all prompt templates load correctly and format properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.llm_adapter import PromptTemplate
import json


def test_prompt_loading():
    """Test that all prompts load from both locations"""
    print("\n[TEST] Prompt Loading from config/prompts")
    print("=" * 50)
    
    config_manager = PromptTemplate('config/prompts')
    
    prompts = ['zero_shot', 'cot', 'few_shot', 'emotion_cot', 'safety', 'instruction']
    
    results = []
    for prompt_name in prompts:
        template = config_manager.load_template(prompt_name)
        if template:
            size = len(template)
            has_placeholder = '{text}' in template
            print(f"  [OK] {prompt_name}.txt - {size} chars, placeholder: {has_placeholder}")
            results.append(True)
        else:
            print(f"  [FAIL] {prompt_name}.txt - Failed to load")
            results.append(False)
    
    print(f"\nResult: {sum(results)}/{len(results)} prompts loaded")
    return all(results)


def test_prompt_formatting():
    """Test that prompts format correctly with text input"""
    print("\n[TEST] Prompt Formatting")
    print("=" * 50)
    
    manager = PromptTemplate('config/prompts')
    test_text = "I feel sad and have no energy. Life seems pointless."
    
    prompts_to_test = ['zero_shot', 'cot', 'emotion_cot', 'instruction']
    
    results = []
    for prompt_name in prompts_to_test:
        formatted = manager.format_prompt(prompt_name, test_text)
        
        # Check if text was inserted and placeholder removed
        text_inserted = test_text in formatted
        placeholder_removed = '{text}' not in formatted
        
        if text_inserted and placeholder_removed:
            print(f"  [OK] {prompt_name}.txt - Formatted correctly")
            results.append(True)
        else:
            print(f"  [FAIL] {prompt_name}.txt - Format issue")
            if not text_inserted:
                print(f"       Text not inserted")
            if not placeholder_removed:
                print(f"       Placeholder still present")
            results.append(False)
    
    print(f"\nResult: {sum(results)}/{len(results)} prompts formatted correctly")
    return all(results)


def test_prompt_content_quality():
    """Test that prompts contain required clinical frameworks"""
    print("\n[TEST] Prompt Content Quality")
    print("=" * 50)
    
    manager = PromptTemplate('config/prompts')
    
    # Required keywords for clinical quality
    required_keywords = {
        'zero_shot': ['DSM-5', 'severity', 'json'],
        'cot': ['reasoning', 'DSM-5', 'step'],
        'emotion_cot': ['emotion', 'DSM-5', 'crisis'],
        'instruction': ['DSM-5', 'PHQ-9', 'severity', 'crisis'],
        'safety': ['crisis', 'intervention', 'professional']
    }
    
    results = []
    for prompt_name, keywords in required_keywords.items():
        template = manager.load_template(prompt_name).lower()
        
        missing = [kw for kw in keywords if kw.lower() not in template]
        
        if not missing:
            print(f"  [OK] {prompt_name}.txt - All keywords present")
            results.append(True)
        else:
            print(f"  [FAIL] {prompt_name}.txt - Missing: {', '.join(missing)}")
            results.append(False)
    
    print(f"\nResult: {sum(results)}/{len(results)} prompts have quality content")
    return all(results)


def test_backward_compatibility():
    """Test that both config/prompts and src/prompts work"""
    print("\n[TEST] Backward Compatibility")
    print("=" * 50)
    
    config_manager = PromptTemplate('config/prompts')
    src_manager = PromptTemplate('src/prompts')
    
    prompts = ['zero_shot', 'cot', 'emotion_cot']
    
    results = []
    for prompt_name in prompts:
        config_template = config_manager.load_template(prompt_name)
        src_template = src_manager.load_template(prompt_name)
        
        if config_template and src_template and config_template == src_template:
            print(f"  [OK] {prompt_name}.txt - Both locations match")
            results.append(True)
        else:
            print(f"  [FAIL] {prompt_name}.txt - Locations don't match")
            results.append(False)
    
    print(f"\nResult: {sum(results)}/{len(results)} prompts synced")
    return all(results)


def main():
    """Run all prompt tests"""
    print("\n" + "=" * 50)
    print("  PROMPT SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Prompt Loading", test_prompt_loading),
        ("Prompt Formatting", test_prompt_formatting),
        ("Content Quality", test_prompt_content_quality),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
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
    
    if passed == total:
        print("\n[SUCCESS] All prompt system tests passed!")
        return True
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
