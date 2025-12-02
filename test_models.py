#!/usr/bin/env python3
"""
Comprehensive test script for models/ folder.
Tests: calibration.py, classical.py, llm_adapter.py
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("MODELS FOLDER VALIDATION TEST")
print("=" * 70)

# Test counters
passed = 0
failed = 0

def test_result(test_name: str, success: bool, details: str = ""):
    """Print test result."""
    global passed, failed
    if success:
        passed += 1
        print(f"✓ {test_name}")
        if details:
            print(f"  → {details}")
    else:
        failed += 1
        print(f"✗ {test_name}")
        if details:
            print(f"  → ERROR: {details}")

# ============================================================================
# TEST 1: Calibration Module - Temperature Scaling
# ============================================================================
print("\n[1/9] Testing calibration.py - TemperatureScaling...")
try:
    from models.calibration import TemperatureScaling, ModelCalibrator
    
    # Create temperature scaling module
    temp_scaling = TemperatureScaling()
    
    # Check initial temperature
    assert hasattr(temp_scaling, 'temperature'), "Missing temperature parameter"
    assert temp_scaling.temperature.item() == 1.5, "Wrong initial temperature"
    
    # Test forward pass
    logits = torch.randn(10, 2)  # 10 samples, 2 classes
    scaled_logits = temp_scaling(logits)
    
    # Check shape preserved
    assert scaled_logits.shape == logits.shape, "Shape mismatch after scaling"
    
    # Check scaling applied (logits / T should be smaller for T > 1)
    assert torch.allclose(scaled_logits, logits / 1.5), "Scaling not applied correctly"
    
    test_result(
        "TemperatureScaling initialization and forward pass",
        True,
        f"Initial T=1.5, shape={scaled_logits.shape}, scaling verified"
    )
except Exception as e:
    test_result("TemperatureScaling", False, str(e))

# ============================================================================
# TEST 2: Calibration Module - ModelCalibrator Temperature Method
# ============================================================================
print("\n[2/9] Testing ModelCalibrator - Temperature method...")
try:
    # Create synthetic calibration data
    np.random.seed(42)
    
    # Overconfident predictions (common in neural networks)
    probs = np.random.dirichlet([10, 1], size=100)  # Skewed towards class 0
    labels = (probs[:, 1] > 0.3).astype(int)  # Generate labels
    logits = np.log(probs + 1e-10)  # Convert to logits
    
    # Create calibrator
    calibrator = ModelCalibrator(method='temperature')
    
    # Fit on validation data
    calibrator.fit(probs, labels, logits=logits)
    
    assert calibrator.is_fitted, "Calibrator not marked as fitted"
    
    # Calibrate probabilities
    calibrated_probs = calibrator.calibrate(probs, logits=logits)
    
    assert calibrated_probs.shape == probs.shape, "Shape mismatch after calibration"
    assert np.all(calibrated_probs >= 0) and np.all(calibrated_probs <= 1), "Invalid probabilities"
    
    # Evaluate calibration (evaluate doesn't take logits parameter)
    metrics = calibrator.evaluate(probs, labels)
    
    assert 'ece' in metrics and 'mce' in metrics, "Missing calibration metrics"
    assert 0 <= metrics['ece'] <= 1, "ECE out of range"
    assert 0 <= metrics['mce'] <= 1, "MCE out of range"
    
    test_result(
        "ModelCalibrator temperature method",
        True,
        f"T={calibrator.calibrator.temperature.item():.3f}, ECE={metrics['ece']:.3f}, MCE={metrics['mce']:.3f}"
    )
except Exception as e:
    test_result("ModelCalibrator temperature", False, str(e))

# ============================================================================
# TEST 3: Calibration Module - Platt Scaling
# ============================================================================
print("\n[3/9] Testing ModelCalibrator - Platt scaling...")
try:
    # Create calibrator with Platt method
    platt_calibrator = ModelCalibrator(method='platt')
    
    # Same synthetic data
    platt_calibrator.fit(probs, labels)
    
    # Calibrate
    platt_calibrated = platt_calibrator.calibrate(probs)
    
    assert platt_calibrated.shape == probs.shape, "Shape mismatch"
    assert np.all(platt_calibrated >= 0) and np.all(platt_calibrated <= 1), "Invalid probabilities"
    
    # Evaluate
    platt_metrics = platt_calibrator.evaluate(probs, labels)
    
    assert 'ece' in platt_metrics, "Missing ECE metric"
    assert 'brier_score' in platt_metrics, "Missing Brier score"
    
    test_result(
        "ModelCalibrator Platt scaling",
        True,
        f"ECE={platt_metrics['ece']:.3f}, Brier={platt_metrics['brier_score']:.3f}"
    )
except Exception as e:
    test_result("ModelCalibrator Platt", False, str(e))

# ============================================================================
# TEST 4: Calibration Module - Isotonic Regression
# ============================================================================
print("\n[4/9] Testing ModelCalibrator - Isotonic regression...")
try:
    # Create calibrator with isotonic method
    iso_calibrator = ModelCalibrator(method='isotonic')
    
    # Fit and calibrate
    iso_calibrator.fit(probs, labels)
    iso_calibrated = iso_calibrator.calibrate(probs)
    
    assert iso_calibrated.shape == probs.shape, "Shape mismatch"
    assert np.all(iso_calibrated >= 0) and np.all(iso_calibrated <= 1), "Invalid probabilities"
    
    # Evaluate
    iso_metrics = iso_calibrator.evaluate(probs, labels)
    
    test_result(
        "ModelCalibrator Isotonic regression",
        True,
        f"ECE={iso_metrics['ece']:.3f}, Accuracy={iso_metrics['accuracy']:.3f}"
    )
except Exception as e:
    test_result("ModelCalibrator Isotonic", False, str(e))

# ============================================================================
# TEST 5: Classical Module - ClassicalTrainer Initialization
# ============================================================================
print("\n[5/9] Testing classical.py - ClassicalTrainer initialization...")
try:
    from models.classical import ClassicalTrainer
    
    # Create mock config with required structure
    class MockConfig:
        class ModelConfig:
            class ClassicalConfig:
                backbone = 'distilbert-base-uncased'
                num_labels = 2
            classical = ClassicalConfig()
        model = ModelConfig()
        
        class DataConfig:
            max_seq_length = 256
        data = DataConfig()
    
    mock_config = MockConfig()
    
    # Initialize trainer
    trainer = ClassicalTrainer(config=mock_config)
    
    assert trainer.tokenizer is not None, "Tokenizer not loaded"
    assert trainer.model is not None, "Model not loaded"
    
    # Check attention output enabled
    assert trainer.model.config.output_attentions, "Attention output not enabled"
    
    test_result(
        "ClassicalTrainer initialization",
        True,
        f"Model={mock_config.model.classical.backbone}, attention=enabled"
    )
except Exception as e:
    test_result("ClassicalTrainer initialization", False, str(e))

# ============================================================================
# TEST 6: Classical Module - Attention Extraction
# ============================================================================
print("\n[6/9] Testing ClassicalTrainer - Attention extraction...")
try:
    # Test text
    text = "I feel hopeless and can't sleep. Everything seems pointless."
    
    # Extract attention with different pooling methods
    for method in ['layer_average', 'last_layer', 'first_last']:
        result = trainer.extract_attention(text, pooling_method=method)
        
        assert isinstance(result, dict), f"Result not a dict for {method}"
        assert 'attention_weights' in result, f"Missing attention_weights for {method}"
        assert 'tokens' in result, f"Missing tokens for {method}"
        assert 'pooling_method' in result, f"Missing pooling_method for {method}"
        assert result['pooling_method'] == method, f"Wrong pooling method: {result['pooling_method']}"
        
        attention = result['attention_weights']
        tokens = result['tokens']
        
        assert attention is not None, "Attention is None"
        assert tokens is not None, "Tokens is None"
        assert len(attention.shape) == 2, "Attention should be 2D"
        assert attention.shape[0] == attention.shape[1], "Attention should be square"
        assert len(tokens) == attention.shape[0], "Token count mismatch"
    
    # Test layer_average (research method)
    result_avg = trainer.extract_attention(text, pooling_method='layer_average')
    attention_avg = result_avg['attention_weights']
    tokens_avg = result_avg['tokens']
    
    test_result(
        "ClassicalTrainer attention extraction",
        True,
        f"Methods: 3/3 pass, layer_average shape={attention_avg.shape}, tokens={len(tokens_avg)}"
    )
except Exception as e:
    test_result("ClassicalTrainer attention", False, str(e))

# ============================================================================
# TEST 7: LLM Adapter - Provider Initialization
# ============================================================================
print("\n[7/9] Testing llm_adapter.py - Provider initialization...")
try:
    from models.llm_adapter import LLMAdapter, PromptTemplate
    
    # Test mock provider (no API key needed)
    adapter = LLMAdapter(provider='mock', model='test-model')
    
    assert adapter.provider == 'mock', "Wrong provider"
    assert adapter.temperature == 0.2, "Wrong temperature"
    assert adapter.max_tokens == 512, "Wrong max_tokens"
    
    # Test PromptTemplate
    templates = PromptTemplate(templates_dir='config/prompts')
    
    # Load a template
    zero_shot = templates.load_template('zero_shot.txt')
    
    assert zero_shot is not None, "Failed to load template"
    assert len(zero_shot) > 0, "Empty template"
    
    test_result(
        "LLMAdapter provider initialization",
        True,
        f"Provider=mock, temperature=0.2, max_tokens=512, template loaded"
    )
except Exception as e:
    test_result("LLMAdapter initialization", False, str(e))

# ============================================================================
# TEST 8: LLM Adapter - Mock Response Generation
# ============================================================================
print("\n[8/9] Testing LLMAdapter - Mock response generation...")
try:
    # Generate mock response
    prompt = "Analyze this text for depression: I feel sad all the time."
    response = adapter.generate(prompt)
    
    assert isinstance(response, dict), "Response not a dict"
    assert 'depression_likelihood' in response, "Missing depression_likelihood"
    assert 'detected_symptoms' in response, "Missing detected_symptoms"
    assert 'explanation' in response, "Missing explanation"
    assert 'requires_crisis_intervention' in response, "Missing crisis flag"
    assert 'note' in response, "Missing mock note"
    
    assert response['requires_crisis_intervention'] == False, "Wrong crisis flag"
    assert 'mock' in response['note'].lower(), "Not identified as mock"
    
    test_result(
        "LLMAdapter mock response generation",
        True,
        f"Keys: {len(response)}, likelihood={response['depression_likelihood']}, symptoms={len(response['detected_symptoms'])}"
    )
except Exception as e:
    test_result("LLMAdapter mock response", False, str(e))

# ============================================================================
# TEST 9: MentalHealthLLM - High-level Interface
# ============================================================================
print("\n[9/9] Testing MentalHealthLLM - High-level interface...")
try:
    from models.llm_adapter import MentalHealthLLM
    
    # Create high-level interface (mock provider)
    llm = MentalHealthLLM(provider='mock', model='test', templates_dir='config/prompts')
    
    # Analyze text
    text = "I've been feeling down lately. Can't sleep, lost interest in everything."
    result = llm.analyze(text, method='zero_shot', include_safety=True)
    
    assert isinstance(result, dict), "Result not a dict"
    assert 'depression_likelihood' in result, "Missing likelihood"
    assert 'method' in result, "Missing method metadata"
    assert 'model' in result, "Missing model metadata"
    assert 'input_text' in result, "Missing input text"
    
    assert result['method'] == 'zero_shot', "Wrong method"
    assert text[:50] in result['input_text'], "Input text not preserved"
    
    # Test batch analysis
    texts = [
        "I feel hopeless",
        "Life is good",
        "I can't go on anymore"
    ]
    
    batch_results = llm.batch_analyze(texts, method='zero_shot')
    
    assert len(batch_results) == 3, "Wrong batch size"
    assert all(isinstance(r, dict) for r in batch_results), "Invalid batch results"
    
    test_result(
        "MentalHealthLLM high-level interface",
        True,
        f"Single: keys={len(result)}, Batch: {len(batch_results)} results"
    )
except Exception as e:
    test_result("MentalHealthLLM", False, str(e))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("MODELS FOLDER TEST SUMMARY")
print("=" * 70)
print(f"✓ Passed: {passed}/9")
print(f"✗ Failed: {failed}/9")
print(f"Success Rate: {passed/9*100:.1f}%")

if failed == 0:
    print("\n🎉 ALL TESTS PASSED - models/ folder validated successfully!")
else:
    print(f"\n⚠️ {failed} test(s) failed - review errors above")

print("=" * 70)

sys.exit(0 if failed == 0 else 1)
