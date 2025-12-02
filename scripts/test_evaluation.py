"""
Test Evaluation Folder - Complete Validation
Tests all 5 evaluation modules with real usage scenarios
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict


def test_metrics():
    """Test basic metrics evaluation"""
    print("\n[TEST 1] Metrics Module")
    print("=" * 50)
    
    from src.evaluation.metrics import Evaluator
    
    # Mock data
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    y_probs = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2], 
               [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6],
               [0.2, 0.8], [0.85, 0.15]]
    
    # Compute metrics
    evaluator = Evaluator()
    metrics = evaluator.compute_classification_metrics(y_true, y_pred, y_probs)
    
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    print(f"  AUC: {metrics.get('auc', 'N/A')}")
    
    # Test fluency
    explanation = "Patient shows depressed mood and anhedonia for 2 weeks"
    fluency = evaluator.evaluate_explanation_fluency(explanation)
    print(f"  Explanation Fluency: {fluency:.3f}")
    
    print("  [OK] Metrics module working")
    return True


def test_clinical_validity():
    """Test DSM-5 and PHQ-9 clinical validity"""
    print("\n[TEST 2] Clinical Validity Module")
    print("=" * 50)
    
    from src.evaluation.clinical_validity import DSM5Criteria, PHQ9Estimator, analyze_clinical_validity
    
    # Test text with depression symptoms
    test_text = """
    I feel so sad and hopeless all the time. Nothing brings me joy anymore.
    I can't sleep at night and feel exhausted during the day. I have no appetite
    and feel worthless. Sometimes I think everyone would be better off without me.
    """
    
    # Test DSM-5 detection
    dsm5_results = DSM5Criteria.detect_symptoms(test_text)
    print(f"  DSM-5 Symptoms Detected: {dsm5_results['num_symptoms']}/9")
    print(f"  Core Symptom Present: {dsm5_results['has_core_symptom']}")
    print(f"  Meets Criteria: {dsm5_results['meets_dsm5_criteria']}")
    print(f"  Severity: {dsm5_results['severity_estimate']}")
    print(f"  Crisis Risk: {dsm5_results['crisis_risk']}")
    
    # Test PHQ-9 estimation
    phq9_results = PHQ9Estimator.estimate_score(dsm5_results)
    print(f"  PHQ-9 Estimated Score: {phq9_results['estimated_score']}/27")
    print(f"  Score Range: {phq9_results['score_range']}")
    print(f"  Severity Level: {phq9_results['severity_level']}")
    
    # Test complete analysis
    complete = analyze_clinical_validity(test_text)
    print(f"  Clinical Summary: {complete['clinical_summary']['severity']}")
    
    print("  [OK] Clinical validity module working")
    return True


def test_explainability_metrics():
    """Test explainability plausibility metrics"""
    print("\n[TEST 3] Explainability Metrics Module")
    print("=" * 50)
    
    from src.evaluation.explainability_metrics import ExplainabilityMetrics
    
    # Mock rationales
    predicted = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])
    human = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.6, 0.85, 0.3, 0.2, 0.55, 0.75, 0.1, 0.4])
    
    # Test Token-F1
    token_f1 = ExplainabilityMetrics.token_f1(predicted, human)
    print(f"  Token-F1: {token_f1:.3f}")
    
    # Test IOU-F1
    iou_f1 = ExplainabilityMetrics.iou_f1(predicted, human)
    print(f"  IOU-F1: {iou_f1:.3f}")
    
    # Test AUPRC
    auprc = ExplainabilityMetrics.auprc(scores, human)
    print(f"  AUPRC: {auprc:.3f}")
    
    # Test batch computation
    predicted_batch = [predicted] * 5
    human_batch = [human] * 5
    scores_batch = [scores] * 5
    
    plausibility = ExplainabilityMetrics.compute_all_plausibility(
        predicted_batch, human_batch, scores_batch
    )
    print(f"  Batch Token-F1 Mean: {plausibility['token_f1_mean']:.3f}")
    print(f"  Batch AUPRC Mean: {plausibility['auprc_mean']:.3f}")
    
    print("  [OK] Explainability metrics module working")
    return True


def test_faithfulness_metrics():
    """Test faithfulness metrics (note: requires model, so basic test only)"""
    print("\n[TEST 4] Faithfulness Metrics Module")
    print("=" * 50)
    
    from src.evaluation.faithfulness_metrics import FaithfulnessMetrics
    
    # Check class exists and has required methods
    required_methods = [
        'comprehensiveness', 'sufficiency', 'decision_flip_rate',
        'monotonicity', 'aopc'
    ]
    
    for method in required_methods:
        if hasattr(FaithfulnessMetrics, method):
            print(f"  [OK] Method '{method}' exists")
        else:
            print(f"  [FAIL] Method '{method}' missing")
            return False
    
    print("  [OK] Faithfulness metrics module structure valid")
    print("  [NOTE] Full testing requires trained model")
    return True


def test_model_comparison():
    """Test model comparison and evaluation"""
    print("\n[TEST 5] Model Comparison Module")
    print("=" * 50)
    
    from src.evaluation.model_comparison import ModelPerformanceEvaluator, get_evaluator
    
    # Test get_evaluator
    evaluator = get_evaluator()
    print(f"  [OK] get_evaluator() working")
    
    # Test get all models
    models = evaluator.get_all_models()
    print(f"  Available Models: {len(models)}")
    for model in models[:3]:
        print(f"    - {model}")
    
    # Test get model metrics
    if models:
        test_model = models[0]
        metrics = evaluator.get_model_metrics(test_model)
        print(f"\n  Metrics for '{test_model}':")
        print(f"    Accuracy: {metrics.get('accuracy', 'N/A')}")
        print(f"    F1 Score: {metrics.get('f1_score', 'N/A')}")
        print(f"    Precision: {metrics.get('precision', 'N/A')}")
        print(f"    Recall: {metrics.get('recall', 'N/A')}")
    
    # Test compare models
    if len(models) >= 2:
        comparison = evaluator.compare_models(models[:2], metric='f1_score')
        print(f"\n  Model Comparison (F1 Score):")
        for model, score in comparison.items():
            print(f"    {model}: {score:.4f}")
    
    # Test get best model
    best_model, best_score = evaluator.get_best_model('accuracy')
    print(f"\n  Best Model (Accuracy): {best_model} ({best_score:.4f})")
    
    # Test add custom model
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    y_proba = [0.9, 0.1, 0.85, 0.6, 0.2, 0.95, 0.3, 0.55, 0.8, 0.9]
    
    evaluator.add_model_metrics(
        'Test Model Custom',
        y_true, y_pred, y_proba,
        'Custom test model'
    )
    print(f"  [OK] Custom model added successfully")
    
    print("  [OK] Model comparison module working")
    return True


def test_usage_scenarios():
    """Test real-world usage scenarios"""
    print("\n[TEST 6] Real-World Usage Scenarios")
    print("=" * 50)
    
    # Scenario 1: Analyze depression text
    from src.evaluation.clinical_validity import analyze_clinical_validity
    
    text = "I've been feeling down for weeks. No energy, can't sleep, feel worthless."
    analysis = analyze_clinical_validity(text)
    
    print(f"  Scenario 1 - Quick Analysis:")
    print(f"    Severity: {analysis['clinical_summary']['severity']}")
    print(f"    PHQ-9 Score: {analysis['clinical_summary']['phq9_score']}")
    print(f"    [OK] Clinical analysis working")
    
    # Scenario 2: Compare multiple models
    from src.evaluation.model_comparison import get_evaluator
    
    evaluator = get_evaluator()
    models = evaluator.get_all_models()[:3]
    table = evaluator.get_metrics_summary_table(models)
    
    print(f"\n  Scenario 2 - Model Comparison Table:")
    print(f"    Table rows: {len(table)}")
    print(f"    [OK] Model comparison working")
    
    # Scenario 3: Evaluate explanation quality
    from src.evaluation.explainability_metrics import ExplainabilityMetrics
    
    pred = np.array([1, 1, 0, 1, 0])
    human = np.array([1, 1, 1, 1, 0])
    f1 = ExplainabilityMetrics.token_f1(pred, human)
    
    print(f"\n  Scenario 3 - Explanation Quality:")
    print(f"    Token-F1: {f1:.3f}")
    print(f"    [OK] Explainability evaluation working")
    
    print("\n  [OK] All usage scenarios working")
    return True


def main():
    """Run all evaluation tests"""
    print("\n" + "=" * 50)
    print("  EVALUATION FOLDER VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Metrics Module", test_metrics),
        ("Clinical Validity", test_clinical_validity),
        ("Explainability Metrics", test_explainability_metrics),
        ("Faithfulness Metrics", test_faithfulness_metrics),
        ("Model Comparison", test_model_comparison),
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
    
    # File status summary
    print("\n" + "=" * 50)
    print("  FILE STATUS & PURPOSE")
    print("=" * 50)
    
    files_info = [
        ("metrics.py", "REQUIRED", "Basic classification metrics (accuracy, F1, AUC)"),
        ("clinical_validity.py", "REQUIRED", "DSM-5 criteria & PHQ-9 score estimation"),
        ("explainability_metrics.py", "REQUIRED", "Plausibility metrics (Token-F1, AUPRC)"),
        ("faithfulness_metrics.py", "OPTIONAL", "Advanced faithfulness metrics (needs trained model)"),
        ("model_comparison.py", "REQUIRED", "Model performance comparison & visualization")
    ]
    
    print("\nFile                         Status      Purpose")
    print("-" * 70)
    for filename, status, purpose in files_info:
        status_color = "REQUIRED" if status == "REQUIRED" else "OPTIONAL"
        print(f"{filename:25} [{status_color:8}] {purpose}")
    
    print("\n" + "=" * 50)
    print("  USAGE SUMMARY")
    print("=" * 50)
    print("\nAll evaluation files are actively used:")
    print("  - metrics.py: Used by test_core.py, app.py")
    print("  - clinical_validity.py: Used by app.py, compare_groq_models.py")
    print("  - explainability_metrics.py: Used by evaluate_explanations.py")
    print("  - faithfulness_metrics.py: Used by test_new_features.py")
    print("  - model_comparison.py: Used by app.py (main Streamlit UI)")
    
    if passed == total:
        print("\n[SUCCESS] All evaluation modules validated!")
        return True
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
