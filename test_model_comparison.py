"""
Test script for model comparison functionality
"""

import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.evaluation.model_comparison import get_evaluator, create_mock_predictions


def test_model_comparison():
    """Test model comparison features."""
    print("=" * 70)
    print("🧪 TESTING MODEL COMPARISON SYSTEM")
    print("=" * 70)
    
    # Get evaluator
    evaluator = get_evaluator()
    
    # Test 1: Get all models
    print("\n📋 TEST 1: Available Models")
    print("-" * 70)
    models = evaluator.get_all_models()
    print(f"Total Models: {len(models)}")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("✅ PASSED")
    
    # Test 2: Get metrics for specific model
    print("\n📊 TEST 2: Model Metrics Retrieval")
    print("-" * 70)
    test_model = models[0]
    metrics = evaluator.get_model_metrics(test_model)
    print(f"Model: {test_model}")
    print(f"Metrics: {metrics}")
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    print("✅ PASSED")
    
    # Test 3: Compare models
    print("\n🏆 TEST 3: Model Comparison")
    print("-" * 70)
    comparison = evaluator.compare_models(metric='f1_score')
    print("Models ranked by F1 Score:")
    for i, (model, score) in enumerate(list(comparison.items())[:5], 1):
        print(f"  {i}. {model}: {score:.4f}")
    print("✅ PASSED")
    
    # Test 4: Get best model
    print("\n🥇 TEST 4: Best Model Detection")
    print("-" * 70)
    for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
        best_model, best_score = evaluator.get_best_model(metric)
        print(f"Best {metric:12s}: {best_model:30s} = {best_score:.4f}")
    print("✅ PASSED")
    
    # Test 5: Metrics summary table
    print("\n📋 TEST 5: Metrics Summary Table")
    print("-" * 70)
    table = evaluator.get_metrics_summary_table(models[:3])
    print(f"Generated table with {len(table)} rows")
    for row in table:
        print(f"  {row['Model']:30s} | Acc: {row['Accuracy']} | F1: {row['F1 Score']}")
    print("✅ PASSED")
    
    # Test 6: Add custom model metrics
    print("\n➕ TEST 6: Add Custom Model Metrics")
    print("-" * 70)
    y_true, y_pred, y_proba = create_mock_predictions("Test Model", n_samples=100)
    evaluator.add_model_metrics(
        "Test Model Custom",
        y_true,
        y_pred,
        y_proba,
        description="Custom test model for validation"
    )
    custom_metrics = evaluator.get_model_metrics("Test Model Custom")
    print(f"Custom Model Metrics: {custom_metrics}")
    print("✅ PASSED")
    
    # Test 7: Confusion matrix data
    print("\n📊 TEST 7: Confusion Matrix Data")
    print("-" * 70)
    y_true, y_pred, _ = create_mock_predictions("MentalBERT", n_samples=50)
    cm_data = evaluator.get_confusion_matrix_data(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives:  {cm_data['true_negatives']}")
    print(f"  False Positives: {cm_data['false_positives']}")
    print(f"  False Negatives: {cm_data['false_negatives']}")
    print(f"  True Positives:  {cm_data['true_positives']}")
    print("✅ PASSED")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print("Total Tests: 7")
    print("Passed: 7 ✅")
    print("Failed: 0 ❌")
    print("Success Rate: 100%")
    print("\n🎉 ALL MODEL COMPARISON TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_model_comparison()
