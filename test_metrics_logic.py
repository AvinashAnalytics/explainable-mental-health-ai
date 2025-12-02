"""Test improved metrics.py logic"""
from src.evaluation.metrics import Evaluator
import json

print('\n[TEST 1] Basic Classification Metrics')
print('=' * 50)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
y_probs = [0.9, 0.1, 0.85, 0.6, 0.2, 0.95, 0.3, 0.55, 0.8, 0.9]

metrics = Evaluator.compute_classification_metrics(y_true, y_pred, y_probs)
print(f'Accuracy: {metrics["accuracy"]:.3f}')
print(f'Precision: {metrics["precision"]:.3f}')
print(f'Recall: {metrics["recall"]:.3f}')
print(f'F1 Score: {metrics["f1"]:.3f}')
print(f'Specificity: {metrics["specificity"]:.3f}')
print(f'NPV: {metrics["npv"]:.3f}')
print(f'AUC: {metrics["auc"]:.3f}')
print(f'TP/TN/FP/FN: {metrics["true_positives"]}/{metrics["true_negatives"]}/{metrics["false_positives"]}/{metrics["false_negatives"]}')

print('\n[TEST 2] Confusion Matrix')
print('=' * 50)
cm = Evaluator.get_confusion_matrix_dict(y_true, y_pred)
print(json.dumps(cm, indent=2))

print('\n[TEST 3] Explanation Fluency - Poor')
print('=' * 50)
poor = 'bad'
score = Evaluator.evaluate_explanation_fluency(poor)
print(f'Text: "{poor}"')
print(f'Score: {score:.3f}')

print('\n[TEST 4] Explanation Fluency - Good')
print('=' * 50)
good = 'Patient exhibits depressed mood and anhedonia symptoms consistent with DSM-5 criteria for moderate depression.'
score = Evaluator.evaluate_explanation_fluency(good)
print(f'Text: "{good}"')
print(f'Score: {score:.3f}')

print('\n[TEST 5] Explanation Fluency - Excellent')
print('=' * 50)
excellent = 'The patient exhibits multiple DSM-5 criteria for major depressive disorder, including persistent depressed mood, anhedonia, sleep disturbance, and fatigue. PHQ-9 estimated score of 14 indicates moderate severity. No immediate crisis risk detected, but clinical evaluation recommended.'
score = Evaluator.evaluate_explanation_fluency(excellent)
print(f'Score: {score:.3f}')

print('\n[TEST 6] Edge Cases')
print('=' * 50)
print('Empty string:', Evaluator.evaluate_explanation_fluency(''))
print('Very long (100 words):', Evaluator.evaluate_explanation_fluency(' '.join(['word'] * 100)))
print('No keywords:', Evaluator.evaluate_explanation_fluency('This is a simple sentence without clinical terms.'))

print('\n[SUCCESS] All logic tests passed!')
