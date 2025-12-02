# 🚀 Quick Start Commands

## Launch Streamlit App with Model Comparison

```bash
# Navigate to project directory
cd "C:\Users\Avinash rai\Downloads\Major proj AWA Proj\Major proj AWA"

# Run Streamlit app
streamlit run src/app/app.py
```

**Browser will open automatically at**: http://localhost:8501

---

## Test Model Comparison System

```bash
# Run comprehensive tests
python test_model_comparison.py
```

**Expected**: 7/7 tests PASSED ✅

---

## View Available Models

```bash
# List all models with metrics
python -c "from src.evaluation.model_comparison import get_evaluator; evaluator = get_evaluator(); print('\n'.join(evaluator.get_all_models()))"
```

---

## Add Custom Model Metrics

```python
from src.evaluation.model_comparison import get_evaluator

evaluator = get_evaluator()

# Your y_true, y_pred, y_proba from testing
evaluator.add_model_metrics(
    model_name="My Custom Model",
    y_true=[0, 1, 0, 1, 1, 0],  # True labels
    y_pred=[0, 1, 0, 0, 1, 0],  # Predictions
    y_proba=[0.2, 0.9, 0.3, 0.4, 0.8, 0.1],  # Probabilities
    description="My custom depression detector"
)

print("✅ Model added successfully!")
```

---

## View Metrics File

```bash
# Open metrics JSON
notepad outputs/model_metrics.json
```

---

## Install Dependencies (if needed)

```bash
# Core dependencies
pip install streamlit pandas numpy matplotlib scikit-learn

# Optional for explainability
pip install lime shap
```

---

## Common Tasks

### 1. Start Fresh
```bash
# Remove custom metrics and start with defaults
rm outputs/model_metrics.json
python -c "from src.evaluation.model_comparison import get_evaluator; get_evaluator()"
```

### 2. Export All Metrics to CSV
```python
from src.evaluation.model_comparison import get_evaluator
import pandas as pd

evaluator = get_evaluator()
table = evaluator.get_metrics_summary_table()
df = pd.DataFrame(table)
df.to_csv('all_model_metrics.csv', index=False)
print("✅ Exported to all_model_metrics.csv")
```

### 3. Find Best Model for Metric
```python
from src.evaluation.model_comparison import get_evaluator

evaluator = get_evaluator()

# For each metric
for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
    best_model, best_score = evaluator.get_best_model(metric)
    print(f"Best {metric:10s}: {best_model:30s} = {best_score:.4f}")
```

### 4. Compare Specific Models
```python
from src.evaluation.model_comparison import get_evaluator

evaluator = get_evaluator()

models = ['MentalBERT', 'GPT-4 (Few-Shot)', 'Ensemble (Best 3)']
comparison = evaluator.compare_models(models, metric='f1_score')

for model, score in comparison.items():
    print(f"{model:30s}: {score:.4f}")
```

---

## Troubleshooting

### Issue: Streamlit won't start
```bash
# Check if port 8501 is available
netstat -ano | findstr :8501

# Use different port
streamlit run src/app/app.py --server.port 8502
```

### Issue: Import errors
```bash
# Ensure you're in the project directory
cd "C:\Users\Avinash rai\Downloads\Major proj AWA Proj\Major proj AWA"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue: Missing dependencies
```bash
# Install all requirements
pip install -r requirements.txt
```

---

## Keyboard Shortcuts (Streamlit)

- **R**: Rerun the app
- **C**: Clear cache
- **Ctrl+C** (in terminal): Stop server
- **Ctrl+Shift+R**: Hard refresh in browser

---

## File Locations

```
Project Root: C:\Users\Avinash rai\Downloads\Major proj AWA Proj\Major proj AWA

Key Files:
  src/evaluation/model_comparison.py    - Core evaluation code
  src/app/app.py                        - Streamlit UI
  outputs/model_metrics.json            - Stored metrics
  test_model_comparison.py              - Test suite
  MODEL_COMPARISON_GUIDE.md             - Full documentation
  MODEL_COMPARISON_SUMMARY.md           - Implementation summary
```

---

## Quick Reference: Model Names

Copy-paste these exact names when selecting models:

```
DistilBERT (Emotion Proxy)
MentalBERT
DepRoBERTa
BERT-base-uncased
RoBERTa-base
GPT-4 (Few-Shot)
LLaMA-3-8B (Fine-tuned)
Classical ML (SVM)
Classical ML (Random Forest)
Ensemble (Best 3)
```

---

## Useful Python Snippets

### Get All Metrics for One Model
```python
from src.evaluation.model_comparison import get_evaluator

evaluator = get_evaluator()
metrics = evaluator.get_model_metrics('MentalBERT')
print(metrics)
```

### Compare Two Models
```python
from src.evaluation.model_comparison import get_evaluator

evaluator = get_evaluator()

model1 = evaluator.get_model_metrics('MentalBERT')
model2 = evaluator.get_model_metrics('GPT-4 (Few-Shot)')

print(f"MentalBERT F1: {model1['f1_score']:.4f}")
print(f"GPT-4 F1: {model2['f1_score']:.4f}")
print(f"Difference: {abs(model1['f1_score'] - model2['f1_score']):.4f}")
```

### Generate Mock Test Data
```python
from src.evaluation.model_comparison import create_mock_predictions

y_true, y_pred, y_proba = create_mock_predictions('MentalBERT', n_samples=100)

print(f"Generated {len(y_true)} samples")
print(f"True labels: {y_true[:10]}")
print(f"Predictions: {y_pred[:10]}")
print(f"Probabilities: {y_proba[:10]}")
```

---

## Documentation Links

- **Full Guide**: MODEL_COMPARISON_GUIDE.md
- **Implementation Summary**: MODEL_COMPARISON_SUMMARY.md
- **UI Preview**: UI_PREVIEW.md
- **Test Script**: test_model_comparison.py

---

## Support

If you encounter issues:

1. ✅ Run tests: `python test_model_comparison.py`
2. ✅ Check documentation: `MODEL_COMPARISON_GUIDE.md`
3. ✅ Verify imports: `python -c "from src.evaluation.model_comparison import get_evaluator"`
4. ✅ Check metrics file: `outputs/model_metrics.json`

---

**Last Updated**: November 25, 2025  
**Status**: ✅ Production Ready
