# Explainability Metrics Suite

Implementation of **plausibility** and **faithfulness** metrics for evaluating model explanations in depression detection, based on the IIT Bombay hate-speech project (endsem_pres.pdf).

## Overview

This suite provides comprehensive evaluation of explainability methods including:
- **Attention supervision** (attention_supervision.py)
- **Explainability metrics** (explainability_metrics.py)
- **Human rationale annotations** (annotation templates)
- **Evaluation pipeline** (evaluate_explanations.py)

## Metrics

### Plausibility Metrics
*Do explanations match human rationales?*

1. **Token-F1**: F1 score between predicted and human rationales
   - Range: [0, 1], higher is better
   - Measures overlap between model attention and human annotations

2. **IOU-F1**: F1 based on Intersection-over-Union
   - Range: [0, 1], higher is better
   - IOU = |predicted ∩ human| / |predicted ∪ human|

3. **AUPRC**: Area under precision-recall curve
   - Range: [0, 1], higher is better
   - Uses continuous attention scores vs binary human rationales

### Faithfulness Metrics
*Do explanations reflect actual model behavior?*

4. **Sufficiency**: How much does keeping only rationale tokens preserve the prediction?
   - Formula: `Sufficiency = P(original) - P(only_rationales)`
   - Range: [-1, 1], **lower is better**
   - Ideal: 0 (rationales fully sufficient)

5. **Comprehensiveness**: How much does removing rationale tokens change the prediction?
   - Formula: `Comprehensiveness = P(original) - P(without_rationales)`
   - Range: [-1, 1], **higher is better**
   - Ideal: High positive value (removing rationales significantly changes prediction)

## Installation

```bash
# Install required packages
pip install torch transformers scikit-learn pandas numpy
```

## Usage

### Step 1: Create Annotation Template

Generate a CSV template for human annotation:

```bash
python create_annotation_template.py \
    --input data/dreaddit_sample.csv \
    --output data/rationale_annotations_template.csv \
    --num_samples 100 \
    --create_examples
```

**Output files:**
- `rationale_annotations_template.csv` - Empty template for annotation
- `example_annotations.csv` - Pre-filled examples for testing

### Step 2: Annotate Important Tokens

Open `rationale_annotations_template.csv` in Excel/Google Sheets and fill the `rationale_tokens` column.

**Format:**
```csv
id,text,label,rationale_tokens,notes
1,"I feel so hopeless and alone",1,"hopeless alone","Clear indicators"
2,"Having a great day",0,"","Positive sentiment"
```

**Annotation Guidelines:**
- Focus on depression indicators: hopelessness, sadness, pain, isolation
- Include negations: "can't", "won't", "nothing"
- 3-7 tokens per sample
- Separate tokens with spaces
- Leave empty for control samples

### Step 3: Evaluate Explanations

Run evaluation with your annotated data:

```bash
python evaluate_explanations.py \
    --annotations data/rationale_annotations.csv \
    --model distilbert-base-uncased \
    --output explainability_results.txt
```

**Example Output:**
```
EXPLAINABILITY EVALUATION RESULTS
================================================================================

📊 PLAUSIBILITY METRICS (Do explanations match human rationales?)
--------------------------------------------------------------------------------
Token-F1:    0.7523 ± 0.1245
IOU-F1:      0.7489 ± 0.1198
AUPRC:       0.8956 ± 0.0876

🔬 FAITHFULNESS METRICS (Do explanations reflect model behavior?)
--------------------------------------------------------------------------------
Sufficiency:        0.1234 ± 0.0567
  (Lower is better: rationales preserve prediction)
Comprehensiveness:  0.6789 ± 0.1023
  (Higher is better: removing rationales changes prediction)
```

## Code Examples

### Computing Metrics Directly

```python
from src.evaluation.explainability_metrics import ExplainabilityMetrics
import numpy as np

# Plausibility: Token-F1
predicted = np.array([1, 1, 0, 1, 0, 0, 1, 0])
human = np.array([1, 1, 1, 1, 0, 0, 0, 0])
token_f1 = ExplainabilityMetrics.token_f1(predicted, human)
print(f"Token-F1: {token_f1:.4f}")  # 0.7500

# Plausibility: AUPRC
scores = np.array([0.9, 0.85, 0.6, 0.8, 0.3, 0.2, 0.1, 0.15])
auprc = ExplainabilityMetrics.auprc(scores, human)
print(f"AUPRC: {auprc:.4f}")  # 1.0000

# Batch computation
predicted_batch = [np.array([1, 1, 0, 1, 0]), ...]
human_batch = [np.array([1, 1, 1, 1, 0]), ...]
scores_batch = [np.array([0.9, 0.85, 0.4, 0.8, 0.2]), ...]

plausibility = ExplainabilityMetrics.compute_all_plausibility(
    predicted_batch, human_batch, scores_batch
)
print(f"Token-F1: {plausibility['token_f1_mean']:.4f}")
```

### Faithfulness Metrics

```python
from src.evaluation.explainability_metrics import ExplainabilityMetrics

# Define model prediction function
def model_predict(text: str):
    # Your model inference
    return np.array([0.3, 0.7])  # [P(control), P(depression)]

# Sufficiency: Keep only rationale tokens
sufficiency = ExplainabilityMetrics.sufficiency(
    model_fn=model_predict,
    original_text="I feel so hopeless and alone",
    rationale_tokens=["hopeless", "alone"],
    tokenizer=tokenizer,
    predicted_class=1  # depression
)
print(f"Sufficiency: {sufficiency:.4f}")  # Lower is better

# Comprehensiveness: Remove rationale tokens
comprehensiveness = ExplainabilityMetrics.comprehensiveness(
    model_fn=model_predict,
    original_text="I feel so hopeless and alone",
    rationale_tokens=["hopeless", "alone"],
    tokenizer=tokenizer,
    predicted_class=1
)
print(f"Comprehensiveness: {comprehensiveness:.4f}")  # Higher is better
```

## Integration with Attention Supervision

Combine with attention supervision training:

```python
from src.explainability.attention_supervision import AttentionSupervisedTrainer
from src.evaluation.explainability_metrics import ExplainabilityMetrics

# Train with attention supervision
trainer = AttentionSupervisedTrainer(model, tokenizer, lambda_weight=1.0)
trainer.train_epoch(train_loader, optimizer, epoch=0)

# Evaluate attention quality
# (Extract attention weights and compare to human rationales)
plausibility = ExplainabilityMetrics.compute_all_plausibility(
    predicted_rationales=extracted_attention,
    human_rationales=annotated_rationales,
    predicted_scores=attention_scores
)

print(f"Attention supervision improves Token-F1: {plausibility['token_f1_mean']:.4f}")
```

## Experimental Results (Expected)

Based on hate-speech project benchmarks:

| Method | Token-F1 | IOU-F1 | AUPRC | Sufficiency↓ | Comprehensiveness↑ |
|--------|----------|---------|-------|--------------|-------------------|
| **No Supervision** | 0.45 | 0.42 | 0.68 | 0.35 | 0.41 |
| **λ = 0.001** | 0.52 | 0.49 | 0.73 | 0.28 | 0.49 |
| **λ = 1.0** | 0.68 | 0.65 | 0.82 | 0.18 | 0.62 |
| **λ = 100** | 0.75 | 0.72 | 0.89 | 0.12 | 0.68 |

- **λ = 100**: Best explainability (Token-F1: 0.75)
- **λ = 0.001**: Best accuracy (minimal impact on classification)

## File Structure

```
Major proj AWA/
├── src/
│   ├── evaluation/
│   │   └── explainability_metrics.py    # Metrics implementation
│   └── explainability/
│       └── attention_supervision.py     # Attention loss
├── create_annotation_template.py        # Generate annotation CSV
├── evaluate_explanations.py             # Full evaluation pipeline
├── data/
│   ├── rationale_annotations_template.csv
│   ├── example_annotations.csv
│   └── explainability_results.txt
└── EXPLAINABILITY_METRICS_README.md     # This file
```

## References

1. **IIT Bombay Hate-Speech Project** (endsem_pres.pdf)
   - Attention supervision with λ tuning
   - Token-F1, IOU-F1, AUPRC metrics
   - Sufficiency and Comprehensiveness

2. **DeYoung et al. (2020)**: ERASER: A Benchmark to Evaluate Rationalized NLP Models

3. **Jain & Wallace (2019)**: Attention is not Explanation

## Next Steps

1. **Annotate 100+ samples** for robust evaluation
2. **Train with attention supervision** (λ ∈ {0.001, 1, 100})
3. **Compare baseline vs supervised** models
4. **Visualize attention heatmaps** in Streamlit dashboard
5. **Implement Integrated Gradients** as alternative explanation method

## Contact

For questions about metrics implementation or annotation guidelines, refer to:
- `explainability_metrics.py` - Metric definitions
- `evaluate_explanations.py` - Usage examples
- `create_annotation_template.py` - Annotation instructions

---

**Status**: ✅ Implementation complete  
**Last Updated**: 2024  
**Version**: 1.0
