# 🚀 Model Training Status

## Training Started: November 25, 2024

### Configuration

**Dataset:** `data/merged_real_dataset.csv`
- Total available: 22,074 samples
- **Training with:** 1,000 samples (sampled for CPU efficiency)
- Train/Test split: 800 / 200 (80% / 20%)

**Label Distribution (1000 samples):**
- Class 0 (Control): 562 samples (56.2%)
- Class 1 (Depression): 438 samples (43.8%)

### Models Being Trained

#### 1. **DistilBERT**
- **Model:** `distilbert-base-uncased`
- **Parameters:** ~67M
- **Description:** Lightweight BERT variant
- **Config:**
  - Epochs: 2
  - Batch Size: 8
  - Learning Rate: 2e-5
- **Est. Time:** 15-20 minutes

#### 2. **DistilRoBERTa-Emotion**
- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Parameters:** ~82M
- **Description:** Pre-trained on emotion data
- **Config:**
  - Epochs: 2
  - Batch Size: 8
  - Learning Rate: 2e-5
- **Est. Time:** 15-20 minutes

### Total Estimated Time: 30-45 minutes

---

## What's Happening

### Current Process

1. ✅ **Data Loading** - Complete
   - Loaded 22,074 samples
   - Sampled 1,000 for training
   - Split into 800 train / 200 test

2. ✅ **Data Splitting** - Complete
   - Stratified split (maintains label distribution)
   - Saved to `data/splits/`

3. 🔄 **Model Training** - In Progress
   - Loading tokenizers and models
   - Tokenizing datasets
   - Training with HuggingFace Trainer
   - Evaluating on test set
   - Saving trained models

4. ⏳ **Testing** - Pending
   - Sample text predictions
   - Performance metrics
   - Comparison charts

5. ⏳ **Report Generation** - Pending
   - JSON report with full results
   - CSV summary table
   - Model rankings

---

## Training Process Details

### Steps Per Model

1. **Load Model & Tokenizer**
   - Download from HuggingFace Hub (if not cached)
   - Initialize for binary classification

2. **Tokenize Data**
   - Convert text to input IDs
   - Apply truncation (max length: 256)
   - Batch processing

3. **Train**
   - 2 epochs over training data
   - Batch size: 8
   - Gradient updates every batch
   - Evaluation after each epoch

4. **Evaluate**
   - Test on 200 held-out samples
   - Compute metrics:
     - Accuracy
     - F1 Score
     - Precision
     - Recall

5. **Save Model**
   - Save to `models/trained/[model_name]/`
   - Include tokenizer config

### Current Status

**Training Device:** CPU  
**Status:** Running Model 1/2 (DistilBERT)  
**Progress:** Tokenizing data...

---

## What to Expect

### Output Files

After training completes, you'll find:

1. **Trained Models**
   - `models/trained/distilbert/`
   - `models/trained/distilroberta-emotion/`
   - Each contains model weights + tokenizer

2. **Data Splits**
   - `data/splits/train.csv` (800 samples)
   - `data/splits/test.csv` (200 samples)

3. **Training Report**
   - `outputs/training_report_[timestamp].json`
   - Complete results with all metrics
   - Sample predictions

4. **Summary CSV**
   - `outputs/training_summary.csv`
   - Quick comparison table
   - Model rankings

### Metrics You'll See

For each model:
- **Accuracy:** Overall correctness
- **F1 Score:** Balance of precision/recall
- **Precision:** True positives / predicted positives
- **Recall:** True positives / actual positives
- **Training Loss:** Model's error during training
- **Evaluation Loss:** Model's error on test set
- **Training Time:** Minutes spent training

### Sample Predictions

Models will be tested on 6 sample texts:
1. Depressive text (should predict: Depression)
2. Positive text (should predict: Control)
3. Sleep issues (Depression)
4. Happy text (Control)
5. Suicidal ideation (Depression - High priority)
6. Gratitude text (Control)

---

## Next Steps (After Training)

### 1. Review Results
```bash
# Check the training report
python -c "import json; print(json.dumps(json.load(open('outputs/training_report_*.json')), indent=2))"
```

### 2. Compare Models
```bash
# View summary table
python -c "import pandas as pd; print(pd.read_csv('outputs/training_summary.csv'))"
```

### 3. Test in Enhanced App
```bash
# Launch Streamlit app
streamlit run src\app\app_enhanced.py
```

### 4. Use Best Model
- Models are auto-loaded in app
- Compare with rule-based and LLM
- Choose best for your use case

---

## Troubleshooting

### If Training is Interrupted

The script handles KeyboardInterrupt gracefully:
- Saves progress so far
- Marks interrupted models as failed
- Continues with report generation

### If Training is Too Slow

Current settings are optimized for CPU:
- 1,000 samples (manageable size)
- 2 epochs (quick training)
- Batch size 8 (memory efficient)

To speed up further:
- Reduce sample size (e.g., 500)
- Use only 1 model
- Reduce epochs to 1

To improve quality:
- Use full 22,074 samples (requires GPU)
- Increase epochs to 3-5
- Larger batch size (16-32)

### If Memory Issues

Reduce batch size:
- Change `batch_size: 8` to `batch_size: 4`
- Or reduce sample size to 500

---

## Expected Results

### Typical Performance (1000 samples, 2 epochs)

**DistilBERT:**
- Accuracy: 75-85%
- F1 Score: 70-80%
- Time: 15-20 min

**DistilRoBERTa-Emotion:**
- Accuracy: 78-88%
- F1 Score: 73-83%
- Time: 15-20 min

*Note: Emotion-pretrained model typically performs better as it understands emotional language*

### Best Practices

1. **Always use test set for final evaluation**
   - Training metrics can be misleading
   - Test set shows real-world performance

2. **Consider class imbalance**
   - 56% Control, 44% Depression (balanced)
   - F1 Score is better metric than accuracy

3. **Look at both precision and recall**
   - High precision: Few false positives
   - High recall: Catches most depression cases

4. **Clinical context matters**
   - False negatives (missing depression) are worse than false positives
   - Prioritize recall in mental health applications

---

## Progress Monitoring

### How to Check Progress

**Terminal Output:**
- Real-time training logs
- Loss values decreasing
- Evaluation after each epoch

**Signs of Good Training:**
- Loss decreases over epochs
- Evaluation metrics improve
- Test accuracy > 75%

**Signs of Issues:**
- Loss not decreasing
- Evaluation worse than training (overfitting)
- Very slow progress (CPU limitation)

---

## After Training

### Integration with Enhanced App

The trained models will automatically appear in the Enhanced Streamlit App:

1. **Model Selection Dropdown**
   - Your trained models added to list
   - Shows accuracy and F1 score
   - Select for predictions

2. **Single Analysis Tab**
   - Use trained model for new texts
   - Compare with rule-based + LLM
   - View confidence scores

3. **Model Comparison Tab**
   - Trained models in metrics table
   - Performance charts
   - Rankings by metric

### Using Trained Models Programmatically

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your trained model
model_path = "models/trained/distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Make prediction
text = "I feel hopeless and can't sleep"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

print(f"Prediction: {'Depression' if pred == 1 else 'Control'}")
print(f"Confidence: {confidence:.2%}")
```

---

## Training Completion

**Status:** 🔄 **IN PROGRESS**

Training started successfully and is running in background.
Check terminal for real-time updates.

**Estimated completion:** ~30-45 minutes from start time

---

*Last Updated: Training in progress...*
*Check back in 30-45 minutes for results!*
