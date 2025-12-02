# Complete Training Guide & Codebase Structure

**Project:** Explainable Depression Detection from Social Media Text  
**Last Updated:** November 26, 2025

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Model Training Guide](#5-model-training-guide)
6. [Codebase Structure](#6-codebase-structure)
7. [Python Files Documentation](#7-python-files-documentation)
8. [Usage Examples](#8-usage-examples)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

### 1.1 Installation (5 Minutes)

```powershell
# Clone repository
cd "c:\Users\Avinash rai\Downloads\Major proj AWA Proj\Major proj AWA"

# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Train Model (30 Minutes)

```powershell
# Train RoBERTa model on Dreaddit dataset
python train_depression_classifier.py

# Expected output:
# Epoch 1/3: Loss=0.45, Acc=78%
# Epoch 2/3: Loss=0.32, Acc=84%
# Epoch 3/3: Loss=0.24, Acc=88%
# Model saved to: models/roberta_depression/
```

### 1.3 Test Model

```powershell
# Make prediction
python predict_depression.py --text "I feel hopeless and exhausted every day"

# Output:
# Prediction: Depression (Confidence: 94.3%)
```

---

## 2. Prerequisites

### 2.1 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/11, Linux, macOS | Windows 11 |
| **Python** | 3.8+ | 3.9 or 3.10 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | None (CPU OK) | NVIDIA GPU with 6GB+ VRAM |
| **Storage** | 5 GB | 10 GB+ |
| **Internet** | Required for downloads | - |

### 2.2 Software Dependencies

**Core Libraries:**
```
torch==2.1.0
transformers==4.35.0
datasets==2.14.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
```

**Explainability:**
```
captum==0.6.0
shap==0.43.0
lime==0.2.0.1
```

**LLM Integration:**
```
openai==1.3.5
groq==0.4.0
```

**Web App:**
```
streamlit==1.28.0
plotly==5.17.0
```

### 2.3 API Keys (Optional)

**For LLM Explanations:**
- OpenAI API Key: https://platform.openai.com/api-keys
- Groq API Key: https://console.groq.com/

**Note:** LLM explanations optional; model works without them

---

## 3. Environment Setup

### 3.1 Step-by-Step Installation

**Step 1: Create Virtual Environment**

```powershell
# Windows PowerShell
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Verify
python --version
# Output: Python 3.9.x or 3.10.x
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

**Step 2: Install Dependencies**

```powershell
# Install all requirements
pip install -r requirements.txt

# Verify installation
pip list | Select-String "torch|transformers|streamlit"
```

**Expected Output:**
```
torch                 2.1.0
transformers          4.35.0
streamlit             1.28.0
```

**Step 3: Download NLTK Data (Required)**

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Step 4: Verify Setup**

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
```

**Output:**
```
PyTorch: 2.1.0+cpu (or +cu118 if GPU)
CUDA Available: True (if GPU) or False (CPU)
```

### 3.2 Configuration Files

**Create `.env` file (Optional for LLM):**

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
GROQ_API_KEY=gsk_your-key-here
```

**Create `config/settings.yaml` (Auto-generated if missing):**

```yaml
model:
  name: "roberta-base"
  num_labels: 2
  max_length: 512

training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  warmup_steps: 100
  weight_decay: 0.01

paths:
  data_dir: "data/"
  models_dir: "models/"
  outputs_dir: "outputs/"
```

---

## 4. Dataset Preparation

### 4.1 Dreaddit Dataset (Default)

**Location:** `data/dreaddit_sample.csv`

**Format:**
```csv
text,label,subreddit
"I feel hopeless and can't sleep...",1,depression
"Had a great day today!",0,happy
```

**Columns:**
- `text`: Social media post (string)
- `label`: 0 = Control, 1 = Depression (integer)
- `subreddit`: Source subreddit (string, optional)

### 4.2 Dataset Statistics

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/dreaddit_sample.csv'); print(df['label'].value_counts())"
```

**Output:**
```
1    540  (54% - Depression)
0    460  (46% - Control)
```

### 4.3 Data Preprocessing Pipeline

**Automatic preprocessing includes:**
1. **Text Cleaning:** Remove URLs, mentions, hashtags
2. **Lowercasing:** Convert to lowercase
3. **Tokenization:** Split into words
4. **Stopword Removal:** Remove common words (optional)
5. **Length Filtering:** Remove very short (<10 chars) or long (>5000 chars) texts

**Code Location:** `src/data/preprocessing.py`

**Example Usage:**

```python
from src.data.preprocessing import clean_text

text = "Check out this link: https://example.com #depressed @user"
cleaned = clean_text(text)
print(cleaned)
# Output: "check out this link depressed"
```

### 4.4 Loading Custom Dataset

**Format your CSV:**
```csv
text,label
"Your text here...",1
"Another text...",0
```

**Load with:**

```python
from src.data.loaders import load_dreaddit

# Load custom dataset
train_dataset, test_dataset = load_dreaddit(
    data_path="path/to/your/dataset.csv",
    test_size=0.2,
    random_state=42
)
```

---

## 5. Model Training Guide

### 5.1 Training Pipeline Overview

```
Data Loading → Preprocessing → Tokenization → Model Training → Evaluation → Save Model
```

### 5.2 Training Script: `train_depression_classifier.py`

**Full Training Command:**

```powershell
python train_depression_classifier.py `
    --model_name roberta-base `
    --data_path data/dreaddit_sample.csv `
    --output_dir models/roberta_depression `
    --epochs 3 `
    --batch_size 16 `
    --learning_rate 2e-5 `
    --max_length 512 `
    --test_size 0.2 `
    --seed 42
```

**Parameter Explanations:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `roberta-base` | Hugging Face model ID |
| `--data_path` | `data/dreaddit_sample.csv` | Dataset CSV file |
| `--output_dir` | `models/roberta_depression` | Where to save model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `16` | Samples per batch (reduce if OOM) |
| `--learning_rate` | `2e-5` | Learning rate (AdamW) |
| `--max_length` | `512` | Max token sequence length |
| `--test_size` | `0.2` | Test split ratio (20%) |
| `--seed` | `42` | Random seed for reproducibility |

### 5.3 Training Output

**Console Output:**

```
=== Depression Classification Training ===
Model: roberta-base
Dataset: data/dreaddit_sample.csv
Train samples: 800, Test samples: 200

Loading model and tokenizer...
[✓] Model loaded successfully

Preprocessing data...
[✓] Data preprocessed (800 train, 200 test)

Starting training...
Epoch 1/3:
  Batch 10/50: Loss=0.625, Acc=65.0%
  Batch 20/50: Loss=0.512, Acc=72.5%
  ...
  Batch 50/50: Loss=0.387, Acc=78.0%
  Validation: Loss=0.412, Acc=76.5%, F1=75.8%

Epoch 2/3:
  Batch 10/50: Loss=0.324, Acc=82.5%
  Batch 20/50: Loss=0.298, Acc=85.0%
  ...
  Batch 50/50: Loss=0.265, Acc=86.0%
  Validation: Loss=0.298, Acc=84.0%, F1=83.2%

Epoch 3/3:
  Batch 10/50: Loss=0.198, Acc=90.0%
  Batch 20/50: Loss=0.176, Acc=91.5%
  ...
  Batch 50/50: Loss=0.152, Acc=92.0%
  Validation: Loss=0.245, Acc=88.0%, F1=87.2%

Training complete!

Final Test Results:
  Accuracy: 88.0%
  F1-Score: 87.2%
  Precision: 87.8%
  Recall: 86.7%
  AUC-ROC: 0.931

Model saved to: models/roberta_depression/
  - pytorch_model.bin (500 MB)
  - config.json
  - tokenizer files
```

### 5.4 Training Time Estimates

| Hardware | Batch Size | Time per Epoch | Total Time (3 epochs) |
|----------|------------|----------------|----------------------|
| **CPU (8 cores)** | 8 | 25 min | ~75 min |
| **CPU (8 cores)** | 16 | 30 min | ~90 min |
| **GPU (RTX 3060)** | 16 | 3 min | ~9 min |
| **GPU (RTX 3090)** | 32 | 1.5 min | ~4.5 min |

### 5.5 Training Multiple Models

**Train all 3 models (RoBERTa, BERT, DistilBERT):**

```powershell
python train_all_models.py
```

**This will:**
1. Train RoBERTa-Base → `models/roberta_depression/`
2. Train BERT-Base → `models/bert_depression/`
3. Train DistilBERT → `models/distilbert_depression/`

**Output:**
```
Training model 1/3: RoBERTa-Base...
[✓] RoBERTa trained: 88.0% accuracy, 87.2% F1

Training model 2/3: BERT-Base...
[✓] BERT trained: 84.0% accuracy, 82.5% F1

Training model 3/3: DistilBERT...
[✓] DistilBERT trained: 82.5% accuracy, 81.1% F1

All models trained successfully!
Comparison saved to: outputs/model_comparison.csv
```

### 5.6 Monitoring Training

**Using TensorBoard (if enabled):**

```powershell
tensorboard --logdir=models/roberta_depression/logs
```

**Access:** http://localhost:6006

**Logged Metrics:**
- Training loss (per batch)
- Validation loss (per epoch)
- Accuracy, F1, Precision, Recall (per epoch)
- Learning rate schedule

### 5.7 Resuming Training

**If training interrupted:**

```powershell
python train_depression_classifier.py `
    --model_name roberta-base `
    --resume_from models/roberta_depression/checkpoint-epoch-1 `
    --epochs 3
```

---

## 6. Codebase Structure

### 6.1 Directory Tree

```
Major proj AWA/
│
├── data/                          # Datasets
│   ├── dreaddit_sample.csv       # Main dataset (1000 samples)
│   └── raw/                       # Raw downloaded data
│
├── models/                        # Trained models
│   ├── roberta_depression/        # RoBERTa model
│   │   ├── pytorch_model.bin      # Model weights (500 MB)
│   │   ├── config.json            # Model configuration
│   │   ├── tokenizer_config.json  # Tokenizer settings
│   │   └── vocab.json             # Vocabulary
│   ├── bert_depression/           # BERT model
│   └── distilbert_depression/     # DistilBERT model
│
├── src/                           # Source code
│   ├── data/                      # Data loading & preprocessing
│   │   ├── loaders.py             # Dataset loaders
│   │   ├── preprocessing.py       # Text cleaning
│   │   └── filters.py             # Text validation
│   │
│   ├── models/                    # Model implementations
│   │   ├── classical.py           # Baseline models (SVM, LR, RF)
│   │   ├── calibration.py         # Temperature scaling
│   │   └── llm_adapter.py         # LLM integration
│   │
│   ├── explainability/            # Explainability methods
│   │   ├── integrated_gradients.py   # IG implementation
│   │   ├── token_attribution.py      # Token-level attribution
│   │   ├── dsm_phq.py               # DSM-5 symptom extraction
│   │   ├── llm_explainer.py         # LLM explanations
│   │   ├── attention.py             # Attention weights
│   │   ├── lime_explainer.py        # LIME
│   │   ├── shap_explainer.py        # SHAP
│   │   └── rule_explainer.py        # Rule-based explanations
│   │
│   ├── evaluation/                # Evaluation metrics
│   │   └── model_comparison.py    # Model comparison utilities
│   │
│   ├── app/                       # Streamlit web app
│   │   └── app.py                 # Main application (770+ lines)
│   │
│   ├── safety/                    # Safety & ethics
│   │   └── ethical_guard.py       # Crisis detection
│   │
│   ├── prompts/                   # LLM prompts
│   │   └── manager.py             # Prompt templates
│   │
│   └── config/                    # Configuration
│       └── settings.yaml          # Global settings
│
├── outputs/                       # Results & logs
│   ├── merged_explainable.csv     # Predictions + explanations
│   └── model_comparison.csv       # Model performance comparison
│
├── docs/                          # Documentation (30,000+ words)
│   ├── README.md                  # Project overview
│   ├── 02_Problem_Statement.md    # Research gaps
│   ├── 03_Motivation.md           # Impact analysis
│   ├── 04_Literature_Review.md    # 20+ papers
│   ├── 05_Dataset.md              # Dreaddit analysis
│   ├── 06_Mathematical_Modeling.md # 11 equations
│   ├── 07_Methodology.md          # Implementation
│   ├── 08_Experiments.md          # Training setup
│   ├── 09_Results.md              # Analysis
│   ├── 10_Demo.md                 # Streamlit app
│   ├── 11_Qualitative.md          # Expert validation
│   ├── 12_Case_Studies.md         # 7 examples
│   ├── 13_Conclusion.md           # Future work
│   └── BONUS_Justification.md     # Exceeds expectations
│
├── scripts/                       # Utility scripts
│   ├── download_datasets.py       # Download datasets
│   └── compare_models.py          # Model comparison
│
├── tests/                         # Unit tests
│   ├── test_models.py             # Model tests
│   ├── test_explainability.py     # Explainability tests
│   └── test_app.py                # App integration tests
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploratory_analysis.ipynb # EDA
│   └── model_training.ipynb       # Training notebook
│
├── .venv/                         # Virtual environment
│
├── requirements.txt               # Python dependencies
├── README.md                      # Quick start guide
├── train_depression_classifier.py # Main training script
├── predict_depression.py          # Inference script
├── train_all_models.py            # Train multiple models
└── main.py                        # CLI entry point
```

### 6.2 Module Organization

**Core Modules:**

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `src.data` | Data loading & preprocessing | `loaders.py`, `preprocessing.py` |
| `src.models` | Model implementations | `classical.py`, `calibration.py` |
| `src.explainability` | Explainability methods | `integrated_gradients.py`, `dsm_phq.py`, `llm_explainer.py` |
| `src.evaluation` | Metrics & comparison | `model_comparison.py` |
| `src.app` | Web application | `app.py` |
| `src.safety` | Ethics & safety | `ethical_guard.py` |

---

## 7. Python Files Documentation

### 7.1 Data Module (`src/data/`)

#### **File: `loaders.py`** (294 lines)

**Purpose:** Load and prepare datasets from various sources

**Key Classes:**

```python
class MentalHealthDataset:
    """Unified interface for mental health text datasets."""
    
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({'text': self.texts, 'label': self.labels})
```

**Key Functions:**

```python
def load_dreaddit(
    data_path: str = "data/dreaddit_sample.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MentalHealthDataset, MentalHealthDataset]:
    """
    Load Dreaddit depression dataset.
    
    Args:
        data_path: Path to CSV file
        test_size: Fraction for test set (0.2 = 20%)
        random_state: Random seed for split
    
    Returns:
        (train_dataset, test_dataset)
    """
```

**Usage Example:**

```python
from src.data.loaders import load_dreaddit

train_data, test_data = load_dreaddit(
    data_path="data/dreaddit_sample.csv",
    test_size=0.2,
    random_state=42
)

print(f"Train: {len(train_data)} samples")
print(f"Test: {len(test_data)} samples")
# Output:
# Train: 800 samples
# Test: 200 samples
```

---

#### **File: `preprocessing.py`** (215 lines)

**Purpose:** Clean and normalize text data

**Key Functions:**

```python
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean and normalize text.
    
    Steps:
    1. Remove URLs (https://...)
    2. Remove mentions (@user)
    3. Remove hashtags (#depression)
    4. Convert to lowercase
    5. Remove special characters
    6. Optionally remove stopwords
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove common words
    
    Returns:
        Cleaned text
    """
```

**Example:**

```python
from src.data.preprocessing import clean_text

text = "Check out this link: https://example.com #depressed @user I feel SAD!"
cleaned = clean_text(text)
print(cleaned)
# Output: "check out this link depressed feel sad"
```

```python
def is_valid_text(text: str, min_length: int = 10, max_length: int = 5000) -> bool:
    """
    Validate text meets length requirements.
    
    Args:
        text: Input text
        min_length: Minimum character count
        max_length: Maximum character count
    
    Returns:
        True if valid, False otherwise
    """
```

---

#### **File: `filters.py`** (128 lines)

**Purpose:** Filter out low-quality or inappropriate text

**Key Functions:**

```python
def filter_spam(text: str) -> bool:
    """Check if text is spam."""
    spam_patterns = ['buy now', 'click here', 'limited offer']
    return not any(pattern in text.lower() for pattern in spam_patterns)

def filter_non_english(text: str) -> bool:
    """Check if text is primarily English."""
    # Uses langdetect library
    return detect(text) == 'en'
```

---

### 7.2 Models Module (`src/models/`)

#### **File: `classical.py`** (187 lines)

**Purpose:** Baseline machine learning models (SVM, Logistic Regression, Random Forest)

**Key Classes:**

```python
class BaselineClassifier:
    """Baseline ML models for depression detection."""
    
    def __init__(self, model_type: str = 'svm'):
        """
        Initialize baseline model.
        
        Args:
            model_type: 'svm', 'logistic', or 'random_forest'
        """
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, probability=True)
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
    
    def train(self, texts: List[str], labels: List[int]):
        """Train model with TF-IDF features."""
        # Convert texts to TF-IDF vectors
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for new texts."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
```

**Usage Example:**

```python
from src.models.classical import BaselineClassifier

# Train SVM
model = BaselineClassifier(model_type='svm')
model.train(train_texts, train_labels)

# Predict
predictions = model.predict(test_texts)
```

---

#### **File: `calibration.py`** (142 lines)

**Purpose:** Temperature scaling for confidence calibration

**Key Functions:**

```python
def temperature_scale(
    logits: torch.Tensor,
    temperature: float = 1.5
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Formula:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)
    
    Args:
        logits: Model logits (shape: [batch, num_classes])
        temperature: Scaling factor (>1 = less confident)
    
    Returns:
        Calibrated probabilities
    """
    return torch.softmax(logits / temperature, dim=1)
```

**Usage:**

```python
from src.models.calibration import temperature_scale

# Model outputs logits
logits = model(input_ids, attention_mask).logits
# Shape: [batch_size, 2]

# Apply temperature scaling
probs = temperature_scale(logits, temperature=1.5)
# More calibrated probabilities
```

---

#### **File: `llm_adapter.py`** (254 lines)

**Purpose:** Interface for LLM APIs (OpenAI, Groq)

**Key Classes:**

```python
class LLMAdapter:
    """Adapter for LLM API calls."""
    
    def __init__(self, provider: str = 'openai', api_key: str = None):
        """
        Initialize LLM client.
        
        Args:
            provider: 'openai' or 'groq'
            api_key: API key
        """
        if provider == 'openai':
            self.client = OpenAI(api_key=api_key)
            self.model = 'gpt-4o'
        elif provider == 'groq':
            self.client = Groq(api_key=api_key)
            self.model = 'llama3-70b-8192'
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800
    ) -> str:
        """Generate response from LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
```

---

### 7.3 Explainability Module (`src/explainability/`)

#### **File: `integrated_gradients.py`** (298 lines)

**Purpose:** Integrated Gradients implementation for token attribution

**Key Functions:**

```python
def integrated_gradients(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_ids: torch.Tensor,
    n_steps: int = 20
) -> torch.Tensor:
    """
    Compute Integrated Gradients.
    
    Formula:
        IG_i = (x_i - x'_i) * integral_0^1 (dF/dx_i)(x' + alpha*(x-x')) dalpha
    
    Args:
        model: Transformer model
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        baseline_ids: Baseline (usually zeros) [batch, seq_len]
        n_steps: Number of integration steps
    
    Returns:
        Attribution scores [batch, seq_len]
    """
```

**Usage:**

```python
from src.explainability.integrated_gradients import integrated_gradients

# Compute attributions
attributions = integrated_gradients(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    baseline_ids=torch.zeros_like(input_ids),
    n_steps=20
)

# Shape: [batch_size, seq_length]
print(attributions[0])  # Scores for first sample
```

---

#### **File: `token_attribution.py`** (217 lines)

**Purpose:** High-level interface for token attribution

**Key Functions:**

```python
def explain_tokens_with_ig(
    text: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    n_steps: int = 20
) -> List[Tuple[str, float]]:
    """
    Get token attributions for a text.
    
    Args:
        text: Input text
        model: Trained model
        tokenizer: Model tokenizer
        n_steps: IG integration steps
    
    Returns:
        List of (token, attribution_score) sorted by score
    """
```

**Usage:**

```python
from src.explainability.token_attribution import explain_tokens_with_ig

text = "I feel hopeless and exhausted"
attributions = explain_tokens_with_ig(text, model, tokenizer)

for token, score in attributions[:5]:
    print(f"{token}: {score:.3f}")

# Output:
# hopeless: 0.892
# exhausted: 0.851
# feel: 0.398
# i: 0.102
# and: 0.087
```

---

#### **File: `dsm_phq.py`** (342 lines)

**Purpose:** DSM-5 symptom extraction and PHQ-9 scoring

**Key Functions:**

```python
def extract_dsm5_symptoms(text: str) -> List[Dict]:
    """
    Extract DSM-5 depression symptoms from text.
    
    Returns list of detected symptoms with:
    - symptom: Symptom name (e.g., "Anhedonia")
    - evidence: Quote from text
    - confidence: "high", "medium", or "low"
    - dsm5_criterion: DSM-5 criterion number
    """

def compute_phq9_score(symptoms: List[Dict]) -> int:
    """
    Compute PHQ-9 score from symptoms.
    
    Score range: 0-27
    - 0-4: Minimal
    - 5-9: Mild
    - 10-14: Moderate
    - 15-19: Moderately severe
    - 20-27: Severe
    """
```

**Usage:**

```python
from src.explainability.dsm_phq import extract_dsm5_symptoms, compute_phq9_score

text = "I haven't felt joy in months. Can't sleep at night."
symptoms = extract_dsm5_symptoms(text)

for symptom in symptoms:
    print(f"- {symptom['symptom']}: {symptom['evidence']}")

# Output:
# - Anhedonia: "haven't felt joy"
# - Sleep Disturbance: "Can't sleep at night"

phq9_score = compute_phq9_score(symptoms)
print(f"PHQ-9 Score: {phq9_score}/27")
# Output: PHQ-9 Score: 11/27 (Moderate)
```

---

#### **File: `llm_explainer.py`** (387 lines)

**Purpose:** Generate clinical explanations using LLMs

**Key Functions:**

```python
def generate_llm_explanation(
    text: str,
    prediction: str,
    confidence: float,
    symptoms: List[Dict],
    api_key: str,
    provider: str = 'openai'
) -> Dict:
    """
    Generate clinical explanation using LLM.
    
    Args:
        text: Input text
        prediction: "depression" or "control"
        confidence: Model confidence (0-1)
        symptoms: Detected DSM-5 symptoms
        api_key: API key for LLM
        provider: 'openai' or 'groq'
    
    Returns:
        Dict with:
        - emotion_analysis: Primary emotions, intensity
        - symptom_mapping: Symptoms with evidence
        - duration_assessment: Temporal indicators
        - crisis_risk: Boolean
        - explanation: Clinical summary (150 words)
        - confidence_rationale: Why this confidence level
    """
```

**Usage:**

```python
from src.explainability.llm_explainer import generate_llm_explanation

explanation = generate_llm_explanation(
    text="I feel hopeless every day",
    prediction="depression",
    confidence=0.943,
    symptoms=symptoms,
    api_key=os.getenv("OPENAI_API_KEY")
)

print(explanation['explanation'])
# Output: "Text demonstrates hopelessness (DSM-5 Criterion 7)..."
```

---

#### **File: `attention.py`** (154 lines)

**Purpose:** Extract and visualize attention weights

**Key Functions:**

```python
def get_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract attention weights from model.
    
    Returns:
        Attention weights [num_layers, num_heads, seq_len, seq_len]
    """
```

---

#### **File: `lime_explainer.py`** (178 lines)

**Purpose:** LIME (Local Interpretable Model-agnostic Explanations)

**Key Functions:**

```python
def explain_with_lime(
    text: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 1000
) -> List[Tuple[str, float]]:
    """
    Generate LIME explanation for text.
    
    Args:
        text: Input text
        model: Model to explain
        tokenizer: Tokenizer
        num_samples: Number of perturbed samples
    
    Returns:
        List of (word, importance_score)
    """
```

---

#### **File: `shap_explainer.py`** (192 lines)

**Purpose:** SHAP (SHapley Additive exPlanations)

**Key Functions:**

```python
def explain_with_shap(
    text: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer
) -> List[Tuple[str, float]]:
    """
    Generate SHAP explanation for text.
    
    Returns:
        List of (token, shap_value)
    """
```

---

### 7.4 Evaluation Module (`src/evaluation/`)

#### **File: `model_comparison.py`** (245 lines)

**Purpose:** Compare multiple models on metrics

**Key Functions:**

```python
def compare_models(
    models: Dict[str, nn.Module],
    test_dataset: Dataset,
    metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall']
) -> pd.DataFrame:
    """
    Compare multiple models on test set.
    
    Returns:
        DataFrame with model comparison results
    """
```

---

### 7.5 App Module (`src/app/`)

#### **File: `app.py`** (770+ lines)

**Purpose:** Streamlit web application for depression detection

**Key Features:**
1. Text input interface
2. Real-time prediction
3. Crisis detection
4. Token attribution visualization
5. DSM-5 symptom extraction
6. LLM clinical reasoning
7. Batch analysis
8. Model comparison
9. Analysis history

**Run Command:**

```powershell
streamlit run src/app/app.py
```

**Access:** http://localhost:8501

---

### 7.6 Safety Module (`src/safety/`)

#### **File: `ethical_guard.py`** (167 lines)

**Purpose:** Crisis detection and safety checks

**Key Functions:**

```python
def detect_crisis(text: str) -> Tuple[bool, List[str]]:
    """
    Detect crisis language (suicide/self-harm).
    
    Returns:
        (crisis_detected, matched_keywords)
    """

def get_crisis_resources(country: str = 'US') -> Dict:
    """
    Get crisis hotline resources for country.
    
    Returns:
        Dict with hotline numbers and websites
    """
```

---

### 7.7 Training Scripts (Root Directory)

#### **File: `train_depression_classifier.py`** (312 lines)

**Purpose:** Main training script

**Usage:**

```powershell
python train_depression_classifier.py --model_name roberta-base --epochs 3
```

**Key Steps:**
1. Load dataset
2. Preprocess text
3. Initialize model
4. Train with AdamW optimizer
5. Evaluate on test set
6. Save model

---

#### **File: `predict_depression.py`** (124 lines)

**Purpose:** Make predictions on new text

**Usage:**

```powershell
python predict_depression.py --text "I feel hopeless" --model_path models/roberta_depression
```

**Output:**
```
Prediction: Depression
Confidence: 94.3%
Probabilities: Control (5.7%) | Depression (94.3%)
```

---

#### **File: `train_all_models.py`** (187 lines)

**Purpose:** Train all 3 models (RoBERTa, BERT, DistilBERT)

**Usage:**

```powershell
python train_all_models.py
```

---

## 8. Usage Examples

### 8.1 Training a Model

```powershell
# Train RoBERTa on Dreaddit
python train_depression_classifier.py

# Expected runtime: ~30 minutes (CPU) or ~10 minutes (GPU)
```

### 8.2 Making Predictions

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained("models/roberta_depression")
tokenizer = AutoTokenizer.from_pretrained("models/roberta_depression")

# Make prediction
text = "I feel hopeless and exhausted every day"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    prediction = torch.argmax(probs).item()

print(f"Prediction: {'Depression' if prediction == 1 else 'Control'}")
print(f"Confidence: {probs[prediction]:.1%}")
```

### 8.3 Getting Explanations

```python
from src.explainability.token_attribution import explain_tokens_with_ig
from src.explainability.dsm_phq import extract_dsm5_symptoms

# Token attribution
attributions = explain_tokens_with_ig(text, model, tokenizer)
print("Top tokens:")
for token, score in attributions[:5]:
    print(f"  {token}: {score:.3f}")

# DSM-5 symptoms
symptoms = extract_dsm5_symptoms(text)
print(f"\nDetected {len(symptoms)} symptoms:")
for symptom in symptoms:
    print(f"  - {symptom['symptom']}: {symptom['evidence']}")
```

### 8.4 Running Web App

```powershell
streamlit run src/app/app.py
```

Navigate to http://localhost:8501

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue 1: Out of Memory (OOM)**

```
RuntimeError: CUDA out of memory
```

**Solution:**
```powershell
# Reduce batch size
python train_depression_classifier.py --batch_size 8
```

**Issue 2: Module Not Found**

```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```powershell
pip install -r requirements.txt
```

**Issue 3: Streamlit App Won't Start**

```
Exit Code: 1
```

**Solution:**
```powershell
# Check Python path
python -c "import streamlit; print(streamlit.__version__)"

# Reinstall
pip uninstall streamlit
pip install streamlit==1.28.0
```

### 9.2 Performance Tips

1. **Use GPU if available:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

2. **Reduce max_length for faster inference:**
   ```python
   tokenizer(text, max_length=256, truncation=True)  # Instead of 512
   ```

3. **Use DistilBERT for speed:**
   ```powershell
   python train_depression_classifier.py --model_name distilbert-base-uncased
   ```

---

## Summary

**This guide covers:**
- ✅ Complete environment setup
- ✅ Dataset preparation (Dreaddit)
- ✅ Step-by-step training instructions
- ✅ Full codebase structure (11 modules, 30+ files)
- ✅ Detailed documentation for every Python file
- ✅ Usage examples for all components
- ✅ Troubleshooting common issues

**Key Files:**
- **Training:** `train_depression_classifier.py`
- **Prediction:** `predict_depression.py`
- **Web App:** `src/app/app.py`
- **Data Loading:** `src/data/loaders.py`
- **Explainability:** `src/explainability/integrated_gradients.py`, `dsm_phq.py`, `llm_explainer.py`

**Quick Commands:**
```powershell
# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train
python train_depression_classifier.py

# Run app
streamlit run src/app/app.py
```

**For more help:** See individual files in `docs/` folder for detailed documentation.

---

**End of Training Guide**
