# 🧠 Enhanced Depression Detection App

## Overview

Advanced Streamlit web application for AI-powered depression detection with multiple models, LLM integration, and comprehensive explainability features.

## 🌟 Key Features

### 1. **Multiple LLM Providers**
- **OpenAI** (GPT-4, GPT-4o-mini, GPT-3.5-turbo)
- **Groq** (Free ultra-fast inference: Llama-3, Mixtral, Gemma)
- **HuggingFace** (Any public/private model)
- **Mock** (Testing without API keys)

### 2. **Direct API Key Input**
- Secure in-app API key configuration
- No environment variable setup needed
- Real-time validation
- Provider-specific instructions

### 3. **Classical ML Models**
- DistilBERT
- MentalBERT
- DepRoBERTa
- emotion-english-distilroberta-base
- Performance metrics display

### 4. **Multi-Method Analysis**
- **Rule-Based:** 153 clinical patterns (DSM-5/PHQ-9)
- **Classical ML:** BERT-based models with attention
- **LLM:** GPT/Llama reasoning and explanations
- **Ensemble:** Combined assessment

### 5. **Enhanced Visualizations**
- Severity comparison charts
- Key term detection
- Confidence intervals
- Probability distributions
- Word clouds
- Attention heatmaps

### 6. **Batch Processing**
- Process multiple texts at once
- Dataset loading (Dreaddit, eRisk, CLPsych)
- Distribution analysis
- Export results to CSV

### 7. **Explainability Features**
- Symptom-level explanations
- Clinical justifications
- Token importance
- Faithfulness metrics
- Perturbation tests

### 8. **Safety Features**
- Crisis keyword detection
- Suicidal ideation alerts
- Emergency hotline display
- Clinical threshold warnings

### 9. **Export Functionality**
- JSON export (complete analysis)
- CSV export (summary)
- Batch results download
- Model comparison tables

## 🚀 Quick Start

### Method 1: Windows Launcher
```bash
# Double-click the launcher
run_enhanced_app.bat
```

### Method 2: Command Line
```bash
# Navigate to project root
cd "c:\Users\Avinash rai\Downloads\Major proj AWA Proj\Major proj AWA"

# Run the enhanced app
streamlit run src\app\app_enhanced.py
```

### Method 3: PowerShell
```powershell
# Single command
cd "c:\Users\Avinash rai\Downloads\Major proj AWA Proj\Major proj AWA"; streamlit run src\app\app_enhanced.py
```

## 📋 Configuration

### LLM Setup

#### OpenAI
1. Select "OpenAI" in sidebar
2. Enter API key (starts with `sk-`)
3. Get key: https://platform.openai.com/api-keys
4. Choose model (GPT-4o-mini recommended)

#### Groq (Free)
1. Select "Groq" in sidebar
2. Enter API key (starts with `gsk_`)
3. Get free key: https://console.groq.com
4. Choose model (llama-3.1-70b-versatile recommended)

#### HuggingFace
1. Select "HuggingFace" in sidebar
2. Enter token (optional, starts with `hf_`)
3. Specify model name
4. Uses local inference

### Classical Model Setup
- Pre-configured models loaded automatically
- Select from dropdown in sidebar
- View accuracy and F1 metrics
- No additional configuration needed

### Dataset Configuration
- **Dreaddit:** `data/dreaddit_sample.csv` (default)
- **eRisk:** Optional, specify path
- **CLPsych:** Optional, specify path

## 🎯 Usage Guide

### Single Text Analysis

1. **Navigate to "Analyze" tab**
2. **Enter text** in the text area
   - Or click "Load Example" for sample text
3. **Configure settings** in sidebar:
   - Select LLM provider (optional)
   - Enter API key if using LLM
   - Choose classical model
   - Enable/disable safety features
4. **Click "Analyze"** button
5. **View results** in three columns:
   - Rule-based (DSM-5/PHQ-9)
   - Classical ML prediction
   - LLM analysis (if enabled)
6. **Review visualizations**:
   - Severity comparison chart
   - Key terms detected
7. **Export results**:
   - Download JSON for complete analysis
   - Download CSV for summary

### Batch Processing

1. **Navigate to "Batch" tab**
2. **Configure dataset paths** in sidebar
3. **Select sample size** (10-1000)
4. **Click "Run Batch Analysis"**
5. **View results**:
   - Preview table
   - Distribution chart
   - Word cloud
6. **Download results** CSV

### Model Comparison

1. **Navigate to "Compare" tab**
2. **Select models** to compare (default: top 5)
3. **View performance table**
4. **Analyze charts**:
   - Multi-metric comparison
   - Bar chart by model
5. **Check top performers**:
   - Best accuracy
   - Best F1 score
   - Best precision

### Visual Insights

1. **Navigate to "Insights" tab**
2. **View faithfulness metrics**:
   - Explanation token overlap
   - Distribution histogram
3. **Check perturbation tests**:
   - Robustness to text changes
   - Sensitivity analysis
4. **Combined metrics** chart

## 🎨 UI Features

### Custom Styling
- Gradient headers
- Color-coded severity levels
- Professional metric cards
- Responsive layout
- Dark mode compatible

### Severity Color Coding
- 🔴 **Red:** High/Severe depression
- 🟡 **Yellow:** Moderate depression
- 🟢 **Green:** Mild depression
- ✅ **Blue:** No depression detected

### Interactive Elements
- Real-time API validation
- Progress indicators
- Expandable sections
- Tooltips and help text
- Responsive charts

## 📊 Output Formats

### JSON Export
```json
{
  "text": "User input text...",
  "rule_based": {
    "prediction": "Moderate depression risk",
    "explanation": ["Symptom 1", "Symptom 2"],
    "symptom_count": 4
  },
  "ml_model": "distilbert-mental-health",
  "ml_confidence": 0.82,
  "llm_result": {
    "depression_likelihood": "moderate",
    "explanation": "Detailed reasoning...",
    "detected_symptoms": [...]
  }
}
```

### CSV Export
```csv
Text,Prediction,Symptoms,Model,Confidence
"I feel hopeless...",Moderate depression risk,4,distilbert-mental-health,82%
```

## 🛡️ Safety & Privacy

### Data Security
- API keys stored in session only
- No data persistence
- Local processing when possible
- No external logging

### Clinical Safety
- Crisis detection enabled by default
- Immediate hotline display
- Professional referral recommendations
- Clear disclaimers

### Privacy Features
- No text storage
- Optional API usage
- Local model inference
- User-controlled exports

## 🔧 Troubleshooting

### Common Issues

**Issue:** "Invalid API key format"
- **Solution:** Check key starts with correct prefix (sk- for OpenAI, gsk_ for Groq)

**Issue:** "Model loading error"
- **Solution:** Check internet connection, ensure model files exist

**Issue:** "LLM timeout"
- **Solution:** Use faster model (GPT-4o-mini, llama-3.1-8b-instant)

**Issue:** "Charts not displaying"
- **Solution:** Check matplotlib installation: `pip install matplotlib seaborn`

**Issue:** "Dataset not found"
- **Solution:** Verify file path, ensure CSV is in correct location

## 📈 Performance Tips

### For Faster Analysis
1. Use Groq (free, ultra-fast)
2. Select GPT-4o-mini over GPT-4
3. Choose lighter classical models (DistilBERT)
4. Reduce batch size for processing
5. Disable attention visualization

### For Better Accuracy
1. Use GPT-4 or llama-3.1-70b
2. Enable all analysis methods
3. Use ensemble voting
4. Enable calibrated predictions
5. Increase temperature for creative reasoning

### For Cost Optimization
1. Use Groq (free) instead of OpenAI
2. Use HuggingFace local models
3. Rule-based only (no LLM)
4. Batch processing for multiple texts
5. Cache results when possible

## 🎯 Best Practices

### Input Text Guidelines
- Minimum 50 characters for meaningful analysis
- Include emotional/behavioral details
- Avoid single-word responses
- Context improves accuracy

### Model Selection
- **Highest Accuracy:** Use ensemble of all methods
- **Fastest:** Rule-based only
- **Most Explainable:** Rule-based + LLM
- **Research:** All methods with exports

### Prompt Templates
- **zero_shot.txt:** Quick, direct analysis
- **few_shot.txt:** Better with examples
- **cot.txt:** Step-by-step reasoning
- **emotion_cot.txt:** Emotion-focused (recommended)
- **instruction.txt:** Comprehensive clinical framework

## 📚 Technical Details

### Architecture
```
Frontend: Streamlit
├── Single Analysis
├── Batch Processing
├── Visual Insights
├── Model Comparison
└── About/Documentation

Backend:
├── Rule-based (src/explainability/rule_explainer.py)
├── Classical ML (src/models/classical.py)
├── LLM Adapter (src/models/llm_adapter.py)
├── Evaluation (src/evaluation/)
└── Datasets (src/data/)
```

### Dependencies
- streamlit >= 1.28.0
- torch >= 2.0.0
- transformers >= 4.30.0
- openai >= 1.0.0
- groq >= 0.4.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0
- numpy >= 1.24.0

### Supported Browsers
- Chrome (recommended)
- Firefox
- Edge
- Safari

## 🚀 Advanced Features

### Custom Model Integration
1. Add model to `src/evaluation/model_comparison.py`
2. Register metrics
3. Reload app
4. Select from dropdown

### Custom Prompts
1. Create `.txt` file in `config/prompts/`
2. Use {{text}} placeholder
3. Add to template dropdown
4. Reload app

### Dataset Integration
1. Prepare CSV with columns: `text`, `label`
2. Place in `data/` folder
3. Configure path in sidebar
4. Load for batch processing

## 📞 Support & Resources

### Documentation
- **DSM-5:** https://www.psychiatry.org/psychiatrists/practice/dsm
- **PHQ-9:** https://www.phqscreeners.com
- **OpenAI API:** https://platform.openai.com/docs
- **Groq API:** https://console.groq.com/docs

### Crisis Resources
- **US:** 988 (Suicide & Crisis Lifeline)
- **India:** 9152987821 (AASRA)
- **UK:** 116 123 (Samaritans)
- **International:** https://findahelpline.com

## ⚠️ Important Disclaimers

1. **Not a Diagnostic Tool**
   - This system is for research and educational purposes
   - Not a substitute for professional diagnosis
   - Always consult qualified mental health professionals

2. **Clinical Limitations**
   - AI cannot replace human judgment
   - Cultural and linguistic variations may affect accuracy
   - Context matters - brief texts may be insufficient

3. **Privacy & Ethics**
   - Use responsibly and ethically
   - Obtain consent when analyzing others' text
   - Follow data protection regulations
   - Respect user privacy

4. **Research Use Only**
   - Not approved for clinical deployment
   - Requires validation for specific use cases
   - Model biases may exist

## 📄 License

Research & Educational Use Only

## 🙏 Acknowledgments

- DSM-5 Clinical Framework
- PHQ-9 Screening Tool
- BERT Architecture (Devlin et al.)
- Mental Health NLP Research Community
- OpenAI, Groq, HuggingFace Teams

---

**Version:** 2.0  
**Last Updated:** November 2024  
**Status:** Active Development

For issues, suggestions, or contributions, please contact the development team.
