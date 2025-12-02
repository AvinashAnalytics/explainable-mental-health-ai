# 📖 User Guide - Explainable Mental Health AI Platform

## 🎯 Welcome

This guide will help you use the Explainable Depression Detection AI system effectively and safely.

**⚠️ IMPORTANT**: This is a research tool, NOT a diagnostic system. Always consult qualified mental health professionals for clinical decisions.

---

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Interface Overview](#interface-overview)
3. [Analysis Modes](#analysis-modes)
4. [Understanding Results](#understanding-results)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

---

## Quick Start

### 1. Launch Application
```bash
streamlit run src/app/app.py
```
Open browser to `http://localhost:8501`

### 2. Select Model
In the left sidebar:
- Choose analysis mode: **Trained Model**, **LLM Only**, or **Compare Both**
- If using LLM: Enter API key and select provider

### 3. Enter Text
In the main panel:
- Type or paste text to analyze
- Or click sample texts to try examples

### 4. Analyze
Click **🔍 Analyze** button

### 5. Review Results
See comprehensive analysis with:
- Risk classification
- Token importance highlighting
- Emotion detection
- LLM reasoning
- Ambiguity assessment

---

## Interface Overview

### Left Sidebar

#### 📊 Analysis Settings
- **Analysis Mode**: Choose between Trained Model, LLM, or Comparison
- **Model Selection**: Pick trained model (BERT, RoBERTa, DistilBERT)
- **LLM Provider**: Choose OpenAI, Groq, Google, or Local LLM

#### 🔑 API Configuration
- **API Keys**: Securely enter API keys (not stored permanently)
- **Model Selection**: Choose specific LLM model
- **Temperature**: Control randomness (0.0-1.0)
- **Top-p**: Control diversity (0.0-1.0)

#### 🎯 Prompt Techniques
- **Zero-Shot**: Direct classification
- **Few-Shot**: With examples
- **Chain-of-Thought**: Step-by-step reasoning

#### 🔄 Session Management
- **View Stats**: Analysis count, high-risk detections
- **Reset Session**: Clear all data and start fresh

#### ⚙️ Advanced Settings
- **Confidence Threshold**: Minimum prediction confidence
- **Show Probabilities**: Display class probabilities
- **Show Explanations**: Include AI reasoning
- **Auto-save**: Automatically save results

#### 🔬 Developer Mode
- **Enable Developer Tools**: Show model internals
  - Raw logits and probabilities
  - Attention matrices
  - Hidden states
  - Gradient analysis
  - Model architecture info

---

### Main Tabs

#### 1️⃣ Analyze Tab
**Single text analysis** - Most commonly used

**Features**:
- Text input area (up to 10,000 characters)
- Sample texts for quick testing
- Crisis detection banner
- Token-level highlighting
- Emotion & symptom dashboard
- Ambiguity assessment
- LLM reasoning
- Developer tools (if enabled)

#### 2️⃣ Batch Processing Tab
**Analyze multiple texts at once**

**How to Use**:
1. Prepare CSV file with columns:
   - `text`: Text to analyze
   - `label` (optional): True label for validation
2. Upload CSV file
3. Select processing mode
4. Click **Process Batch**
5. View aggregate statistics and download results

**Output Includes**:
- Risk distribution pie chart
- Confidence bar chart
- Word cloud of depressive vocabulary
- Downloadable results CSV

#### 3️⃣ Compare All Models Tab
**Side-by-side model comparison**

Compares:
- All trained models (BERT, RoBERTa, DistilBERT)
- LLM providers (if configured)

**Displays**:
- Annotated comparison table
- Prediction agreement/disagreement
- Confidence scores
- Explanation quality ratings

#### 4️⃣ Model Info Tab
**Model performance metrics and details**

**Shows**:
- Model personality cards (strengths/weaknesses)
- Performance metrics (accuracy, F1, precision, recall)
- Training time and parameters
- Ideal use cases
- Emoji indicators (🏆 BEST, ⚡ FASTEST, 🎯 ACCURATE)

#### 5️⃣ Training History Tab
**Training curves and learning progression**

**Visualizations**:
- Accuracy over epochs
- Loss curves
- F1 score progression
- Learning rate schedule
- Training/validation split performance

#### 6️⃣ Dataset Analytics Tab
**Dataset statistics and distribution**

**Analysis**:
- Class balance charts
- Emotional frequency distribution
- Text length distribution
- Word frequency analysis
- Sample diversity metrics

#### 7️⃣ Error Analysis Tab
**Misclassification investigation**

**Features**:
- View misclassified samples
- Predicted vs true label comparison
- Failure reason explanation
- Ambiguity scores
- Improvement suggestions

#### 8️⃣ Session History Tab
**Review past analyses**

**Capabilities**:
- Last 10-20 analyses stored locally
- Search by date/text/risk level
- Trend analysis over time
- Export history as CSV
- Privacy-preserving (no cloud storage)

---

## Analysis Modes

### 1. Trained Model Only
**Use Case**: Fast, offline analysis

**Advantages**:
- No API key required
- Fastest processing
- Consistent results
- Works offline

**Process**:
1. Text preprocessing
2. Crisis detection
3. Model classification
4. Token attribution (Integrated Gradients)
5. Emotion & symptom detection
6. Ambiguity assessment

**Output**:
- Binary classification (Control/Depression)
- Confidence score
- Risk level (Low/Moderate/High)
- Token importance highlighting
- Emotion dashboard

---

### 2. LLM Only
**Use Case**: Detailed reasoning and nuanced analysis

**Advantages**:
- Rich natural language explanations
- Context-aware analysis
- Multiple prompt techniques
- Latest LLM capabilities

**Requires**:
- API key (OpenAI, Groq, or Google)
- Internet connection

**Process**:
1. Text preprocessing
2. Crisis detection
3. LLM inference with selected prompt technique
4. Structured explanation parsing
5. Emotion & symptom extraction from LLM output

**Output**:
- Classification with confidence
- Professional assessment
- Key evidence summary
- Emotional profile
- Cognitive distortions identified
- Clinical context (DSM-5 references)

---

### 3. Compare Both
**Use Case**: Maximum confidence and cross-validation

**Advantages**:
- Dual verification
- Agreement analysis
- Best of both approaches
- Research-grade output

**Process**:
1. Run trained model analysis
2. Run LLM analysis
3. Compare predictions
4. Highlight agreements/disagreements
5. Combined confidence assessment

**Output**:
- Side-by-side comparison
- Agreement indicator
- Comparative table
- Recommended action based on agreement

---

## Understanding Results

### Risk Classification

#### 🟢 Low Risk (Control)
**Indicators**:
- Confidence > 60%
- Minimal depressive language
- Positive or neutral sentiment

**Interpretation**: Text shows minimal signs of depression-related distress.

**Example**: "Had a great day at work! Looking forward to the weekend."

---

#### 🟡 Moderate Risk
**Indicators**:
- Confidence 40-60%
- Mixed emotional content
- Some concerning phrases

**Interpretation**: Ambiguous. May require human review.

**Example**: "Work is stressful but I'm managing. Some days are harder than others."

---

#### 🔴 High Risk (Depression)
**Indicators**:
- Confidence > 60%
- Strong depressive language
- Multiple concerning phrases
- May trigger crisis detection

**Interpretation**: Text shows significant signs of depression-related distress.

**Example**: "I feel hopeless. Nothing brings me joy anymore and I can't see a point."

---

### Token Importance Colors

The Integrated Gradients method highlights words based on their contribution to the prediction:

#### 🔴 High Importance (Red)
**Score**: ≥ 0.75 (top 25%)

Words that strongly influenced the depression classification:
- "hopeless", "worthless", "hate", "never"
- Self-deprecating terms
- Absolute negative statements

#### 🟡 Medium Importance (Orange)
**Score**: 0.40-0.75 (middle 35%)

Words that moderately influenced the prediction:
- "tired", "difficult", "struggling"
- Negative emotion words
- Problem descriptions

#### 🟢 Low Importance (Green)
**Score**: < 0.40 (bottom 40%)

Words with minimal influence:
- Articles ("the", "a", "an")
- Conjunctions ("and", "but")
- Neutral descriptors

---

### Emotion Dashboard

#### Primary Emotions Detected
- **Sadness**: Melancholic expressions
- **Hopelessness**: Pessimistic outlook
- **Anger**: Frustration or rage
- **Anxiety**: Worry and fear
- **Numbness**: Emotional void

#### Cognitive Patterns
- **Catastrophizing**: Worst-case thinking
- **All-or-Nothing**: Black-and-white thinking
- **Personalization**: Excessive self-blame
- **Overgeneralization**: Broad conclusions from specific events

#### Clinical Symptoms
- **Insomnia**: Sleep disturbances
- **Anhedonia**: Loss of pleasure
- **Fatigue**: Low energy
- **Concentration Issues**: Difficulty focusing
- **Appetite Changes**: Eating disturbances

---

### Ambiguity & Uncertainty

#### Confidence Interpretation

**High Confidence (> 80%)**
- Model is very certain
- Clear linguistic patterns
- Low ambiguity

**Medium Confidence (50-80%)**
- Some uncertainty
- Mixed signals
- May need human review

**Low Confidence (< 50%)**
- High uncertainty
- Ambiguous language
- Strongly recommend human review

#### Uncertainty Reasons
- **Brevity**: Text too short (< 20 words)
- **Mixed Signals**: Contradictory emotions
- **Context Needed**: Insufficient information
- **Sarcasm/Irony**: Difficult to detect
- **Cultural Factors**: Unfamiliar expressions

---

### LLM Reasoning

LLM explanations include:

#### Summary
Brief overview of assessment (2-3 sentences)

#### Key Evidence
Specific phrases that influenced the decision:
- Direct quotes from text
- Linguistic patterns identified
- Emotional indicators

#### Emotional Profile
- Dominant emotions
- Intensity ratings
- Emotional progression through text

#### Cognitive Distortions
- Types identified (e.g., catastrophizing)
- Examples from text
- Severity assessment

#### Clinical Context
- DSM-5 criteria references (informational only)
- Symptom alignment
- Disclaimer: "Not a diagnosis"

#### Recommendations
- Suggested next steps
- Professional consultation advice
- Self-care suggestions (non-clinical)

---

## Advanced Features

### Developer Mode

Enable in **Advanced Settings** → **Developer Mode**

#### Raw Logits
**What it shows**: Unnormalized model outputs before softmax

**Use Case**:
- Understand model confidence
- Debug predictions
- Research analysis

**Interpretation**:
- Higher absolute values = more certain
- Logit difference > 2.0 = high confidence
- Close logits = uncertain prediction

---

#### Attention Matrices
**What it shows**: Which words the model focused on

**Use Case**:
- Understand model reasoning
- Identify important relationships
- Validate attribution methods

**Visualization**:
- Heatmap: Darker = more attention
- Token-to-token relationships
- Layer-wise attention patterns

---

#### Hidden States
**What it shows**: Internal layer activations

**Use Case**:
- Deep model analysis
- Feature engineering research
- Model debugging

**Metrics**:
- Mean: Average activation
- Std: Activation variance
- Norm: Activation magnitude

---

#### Gradient Analysis
**What it shows**: How model parameters contribute

**Use Case**:
- Identify vanishing/exploding gradients
- Optimize training
- Research gradient flow

**Warnings**:
- **Vanishing**: Gradients < 1e-7 (training issue)
- **Exploding**: Gradients > 1e3 (instability)

---

### Batch Processing Best Practices

#### CSV Format
```csv
text,label
"I feel great today!",0
"I hate everything.",1
"Work is busy but manageable.",0
```

**Requirements**:
- `text` column: Required
- `label` column: Optional (0=Control, 1=Depression)
- UTF-8 encoding
- Max 1000 rows recommended

#### Processing Tips
1. **Start Small**: Test with 10-20 samples
2. **Check Format**: Validate CSV structure
3. **Monitor Progress**: Watch processing bar
4. **Save Results**: Download CSV after processing
5. **Review Outliers**: Check low-confidence predictions

---

### Export Options

#### Text File
- Plain text summary
- Includes prediction, confidence, key phrases
- Easy to read and share

#### CSV File
- Structured data
- Batch results with all metrics
- Import into Excel/analysis tools

#### PDF Report (Phase 16 - Future)
- Professional format
- Charts and visualizations
- Full analysis report
- Suitable for presentations

---

## Best Practices

### Input Text Guidelines

#### ✅ Good Practices
- **Length**: 20-500 words ideal
- **Clarity**: Clear, coherent sentences
- **Context**: Provide sufficient background
- **Language**: English only (current version)
- **Authenticity**: Real or realistic text

#### ❌ Avoid
- Very short text (< 10 words)
- Non-English text
- Code or structured data
- Excessive special characters
- Completely nonsensical text

---

### Interpretation Guidelines

#### Do's
✅ Use results as **screening indicators**  
✅ Consider results **in context**  
✅ Cross-check with **multiple methods**  
✅ Review **uncertainty scores**  
✅ Consult **mental health professionals** for serious concerns

#### Don'ts
❌ **Never use as sole diagnostic tool**  
❌ **Don't ignore crisis warnings**  
❌ **Don't over-interpret low-confidence results**  
❌ **Don't substitute for professional help**  
❌ **Don't share sensitive data insecurely**

---

### Privacy & Ethics

#### Data Handling
- **No permanent storage**: Text not saved unless you export
- **Local processing**: Trained models run on your machine
- **API privacy**: LLM calls follow provider's privacy policy
- **Session data**: Cleared on browser close or manual reset

#### Ethical Use
- **Research purposes only**: Not for clinical decision-making
- **Informed consent**: Users should know they're being analyzed
- **Transparent limitations**: Be clear about system capabilities
- **Professional oversight**: Supervised use in sensitive contexts

---

## FAQ

### General Questions

**Q: Is this a medical device?**  
A: No. This is a research tool for educational and screening purposes only.

**Q: Can I use this to diagnose depression?**  
A: Absolutely not. Only qualified mental health professionals can diagnose depression.

**Q: Is my data secure?**  
A: Yes. Text is processed locally (trained models) or via secure API calls (LLMs). Nothing is permanently stored without your explicit export action.

**Q: Does this work offline?**  
A: Trained Model mode works completely offline. LLM mode requires internet connection.

---

### Technical Questions

**Q: Which model should I use?**  
A: 
- **BERT**: Balanced performance, good for general use
- **RoBERTa**: Best accuracy, slower
- **DistilBERT**: Fastest, slightly lower accuracy

**Q: What's the difference between prompt techniques?**  
A:
- **Zero-Shot**: Direct classification, fastest
- **Few-Shot**: Includes examples, more accurate
- **Chain-of-Thought**: Step-by-step reasoning, most detailed

**Q: Why do models disagree?**  
A: Models use different training data and architectures. Disagreement often indicates ambiguous text that needs human review.

**Q: What's Integrated Gradients?**  
A: A research-grade method for explaining which words influenced the prediction. More reliable than attention weights.

---

### Troubleshooting

**Q: "Model not found" error**  
A: Ensure models are in `models/trained/` directory. Check `DEPLOYMENT_GUIDE.md` for setup.

**Q: "API key invalid" error**  
A: Verify your API key is correct and active. Check provider's dashboard for status.

**Q: Results seem wrong**  
A: Check:
- Text length (20-500 words ideal)
- Language (English only)
- Confidence score (< 60% = uncertain)
- Consider using Compare mode for validation

**Q: App is slow**  
A: 
- Use DistilBERT for faster inference
- Close other applications
- Check system resources (RAM, CPU)
- Consider batch processing for multiple texts

**Q: Developer mode tabs empty**  
A: Enable specific features in Developer Mode settings (sidebar).

---

## Getting Help

### Documentation
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Token Attribution**: `docs/TOKEN_ATTRIBUTION_DOCUMENTATION.md`
- **API Reference**: Check function docstrings in code

### Support
- **GitHub Issues**: Report bugs and request features
- **Email**: [Contact information]

### Updates
Stay updated with latest improvements:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

---

## Quick Reference

### Keyboard Shortcuts
- **Ctrl/Cmd + Enter**: Submit text (in input box)
- **Tab**: Navigate between fields
- **Esc**: Close modals/expanders

### Color Codes
- 🔴 Red: High risk / High importance
- 🟡 Orange/Yellow: Moderate risk / Medium importance
- 🟢 Green: Low risk / Low importance
- 🔵 Blue: Informational
- ⚪ Gray: Neutral

### Confidence Thresholds
- **> 80%**: High confidence ✅
- **60-80%**: Moderate confidence ⚠️
- **< 60%**: Low confidence (review recommended) 👁️

---

**Last Updated**: November 26, 2025  
**Version**: 3.0 - Complete Edition  
**Status**: ✅ Production Ready

---

**Remember**: This tool assists research and screening. Always prioritize professional mental health care for clinical decisions. 🧠❤️
