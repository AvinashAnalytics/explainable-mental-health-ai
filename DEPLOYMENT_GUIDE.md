# 🚀 Deployment Guide - Explainable Mental Health AI Platform

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [HuggingFace Spaces Deployment](#huggingface-spaces-deployment)
5. [Production Considerations](#production-considerations)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 5GB free space
- **OS**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 20.04+)

### Required Software
- Git
- Python 3.8+
- pip or conda
- (Optional) Docker Desktop
- (Optional) CUDA 11.x for GPU support

---

## Local Deployment

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd "Major proj AWA"
```

### Step 2: Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Models
Ensure trained models are in `models/trained/` directory:
```
models/
  trained/
    bert-base-uncased/
      config.json
      pytorch_model.bin
      tokenizer.json
      ...
    roberta-base/
      config.json
      pytorch_model.bin
      ...
```

### Step 5: Run Application
```bash
streamlit run src/app/app.py
```

**Access**: Open browser to `http://localhost:8501`

---

## Docker Deployment

### Step 1: Create Dockerfile
Create `Dockerfile` in project root:

```dockerfile
# Use official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
ENTRYPOINT ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create .dockerignore
```
.venv/
__pycache__/
*.pyc
.git/
.gitignore
*.md
app.log
outputs/*.json
```

### Step 3: Build Docker Image
```bash
docker build -t mental-health-ai:latest .
```

### Step 4: Run Container
```bash
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  mental-health-ai:latest
```

### Step 5: Docker Compose (Optional)
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## HuggingFace Spaces Deployment

### Step 1: Create Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose **Streamlit** as SDK
4. Name: `mental-health-ai-explainable`

### Step 2: Prepare Files
Create `app.py` in root (entry point):
```python
import sys
sys.path.append('src/app')
from app import main

if __name__ == "__main__":
    main()
```

### Step 3: Create requirements.txt
Ensure all dependencies listed:
```
streamlit>=1.30.0
torch>=2.0.0
transformers>=4.35.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
```

### Step 4: Create README.md for Space
```markdown
---
title: Explainable Mental Health AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# Explainable Depression Detection AI

Research-grade mental health text analysis platform.
```

### Step 5: Push to Space
```bash
git remote add hf https://huggingface.co/spaces/USERNAME/mental-health-ai-explainable
git add .
git commit -m "Deploy to HuggingFace Spaces"
git push hf main
```

### Step 6: Configure Space Settings
- **Hardware**: CPU Basic (free) or GPU T4 (paid)
- **Visibility**: Private (recommended for medical data)
- **Persistent Storage**: Enable for model caching

---

## Production Considerations

### Performance Optimization

#### 1. Model Caching
Already implemented with `@st.cache_resource`:
```python
@st.cache_resource
def load_trained_model(model_name: str):
    # Model loading cached
    ...
```

#### 2. Enable GPU (if available)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

#### 3. Lazy Loading
Models load only when needed via session state.

#### 4. Data Caching
Training reports and metrics cached with TTL:
```python
@st.cache_data(ttl=300)  # 5 minutes
def load_training_report():
    ...
```

### Security Best Practices

#### 1. API Key Management
- **Never commit API keys to git**
- Use environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

#### 2. Input Validation
All user inputs validated:
```python
def validate_text_input(text: str) -> Tuple[bool, str]:
    # Length checks, content validation
    ...
```

#### 3. Rate Limiting
For production, add rate limiting:
```python
# Using streamlit-authenticator
import streamlit_authenticator as stauth

# Rate limit decorator
@st.cache_data(ttl=60)
def rate_limit_check(user_id):
    ...
```

### Scalability

#### 1. Load Balancing
Use Nginx for multiple instances:
```nginx
upstream streamlit_backend {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}
```

#### 2. Redis Session Storage
For multi-instance deployments:
```python
import redis
import pickle

r = redis.Redis(host='localhost', port=6379)

def save_session(session_id, data):
    r.setex(session_id, 3600, pickle.dumps(data))
```

### Monitoring

#### 1. Application Logs
Logs saved to `app.log`:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("User analyzed text")
```

#### 2. Health Checks
Add endpoint monitoring:
```bash
curl http://localhost:8501/_stcore/health
```

#### 3. Error Tracking
Integrate Sentry (optional):
```python
import sentry_sdk
sentry_sdk.init(dsn="YOUR_DSN")
```

---

## Environment Variables

### Required Variables
```bash
# LLM API Keys (at least one required for LLM features)
OPENAI_API_KEY=sk-proj-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...

# Local LLM (optional)
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
```

### Optional Variables
```bash
# Performance
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_ENABLE_CORS=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log

# Feature Flags
ENABLE_DEVELOPER_MODE=true
ENABLE_BATCH_PROCESSING=true
```

### Setting Environment Variables

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Docker:**
```bash
docker run -e OPENAI_API_KEY="sk-..." mental-health-ai:latest
```

**`.env` file (recommended):**
```
# Create .env in project root
OPENAI_API_KEY=sk-proj-...
GROQ_API_KEY=gsk_...

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

---

## Troubleshooting

### Common Issues

#### 1. "Model not found"
**Solution**: Ensure models exist in `models/trained/`
```bash
ls models/trained/
# Should show: bert-base-uncased, roberta-base, etc.
```

#### 2. "CUDA out of memory"
**Solutions**:
- Reduce batch size
- Use CPU instead: `device = "cpu"`
- Close other GPU applications

#### 3. "ModuleNotFoundError"
**Solution**: Reinstall dependencies
```bash
pip install --force-reinstall -r requirements.txt
```

#### 4. "Port 8501 already in use"
**Solution**: Kill existing process or use different port
```bash
# Kill process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run src/app/app.py --server.port 8502
```

#### 5. "API Rate Limit Exceeded"
**Solutions**:
- Wait for rate limit reset
- Use different API key
- Reduce request frequency

#### 6. "Slow loading times"
**Solutions**:
- Enable model caching (already implemented)
- Use SSD storage for models
- Increase RAM allocation

### Debug Mode

Enable detailed logging:
```bash
streamlit run src/app/app.py --logger.level=debug
```

Check logs:
```bash
tail -f app.log
```

### Performance Profiling

Use Streamlit profiler:
```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    # Your code
    result = analyze_text(text)

stats = pstats.Stats(pr)
stats.print_stats()
```

---

## Support & Resources

### Documentation
- **User Guide**: `USER_GUIDE.md`
- **API Documentation**: `docs/API_DOCUMENTATION.md`
- **Token Attribution**: `docs/TOKEN_ATTRIBUTION_DOCUMENTATION.md`

### Community
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share ideas

### Updates
Check for updates:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

---

## Checklist Before Deployment

- [ ] All dependencies installed
- [ ] Models downloaded and in correct directory
- [ ] API keys configured (if using LLM features)
- [ ] Environment variables set
- [ ] Security best practices followed
- [ ] Health checks working
- [ ] Logging configured
- [ ] Error handling tested
- [ ] Performance acceptable
- [ ] Documentation reviewed

---

**Last Updated**: November 26, 2025  
**Version**: 3.0 - Production Ready  
**Status**: ✅ Complete
