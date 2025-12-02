from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = Path('models/trained/distilbert_20251126_234714')
text = "I feel hopeless and tired all the time"

print('Loading model...')
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print('Tokenizing...')
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).cpu().item())
        conf = float(probs[0, pred].cpu().item())
    print('Prediction:', pred, 'Confidence:', conf)
except Exception as e:
    print('Error during quick predict test:', e)
