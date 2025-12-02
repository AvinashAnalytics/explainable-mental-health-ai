"""
Predict Depression with Trained Model + LLM Explanations

Usage:
    python predict_depression.py --model models/trained/roberta_20241125 --text "I feel hopeless"
    python predict_depression.py --model models/trained/roberta_20241125 --file test_samples.txt
    python predict_depression.py --model models/trained/roberta_20241125 --interactive

What this script does:
1. Loads fine-tuned BERT/RoBERTa model
2. Predicts depression from text
3. Extracts attention weights + IG attributions
4. Uses LLM (Groq/GPT-4) to generate human explanation
5. Shows complete explainable prediction
"""

import os
import sys
import argparse
import logging
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.llm_adapter import MentalHealthLLM

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict depression with explanations')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to analyze')
    parser.add_argument('--file', type=str, default=None,
                        help='File with texts (one per line)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    
    parser.add_argument('--llm-provider', type=str, default='groq',
                        choices=['groq', 'openai'],
                        help='LLM provider for explanations')
    parser.add_argument('--llm-model', type=str, default='llama-3.1-70b',
                        help='LLM model name')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM explanation generation')
    
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top rationale tokens to show')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()


class DepressionPredictor:
    """Depression prediction with explainability."""
    
    def __init__(self, model_path: str, use_cuda: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_attentions=True,
            output_hidden_states=True
        ).to(self.device)
        self.model.eval()
        
        logger.info("✅ Model loaded successfully\n")
    
    def predict(self, text: str, top_k: int = 5) -> Dict:
        """Predict with attention and IG explanations."""
        
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get prediction
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        
        # Extract attention weights
        attentions = outputs.attentions[-1]  # Last layer
        attention = attentions.mean(dim=1).squeeze().cpu().numpy()
        token_scores = attention.mean(axis=0)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Extract top attention tokens
        token_attention_pairs = [
            (token, score) for token, score in zip(tokens, token_scores)
            if token not in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']
        ]
        token_attention_pairs = sorted(
            token_attention_pairs, key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # Compute Integrated Gradients
        def forward_func(input_ids):
            return self.model(input_ids=input_ids).logits[:, pred_label]
        
        lig = LayerIntegratedGradients(forward_func, self.model.get_input_embeddings())
        
        input_ids = inputs['input_ids']
        baseline_ids = torch.zeros_like(input_ids)
        
        attributions, _ = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            n_steps=50,
            return_convergence_delta=True
        )
        
        attr_scores = attributions.sum(dim=-1).squeeze().cpu().numpy()
        attr_scores = (attr_scores - attr_scores.min()) / (attr_scores.max() - attr_scores.min() + 1e-10)
        
        # Top IG tokens
        token_ig_pairs = [
            (token, score) for token, score in zip(tokens, attr_scores)
            if token not in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']
        ]
        token_ig_pairs = sorted(token_ig_pairs, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Combine scores
        rationale_scores = {}
        for (token_att, score_att), (token_ig, score_ig) in zip(
            token_attention_pairs, token_ig_pairs
        ):
            combined = 0.5 * score_att + 0.5 * score_ig
            token_clean = token_att.replace('Ġ', '').replace('▁', '').strip()
            if token_clean and len(token_clean) > 1:
                rationale_scores[token_clean] = combined
        
        top_rationales = sorted(
            rationale_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        return {
            'text': text,
            'prediction': 'Depression' if pred_label == 1 else 'No Depression',
            'label': pred_label,
            'confidence': confidence,
            'probabilities': {
                'control': float(probs[0]),
                'depression': float(probs[1])
            },
            'attention_tokens': [(t.replace('Ġ', '').replace('▁', ''), s) 
                                 for t, s in token_attention_pairs],
            'ig_tokens': [(t.replace('Ġ', '').replace('▁', ''), s) 
                          for t, s in token_ig_pairs],
            'rationale_tokens': [t for t, _ in top_rationales]
        }
    
    def generate_llm_explanation(
        self, result: Dict, llm_provider: str, llm_model: str
    ) -> str:
        """Generate LLM explanation."""
        
        try:
            llm = MentalHealthLLM(provider=llm_provider, model=llm_model)
            
            prompt = f"""You are a mental health AI assistant. A depression detection model analyzed the following text.

Text: "{result['text']}"

Model Prediction: {result['prediction']}
Confidence: {result['confidence']:.2%}

Key Words (from model attention): {', '.join(result['rationale_tokens'][:5])}

Task: Provide a clear, empathetic explanation (3-4 sentences) of why the model made this prediction. Use Chain-of-Thought reasoning to:
1. Identify emotional cues in the text
2. Relate them to depression symptoms
3. Explain which words were most indicative

Keep it concise and human-friendly."""
            
            explanation = llm.generate(prompt, temperature=0.2, max_tokens=200)
            return explanation
            
        except Exception as e:
            return f"LLM explanation unavailable: {str(e)}"


def display_result(result: Dict, llm_explanation: str = None):
    """Display prediction results."""
    
    print("\n" + "=" * 70)
    print("📝 DEPRESSION DETECTION RESULT")
    print("=" * 70)
    
    print(f"\nInput Text:")
    print(f"  {result['text']}")
    
    print(f"\n🔮 Model Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Probabilities:")
    print(f"     Control:    {result['probabilities']['control']:.2%}")
    print(f"     Depression: {result['probabilities']['depression']:.2%}")
    
    print(f"\n🎯 Top Rationale Tokens (Attention + IG):")
    for i, (token, score) in enumerate(zip(
        result['rationale_tokens'], 
        [s for _, s in result['attention_tokens'][:5]]
    ), 1):
        print(f"   {i}. '{token}' (score: {score:.3f})")
    
    if llm_explanation:
        print(f"\n🤖 LLM Explanation:")
        print(f"   {llm_explanation}")
    
    print("\n" + "=" * 70)


def main():
    args = parse_args()
    
    # Load model
    predictor = DepressionPredictor(args.model, use_cuda=not args.no_cuda)
    
    # Setup LLM
    use_llm = not args.no_llm and (
        os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
    )
    
    if use_llm:
        logger.info(f"LLM enabled: {args.llm_provider}/{args.llm_model}\n")
    else:
        logger.info("LLM disabled (set GROQ_API_KEY or OPENAI_API_KEY to enable)\n")
    
    # Process texts
    if args.interactive:
        logger.info("Interactive mode (type 'quit' to exit)")
        while True:
            text = input("\nEnter text to analyze: ").strip()
            if text.lower() == 'quit':
                break
            if not text:
                continue
            
            result = predictor.predict(text, top_k=args.top_k)
            
            llm_explanation = None
            if use_llm:
                llm_explanation = predictor.generate_llm_explanation(
                    result, args.llm_provider, args.llm_model
                )
            
            display_result(result, llm_explanation)
    
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(texts)} texts from {args.file}\n")
        
        for i, text in enumerate(texts, 1):
            logger.info(f"\n[{i}/{len(texts)}]")
            result = predictor.predict(text, top_k=args.top_k)
            
            llm_explanation = None
            if use_llm:
                llm_explanation = predictor.generate_llm_explanation(
                    result, args.llm_provider, args.llm_model
                )
            
            display_result(result, llm_explanation)
    
    elif args.text:
        result = predictor.predict(args.text, top_k=args.top_k)
        
        llm_explanation = None
        if use_llm:
            llm_explanation = predictor.generate_llm_explanation(
                result, args.llm_provider, args.llm_model
            )
        
        display_result(result, llm_explanation)
    
    else:
        logger.error("Provide --text, --file, or --interactive")
        sys.exit(1)


if __name__ == '__main__':
    main()
