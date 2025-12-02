"""
LIME (Local Interpretable Model-agnostic Explanations) for mental health predictions.

Research Basis:
- Ribeiro et al. 2016: "Why Should I Trust You?" - LIME provides local explanations
- Used in MultiClass-Depression-Detection (thesis project) for visual word importance
- Generates HTML visualizations showing word-level attribution

Usage:
    explainer = LIMEExplainer(model=bert_model, tokenizer=bert_tokenizer)
    html = explainer.explain(text="I feel hopeless", class_names=["control", "depression"])
"""

import numpy as np
from typing import List, Callable, Dict, Any
import html as html_lib

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: lime package not installed. Run: pip install lime")


class LIMEExplainer:
    """
    LIME explainer for text-based mental health models.
    
    Provides word-level importance scores and HTML visualizations.
    """
    
    def __init__(self, model: Any, tokenizer: Any = None, class_names: List[str] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model: PyTorch/Sklearn model with predict_proba() method
            tokenizer: Optional tokenizer for preprocessing
            class_names: List of class names (e.g., ["control", "depression"])
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")
        
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names or ["control", "depression"]
        self.explainer = LimeTextExplainer(class_names=self.class_names)
    
    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper for model prediction compatible with LIME.
        
        Args:
            texts: List of perturbed text samples
        
        Returns:
            (n_samples, n_classes) probability array
        """
        import torch
        
        # Handle tokenization
        if self.tokenizer:
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
        else:
            # Assume model has internal tokenization
            inputs = texts
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(inputs)
            elif hasattr(self.model, 'forward'):
                # PyTorch model
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0]
                
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                raise ValueError("Model must have predict_proba() or forward() method")
        
        # Ensure 2D array (n_samples, n_classes)
        if len(probs.shape) == 1:
            # Binary classification with single probability
            probs = np.column_stack([1 - probs, probs])
        
        return probs
    
    def explain(self, text: str, num_features: int = 10, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single text sample.
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of perturbed samples for LIME
        
        Returns:
            Dictionary with:
                - html: HTML visualization
                - word_scores: List of (word, score) tuples
                - prediction: Model prediction
                - probabilities: Class probabilities
        """
        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text, 
            self._predict_fn, 
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction and probabilities
        probs = self._predict_fn([text])[0]
        predicted_class = int(np.argmax(probs))
        
        # Extract word-level scores for predicted class
        word_scores = exp.as_list(label=predicted_class)
        
        # Generate HTML visualization
        html = self._generate_html(text, word_scores, predicted_class, probs, num_features)
        
        return {
            'html': html,
            'word_scores': word_scores,
            'prediction': self.class_names[predicted_class],
            'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probs)}
        }
    
    def _generate_html(self, text: str, word_scores: List[tuple], predicted_class: int, probs: np.ndarray, num_features: int = 10) -> str:
        """
        Generate HTML visualization with color-coded word importance.
        
        Red: Positive contribution to depression class
        Green: Negative contribution (protective)
        Intensity: Proportional to importance magnitude
        """
        # Create word → score mapping
        score_map = {word.lower(): score for word, score in word_scores}
        
        # Split text into words
        words = text.split()
        
        # Generate HTML spans for each word
        html_words = []
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            score = score_map.get(clean_word, 0.0)
            
            # Color: red for positive (depression), green for negative (control)
            if score > 0:
                color = f"rgba(255, 0, 0, {min(abs(score), 1.0)})"
            else:
                color = f"rgba(0, 255, 0, {min(abs(score), 1.0)})"
            
            html_words.append(
                f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; '
                f'border-radius: 3px;" title="Score: {score:.3f}">{html_lib.escape(word)}</span>'
            )
        
        # Build full HTML
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background: #f9f9f9; border-radius: 8px;">
            <h3 style="color: #333;">LIME Explanation</h3>
            <div style="margin: 10px 0;">
                <strong>Prediction:</strong> {self.class_names[predicted_class]} 
                ({probs[predicted_class]:.2%} confidence)
            </div>
            <div style="margin: 10px 0;">
                <strong>Probabilities:</strong>
                {' | '.join([f'{name}: {prob:.2%}' for name, prob in zip(self.class_names, probs)])}
            </div>
            <div style="margin: 20px 0; line-height: 1.8;">
                <strong>Word-Level Attribution:</strong><br/>
                {' '.join(html_words)}
            </div>
            <div style="margin: 20px 0; font-size: 12px; color: #666;">
                <strong>Legend:</strong> 
                <span style="background-color: rgba(255, 0, 0, 0.5); padding: 2px 6px; border-radius: 3px;">
                    Red = Pro-Depression
                </span>
                <span style="background-color: rgba(0, 255, 0, 0.5); padding: 2px 6px; border-radius: 3px;">
                    Green = Pro-Control
                </span>
            </div>
            <div style="margin: 20px 0;">
                <strong>Top {num_features} Important Words:</strong>
                <ol>
                    {''.join([f'<li>{word}: {score:.3f}</li>' for word, score in word_scores[:10]])}
                </ol>
            </div>
        </div>
        """
        return html
    
    def get_token_scores(self, text: str, num_samples: int = 1000) -> Dict[str, float]:
        """
        Get raw token importance scores compatible with other explainability methods.
        
        Args:
            text: Input text
            num_samples: Number of LIME perturbation samples
        
        Returns:
            Dictionary mapping tokens to importance scores (normalized 0-1)
        """
        exp = self.explainer.explain_instance(
            text,
            self._predict_fn,
            num_features=50,  # Get all features
            num_samples=num_samples
        )
        
        # Get prediction
        probs = self._predict_fn([text])[0]
        predicted_class = int(np.argmax(probs))
        
        # Extract scores for predicted class
        word_scores = exp.as_list(label=predicted_class)
        
        # Normalize to 0-1 range
        scores_dict = {word.lower(): score for word, score in word_scores}
        
        if scores_dict:
            min_score = min(scores_dict.values())
            max_score = max(scores_dict.values())
            
            if max_score - min_score > 0:
                scores_dict = {
                    word: (score - min_score) / (max_score - min_score)
                    for word, score in scores_dict.items()
                }
        
        return scores_dict
    
    def explain_batch(self, texts: List[str], num_features: int = 10, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for multiple texts.
        
        Args:
            texts: List of texts to explain
            num_features: Number of top features per explanation
            num_samples: Number of perturbation samples
        
        Returns:
            List of explanation dictionaries
        """
        return [self.explain(text, num_features, num_samples) for text in texts]
    
    def save_html(self, html: str, filepath: str):
        """Save HTML explanation to file."""
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LIME Explanation</title>
</head>
<body>
    {html}
</body>
</html>
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_html)


def create_lime_explainer(model: Any, tokenizer: Any = None, class_names: List[str] = None):
    """
    Factory function to create LIME explainer with fallback.
    
    Args:
        model: Model to explain
        tokenizer: Optional tokenizer
        class_names: Class names
    
    Returns:
        LIMEExplainer instance or None if LIME unavailable
    """
    if not LIME_AVAILABLE:
        print("Warning: LIME not available. Install with: pip install lime")
        return None
    
    try:
        return LIMEExplainer(model, tokenizer, class_names)
    except Exception as e:
        print(f"Error: Failed to create LIME explainer: {e}")
        return None




def explain_with_lime(model, text: str, tokenizer=None, class_names=None, 
                      num_features=10, save_path=None) -> Dict[str, Any]:
    """
    Convenience function for one-off LIME explanations.
    
    Args:
        model: PyTorch/Sklearn model
        text: Input text
        tokenizer: Optional tokenizer
        class_names: List of class names
        num_features: Number of top features to show
        save_path: Optional path to save HTML
    
    Returns:
        Dictionary with html, word_scores, prediction, probabilities
    """
    explainer = LIMEExplainer(model, tokenizer, class_names)
    result = explainer.explain(text, num_features=num_features)
    
    if save_path:
        explainer.save_html(result['html'], save_path)
        print(f"LIME explanation saved to {save_path}")
    
    return result
