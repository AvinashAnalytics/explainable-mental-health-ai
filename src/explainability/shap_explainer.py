"""
SHAP (SHapley Additive exPlanations) for Depression Detection

Research Basis:
- Lundberg & Lee 2017: "A Unified Approach to Interpreting Model Predictions"
- Game-theoretic approach to feature attribution
- Used in mental health AI for model-agnostic explanations

Properties:
- Local accuracy: Explanation model matches original model locally
- Missingness: Missing features have zero attribution
- Consistency: Monotonic relationship between feature value and attribution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


class SHAPExplainer:
    """
    SHAP explainer for text-based depression detection models.
    
    Supports:
    - Transformer models (using Partition explainer)
    - Any model (using Kernel explainer - slower)
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        explainer_type: str = 'partition',
        max_evals: int = 500,
        class_names: List[str] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Model to explain
            tokenizer: Tokenizer for preprocessing
            explainer_type: 'partition' (fast, for transformers) or 'kernel' (slow, universal)
            max_evals: Maximum model evaluations for kernel explainer
            class_names: Class labels
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        self.model = model
        self.tokenizer = tokenizer
        self.explainer_type = explainer_type
        self.max_evals = max_evals
        self.class_names = class_names or ['control', 'depression']
        
        # Will be initialized on first call
        self.explainer = None
    
    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        """Prediction function compatible with SHAP."""
        import torch
        
        self.model.eval()
        
        # Tokenize
        if self.tokenizer:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
        else:
            inputs = texts
        
        # Get predictions
        with torch.no_grad():
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(inputs)
            else:
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        return probs
    
    def explain(
        self,
        text: str,
        num_samples: int = 100,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for text.
        
        Args:
            text: Input text to explain
            num_samples: Number of background samples for kernel explainer
            target_class: Class to explain (None = predicted class)
        
        Returns:
            Dictionary with:
                - shap_values: SHAP values for each token
                - tokens: List of tokens
                - prediction: Predicted class
                - base_value: Expected model output
                - token_attributions: List of (token, shap_value) pairs
        """
        # Initialize explainer if needed
        if self.explainer is None:
            if self.explainer_type == 'partition':
                # Partition explainer for transformers (hierarchical)
                self.explainer = shap.Explainer(
                    self._predict_fn,
                    self.tokenizer
                )
            else:
                # Kernel explainer (model-agnostic, slower)
                background_data = [""] * num_samples  # Empty strings as background
                self.explainer = shap.KernelExplainer(
                    self._predict_fn,
                    background_data,
                    link='identity'
                )
        
        # Get prediction
        probs = self._predict_fn([text])[0]
        predicted_class = int(np.argmax(probs))
        
        if target_class is None:
            target_class = predicted_class
        
        # Compute SHAP values
        try:
            shap_values = self.explainer([text])
            
            # Extract values for target class
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0, :, target_class]
                base_value = shap_values.base_values[0, target_class] if hasattr(shap_values, 'base_values') else 0.0
                tokens = shap_values.data[0] if hasattr(shap_values, 'data') else text.split()
            else:
                values = shap_values[0][:, target_class] if len(shap_values[0].shape) > 1 else shap_values[0]
                base_value = 0.0
                tokens = text.split()
        
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}. Falling back to simple tokenization.")
            # Fallback: use word-level simple approximation
            tokens = text.split()
            values = np.zeros(len(tokens))
            base_value = probs[target_class]
        
        # Create token-attribution pairs
        token_attributions = [(token, float(value)) for token, value in zip(tokens, values)]
        
        return {
            'shap_values': values.tolist() if isinstance(values, np.ndarray) else values,
            'tokens': tokens,
            'prediction': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'base_value': float(base_value),
            'token_attributions': token_attributions,
            'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probs)}
        }
    
    def get_top_tokens(
        self,
        text: str,
        top_k: int = 10,
        target_class: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k tokens by absolute SHAP value.
        
        Args:
            text: Input text
            top_k: Number of top tokens
            target_class: Class to explain
        
        Returns:
            List of (token, shap_value) tuples
        """
        result = self.explain(text, target_class=target_class)
        token_attrs = result['token_attributions']
        
        # Sort by absolute SHAP value
        sorted_attrs = sorted(token_attrs, key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_attrs[:top_k]
    
    def visualize(
        self,
        text: str,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> str:
        """
        Generate HTML visualization of SHAP attributions.
        
        Args:
            text: Input text
            target_class: Class to explain
            normalize: Normalize SHAP values to [0, 1]
        
        Returns:
            HTML string with color-coded tokens
        """
        result = self.explain(text, target_class=target_class)
        tokens = result['tokens']
        shap_values = np.array(result['shap_values'])
        
        # Normalize if requested
        if normalize and len(shap_values) > 0:
            val_min, val_max = shap_values.min(), shap_values.max()
            if val_max - val_min > 0:
                shap_values = (shap_values - val_min) / (val_max - val_min)
        
        # Generate HTML
        html_parts = [
            '<div style="font-family: Arial, sans-serif; padding: 15px; background: #f9f9f9; border-radius: 8px;">',
            f'<h3 style="margin-top: 0;">SHAP Explanation</h3>',
            f'<p><strong>Prediction:</strong> {result["predicted_class_name"]} '
            f'({result["probabilities"][result["predicted_class_name"]]:.2%})</p>',
            f'<p><strong>Base Value:</strong> {result["base_value"]:.4f}</p>',
            '<div style="line-height: 2; margin: 20px 0;">'
        ]
        
        for token, value in zip(tokens, shap_values):
            # Color: red for positive SHAP (pro-depression), blue for negative
            if value > 0:
                color = f"rgba(255, 0, 0, {min(abs(value), 1.0)})"
            else:
                color = f"rgba(0, 0, 255, {min(abs(value), 1.0)})"
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 6px; margin: 2px; '
                f'border-radius: 3px; display: inline-block;" title="SHAP: {value:.4f}">{token}</span>'
            )
        
        html_parts.append('</div>')
        html_parts.append(
            '<p style="font-size: 12px; color: #666;">'
            '<strong>Legend:</strong> '
            '<span style="background-color: rgba(255,0,0,0.5); padding: 2px 6px; border-radius: 3px;">Red = Increases Depression Probability</span> '
            '<span style="background-color: rgba(0,0,255,0.5); padding: 2px 6px; border-radius: 3px;">Blue = Decreases Depression Probability</span>'
            '</p>'
        )
        html_parts.append('<p style="font-size: 11px; color: #999;">Research: Lundberg & Lee 2017 - SHAP (Game Theory)</p>')
        html_parts.append('</div>')
        
        return ''.join(html_parts)
    
    def explain_batch(
        self,
        texts: List[str],
        target_class: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple texts in batch.
        
        Args:
            texts: List of texts
            target_class: Class to explain
        
        Returns:
            List of explanation dictionaries
        """
        return [self.explain(text, target_class=target_class) for text in texts]
    
    def get_feature_importance_summary(
        self,
        texts: List[str],
        target_class: int = 1
    ) -> Dict[str, float]:
        """
        Get aggregated feature importance across multiple texts.
        
        Args:
            texts: List of texts
            target_class: Class to explain
        
        Returns:
            Dictionary mapping tokens to average absolute SHAP values
        """
        token_importance = {}
        token_counts = {}
        
        for text in texts:
            result = self.explain(text, target_class=target_class)
            
            for token, shap_val in result['token_attributions']:
                token = token.lower()
                if token not in token_importance:
                    token_importance[token] = 0.0
                    token_counts[token] = 0
                
                token_importance[token] += abs(shap_val)
                token_counts[token] += 1
        
        # Average
        for token in token_importance:
            token_importance[token] /= token_counts[token]
        
        # Sort by importance
        sorted_importance = dict(sorted(token_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance


def create_shap_explainer(
    model: Any,
    tokenizer: Any,
    explainer_type: str = 'partition',
    class_names: List[str] = None
) -> Optional[SHAPExplainer]:
    """
    Factory function to create SHAP explainer with fallback.
    
    Args:
        model: Model to explain
        tokenizer: Tokenizer
        explainer_type: 'partition' or 'kernel'
        class_names: Class labels
    
    Returns:
        SHAPExplainer or None if unavailable
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Install with: pip install shap")
        return None
    
    try:
        return SHAPExplainer(model, tokenizer, explainer_type, class_names=class_names)
    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        return None
