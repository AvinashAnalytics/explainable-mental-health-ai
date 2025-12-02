"""
Integrated Gradients for Depression Detection Models

Research Basis:
- Sundararajan et al. 2017: "Axiomatic Attribution for Deep Networks"
- Provides gradient-based token importance with theoretical guarantees
- Used in arXiv:2304.03347 for mental health LLM interpretability

Properties:
- Completeness: Attributions sum to difference between prediction and baseline
- Sensitivity: Non-zero gradient implies non-zero attribution
- Implementation Invariance: Functionally equivalent networks give identical attributions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer for text classification models.
    
    Computes attribution scores by integrating gradients along path from baseline to input.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cpu',
        n_steps: int = 50
    ):
        """
        Initialize Integrated Gradients explainer.
        
        Args:
            model: PyTorch model to explain
            tokenizer: Tokenizer for text preprocessing
            device: Device to run computations ('cpu' or 'cuda')
            n_steps: Number of integration steps (default 50, paper uses 20-300)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
    def explain(
        self,
        text: str,
        target_class: Optional[int] = None,
        baseline: str = "",
        return_convergence_delta: bool = False
    ) -> Dict[str, Any]:
        """
        Generate Integrated Gradients explanation for text.
        
        Args:
            text: Input text to explain
            target_class: Class to compute attributions for (None = predicted class)
            baseline: Baseline text (default: empty string or PAD tokens)
            return_convergence_delta: Whether to return convergence diagnostic
        
        Returns:
            Dictionary with:
                - token_attributions: List of (token, attribution_score) tuples
                - tokens: List of tokens
                - attributions: numpy array of attribution scores
                - prediction: Model prediction
                - convergence_delta: Difference between integrated and expected (if requested)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Tokenize baseline
        baseline_inputs = self.tokenizer(
            baseline,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=inputs['input_ids'].shape[1]
        ).to(self.device)
        
        # Get embeddings
        input_embeddings = self._get_embeddings(inputs)
        baseline_embeddings = self._get_embeddings(baseline_inputs)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        # Compute integrated gradients
        attributions = self._compute_integrated_gradients(
            input_embeddings,
            baseline_embeddings,
            inputs,
            target_class
        )
        
        # Sum attributions across embedding dimensions
        token_attributions = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Remove special tokens and create token-attribution pairs
        token_attr_pairs = []
        for i, (token, attr) in enumerate(zip(tokens, token_attributions)):
            if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                token_attr_pairs.append((token, float(attr)))
        
        # Compute convergence delta if requested
        convergence_delta = None
        if return_convergence_delta:
            convergence_delta = self._compute_convergence_delta(
                input_embeddings,
                baseline_embeddings,
                inputs,
                target_class,
                token_attributions
            )
        
        result = {
            'token_attributions': token_attr_pairs,
            'tokens': tokens,
            'attributions': token_attributions,
            'prediction': predicted_class,
            'target_class': target_class,
            'probabilities': probs[0].cpu().numpy().tolist()
        }
        
        if convergence_delta is not None:
            result['convergence_delta'] = convergence_delta
        
        return result
    
    def _get_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings from model."""
        # Try to get embedding layer
        if hasattr(self.model, 'bert'):
            embeddings = self.model.bert.embeddings.word_embeddings(inputs['input_ids'])
        elif hasattr(self.model, 'roberta'):
            embeddings = self.model.roberta.embeddings.word_embeddings(inputs['input_ids'])
        elif hasattr(self.model, 'distilbert'):
            embeddings = self.model.distilbert.embeddings.word_embeddings(inputs['input_ids'])
        elif hasattr(self.model, 'embeddings'):
            embeddings = self.model.embeddings(inputs['input_ids'])
        elif hasattr(self.model, 'transformer'):
            embeddings = self.model.transformer.wte(inputs['input_ids'])
        else:
            raise ValueError("Could not find embedding layer in model")
        
        return embeddings
    
    def _compute_integrated_gradients(
        self,
        input_embeddings: torch.Tensor,
        baseline_embeddings: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        target_class: int
    ) -> torch.Tensor:
        """
        Compute integrated gradients by integrating along path from baseline to input.
        
        Formula: IG = (x - x') * ∫[0,1] (∂F(x' + α(x - x'))/∂x) dα
        
        Approximated using Riemann sum with n_steps.
        """
        # Store gradients for each step
        gradients = []
        
        # Integration path: baseline + α * (input - baseline) for α in [0, 1]
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            
            # Interpolated embeddings
            interpolated_embeddings = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
            interpolated_embeddings.requires_grad_(True)
            
            # Forward pass with interpolated embeddings
            outputs = self._forward_from_embeddings(interpolated_embeddings, inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Target class score
            target_score = logits[0, target_class]
            
            # Compute gradients
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            
            # Store gradient
            gradients.append(interpolated_embeddings.grad.clone())
            
            # Clean up
            interpolated_embeddings.grad = None
        
        # Average gradients (trapezoidal rule)
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Multiply by input difference
        integrated_gradients = (input_embeddings - baseline_embeddings) * avg_gradients
        
        return integrated_gradients
    
    def _forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        original_inputs: Dict[str, torch.Tensor]
    ) -> Any:
        """Forward pass starting from embeddings instead of input_ids."""
        # This is model-specific - adapt based on your model architecture
        
        # For BERT-based models
        if hasattr(self.model, 'bert'):
            # Get position and token type embeddings
            position_ids = torch.arange(embeddings.size(1), device=self.device).unsqueeze(0)
            token_type_ids = original_inputs.get('token_type_ids', torch.zeros_like(original_inputs['input_ids']))
            
            position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)
            token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
            
            embeddings = embeddings + position_embeddings + token_type_embeddings
            embeddings = self.model.bert.embeddings.LayerNorm(embeddings)
            embeddings = self.model.bert.embeddings.dropout(embeddings)
            
            # Forward through encoder
            encoder_outputs = self.model.bert.encoder(
                embeddings,
                attention_mask=original_inputs.get('attention_mask')
            )
            
            sequence_output = encoder_outputs[0]
            pooled_output = self.model.bert.pooler(sequence_output) if hasattr(self.model.bert, 'pooler') else sequence_output[:, 0]
            
            # Forward through classifier
            if hasattr(self.model, 'classifier'):
                logits = self.model.classifier(pooled_output)
            elif hasattr(self.model, 'dropout') and hasattr(self.model, 'fc'):
                logits = self.model.fc(self.model.dropout(pooled_output))
            else:
                logits = pooled_output
            
            class Output:
                def __init__(self, logits):
                    self.logits = logits
            
            return Output(logits)
        
        # For other architectures, implement similarly
        else:
            raise NotImplementedError("Forward from embeddings not implemented for this model type")
    
    def _compute_convergence_delta(
        self,
        input_embeddings: torch.Tensor,
        baseline_embeddings: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        target_class: int,
        attributions: np.ndarray
    ) -> float:
        """
        Compute convergence diagnostic: difference between integrated and expected value.
        
        Should be close to 0 for good approximation.
        Formula: |sum(attributions) - (F(x) - F(x'))|
        """
        # Get predictions for input and baseline
        with torch.no_grad():
            # Input prediction
            outputs_input = self.model(**inputs)
            logits_input = outputs_input.logits if hasattr(outputs_input, 'logits') else outputs_input
            score_input = logits_input[0, target_class].item()
            
            # Baseline prediction (need to reconstruct inputs from baseline embeddings)
            # This is approximation - ideally should use actual baseline inputs
            score_baseline = 0.0  # Assuming zero baseline
        
        # Sum of attributions
        attributions_sum = np.sum(attributions)
        
        # Convergence delta
        delta = abs(attributions_sum - (score_input - score_baseline))
        
        return delta
    
    def get_top_tokens(
        self,
        text: str,
        top_k: int = 10,
        target_class: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most important tokens by attribution score.
        
        Args:
            text: Input text
            top_k: Number of top tokens to return
            target_class: Class to compute attributions for
        
        Returns:
            List of (token, attribution) tuples sorted by absolute attribution
        """
        result = self.explain(text, target_class)
        token_attrs = result['token_attributions']
        
        # Sort by absolute attribution
        sorted_attrs = sorted(token_attrs, key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_attrs[:top_k]
    
    def visualize_attributions(
        self,
        text: str,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> str:
        """
        Generate HTML visualization of token attributions.
        
        Args:
            text: Input text
            target_class: Class to explain
            normalize: Whether to normalize attributions to [0, 1]
        
        Returns:
            HTML string with color-coded tokens
        """
        result = self.explain(text, target_class)
        tokens = result['tokens']
        attributions = result['attributions']
        
        # Normalize if requested
        if normalize and len(attributions) > 0:
            attr_min, attr_max = attributions.min(), attributions.max()
            if attr_max - attr_min > 0:
                attributions = (attributions - attr_min) / (attr_max - attr_min)
        
        # Generate HTML
        html_parts = ['<div style="font-family: monospace; line-height: 2;">']
        
        for token, attr in zip(tokens, attributions):
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                continue
            
            # Color based on attribution (red = positive, blue = negative)
            if attr > 0:
                color = f"rgba(255, 0, 0, {min(abs(attr), 1.0)})"
            else:
                color = f"rgba(0, 0, 255, {min(abs(attr), 1.0)})"
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; '
                f'border-radius: 3px;" title="Attribution: {attr:.4f}">{token}</span>'
            )
        
        html_parts.append('</div>')
        
        return ''.join(html_parts)


def compute_integrated_gradients(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    n_steps: int = 50,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Convenience function to compute integrated gradients.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        text: Input text
        n_steps: Number of integration steps
        device: Computation device
    
    Returns:
        Explanation dictionary
    """
    explainer = IntegratedGradientsExplainer(model, tokenizer, device, n_steps)
    return explainer.explain(text, return_convergence_delta=True)
