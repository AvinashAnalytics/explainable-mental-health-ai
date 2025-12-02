"""
Comprehensive Faithfulness Metrics for Explainability Evaluation

Research Basis:
- DeYoung et al. 2020: "ERASER: A Benchmark to Evaluate Rationalized NLP Models"
- Jacovi & Goldberg 2020: "Towards Faithfully Interpretable NLP Systems"
- arXiv:2304.03347: Mental Health LLM Interpretability

Faithfulness Metrics (Complete Implementation):
1. Comprehensiveness: ΔP when removing rationale tokens
2. Sufficiency: ΔP when keeping only rationale tokens
3. Decision Flip Rate: % predictions that change when rationale removed
4. Monotonicity: Correlation between token addition and confidence increase
5. AOPC (Area Over Perturbation Curve): Average confidence drop curve
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class FaithfulnessMetrics:
    """
    Complete faithfulness metrics for explanation evaluation.
    
    All metrics measure how well explanations reflect actual model behavior.
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = 'cpu'):
        """
        Initialize faithfulness evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            device: Computation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def comprehensiveness(
        self,
        text: str,
        rationale_indices: List[int],
        predicted_class: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Comprehensiveness: How much does prediction change when rationale removed?
        
        Formula: comp = P_original(y) - P_masked(y)
        
        Higher = better (rationale is necessary for prediction)
        
        Args:
            text: Original text
            rationale_indices: Indices of important tokens
            predicted_class: Class to evaluate (None = predicted class)
        
        Returns:
            Dictionary with comprehensiveness score and details
        """
        # Get original prediction
        original_probs = self._predict(text)
        
        if predicted_class is None:
            predicted_class = int(np.argmax(original_probs))
        
        original_conf = original_probs[predicted_class]
        
        # Mask rationale tokens
        tokens = text.split()
        masked_tokens = [
            '[MASK]' if i in rationale_indices else token
            for i, token in enumerate(tokens)
        ]
        masked_text = ' '.join(masked_tokens)
        
        # Get prediction without rationale
        masked_probs = self._predict(masked_text)
        masked_conf = masked_probs[predicted_class]
        
        # Comprehensiveness score
        comp_score = original_conf - masked_conf
        
        return {
            'comprehensiveness': float(comp_score),
            'original_confidence': float(original_conf),
            'masked_confidence': float(masked_conf),
            'confidence_drop': float(comp_score),
            'predicted_class': predicted_class
        }
    
    def sufficiency(
        self,
        text: str,
        rationale_indices: List[int],
        predicted_class: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Sufficiency: Is rationale alone sufficient to maintain prediction?
        
        Formula: suff = P_original(y) - P_rationale_only(y)
        
        Lower = better (rationale is sufficient)
        
        Args:
            text: Original text
            rationale_indices: Indices of important tokens
            predicted_class: Class to evaluate
        
        Returns:
            Dictionary with sufficiency score and details
        """
        # Get original prediction
        original_probs = self._predict(text)
        
        if predicted_class is None:
            predicted_class = int(np.argmax(original_probs))
        
        original_conf = original_probs[predicted_class]
        
        # Keep only rationale tokens
        tokens = text.split()
        rationale_tokens = [tokens[i] for i in rationale_indices if i < len(tokens)]
        rationale_text = ' '.join(rationale_tokens)
        
        # Handle empty rationale
        if not rationale_text.strip():
            return {
                'sufficiency': 1.0,  # Worst case: no rationale provided
                'original_confidence': float(original_conf),
                'rationale_confidence': 0.0,
                'confidence_drop': float(original_conf),
                'predicted_class': predicted_class
            }
        
        # Get prediction with only rationale
        rationale_probs = self._predict(rationale_text)
        rationale_conf = rationale_probs[predicted_class]
        
        # Sufficiency score
        suff_score = original_conf - rationale_conf
        
        return {
            'sufficiency': float(suff_score),
            'original_confidence': float(original_conf),
            'rationale_confidence': float(rationale_conf),
            'confidence_drop': float(suff_score),
            'predicted_class': predicted_class
        }
    
    def decision_flip_rate(
        self,
        texts: List[str],
        rationale_indices_list: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Decision Flip Rate: % of predictions that change when rationale removed.
        
        Measures: Does removing explanation tokens actually change the decision?
        
        Args:
            texts: List of input texts
            rationale_indices_list: List of rationale indices for each text
        
        Returns:
            Dictionary with flip rate and per-sample results
        """
        flips = []
        results = []
        
        for text, rationale_indices in zip(texts, rationale_indices_list):
            # Original prediction
            original_probs = self._predict(text)
            original_class = int(np.argmax(original_probs))
            
            # Mask rationale
            tokens = text.split()
            masked_tokens = [
                '[MASK]' if i in rationale_indices else token
                for i, token in enumerate(tokens)
            ]
            masked_text = ' '.join(masked_tokens)
            
            # Masked prediction
            masked_probs = self._predict(masked_text)
            masked_class = int(np.argmax(masked_probs))
            
            # Check if decision flipped
            flipped = (original_class != masked_class)
            flips.append(flipped)
            
            results.append({
                'text': text,
                'original_class': original_class,
                'masked_class': masked_class,
                'flipped': flipped,
                'confidence_change': float(original_probs[original_class] - masked_probs[original_class])
            })
        
        flip_rate = np.mean(flips)
        
        return {
            'decision_flip_rate': float(flip_rate),
            'num_flips': int(np.sum(flips)),
            'total_samples': len(texts),
            'per_sample_results': results
        }
    
    def monotonicity(
        self,
        text: str,
        token_importance_scores: List[float],
        num_steps: int = 10
    ) -> Dict[str, float]:
        """
        Monotonicity: Does adding tokens by importance increase confidence monotonically?
        
        Measures: Correlation between % tokens added and prediction confidence.
        
        Args:
            text: Input text
            token_importance_scores: Importance score for each token
            num_steps: Number of steps to test
        
        Returns:
            Dictionary with monotonicity score (Spearman correlation)
        """
        # Get original prediction
        original_probs = self._predict(text)
        predicted_class = int(np.argmax(original_probs))
        
        # Sort tokens by importance (descending)
        tokens = text.split()
        token_importance_pairs = list(zip(tokens, token_importance_scores))
        sorted_pairs = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)
        
        # Gradually add tokens and measure confidence
        confidences = []
        token_fractions = []
        
        for step in range(1, min(num_steps + 1, len(sorted_pairs) + 1)):
            # Take top step tokens
            num_tokens = max(1, int(len(sorted_pairs) * step / num_steps))
            selected_tokens = [pair[0] for pair in sorted_pairs[:num_tokens]]
            partial_text = ' '.join(selected_tokens)
            
            # Get confidence
            probs = self._predict(partial_text)
            confidence = probs[predicted_class]
            
            confidences.append(confidence)
            token_fractions.append(num_tokens / len(tokens))
        
        # Compute Spearman correlation
        if len(confidences) > 1:
            correlation, p_value = spearmanr(token_fractions, confidences)
        else:
            correlation, p_value = 0.0, 1.0
        
        return {
            'monotonicity': float(correlation),
            'p_value': float(p_value),
            'confidences': [float(c) for c in confidences],
            'token_fractions': token_fractions,
            'is_monotonic': correlation > 0.5 and p_value < 0.05
        }
    
    def aopc(
        self,
        text: str,
        token_importance_scores: List[float],
        num_steps: int = 10
    ) -> Dict[str, float]:
        """
        AOPC (Area Over Perturbation Curve): Average confidence drop as tokens removed.
        
        Measures: How quickly does confidence drop when removing important tokens?
        
        Args:
            text: Input text
            token_importance_scores: Importance score for each token
            num_steps: Number of perturbation steps
        
        Returns:
            Dictionary with AOPC score
        """
        # Get original prediction
        original_probs = self._predict(text)
        predicted_class = int(np.argmax(original_probs))
        original_conf = original_probs[predicted_class]
        
        # Sort tokens by importance (descending)
        tokens = text.split()
        token_importance_pairs = list(zip(range(len(tokens)), token_importance_scores))
        sorted_pairs = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)
        
        # Gradually remove top tokens and measure confidence drop
        confidence_drops = []
        
        for step in range(1, min(num_steps + 1, len(sorted_pairs) + 1)):
            # Remove top step% of tokens
            num_remove = max(1, int(len(sorted_pairs) * step / num_steps))
            remove_indices = set([pair[0] for pair in sorted_pairs[:num_remove]])
            
            # Create perturbed text
            perturbed_tokens = [
                '[MASK]' if i in remove_indices else token
                for i, token in enumerate(tokens)
            ]
            perturbed_text = ' '.join(perturbed_tokens)
            
            # Get confidence
            probs = self._predict(perturbed_text)
            perturbed_conf = probs[predicted_class]
            
            # Compute drop
            drop = original_conf - perturbed_conf
            confidence_drops.append(drop)
        
        # AOPC is average confidence drop
        aopc_score = np.mean(confidence_drops)
        
        return {
            'aopc': float(aopc_score),
            'confidence_drops': [float(d) for d in confidence_drops],
            'max_drop': float(np.max(confidence_drops)),
            'min_drop': float(np.min(confidence_drops))
        }
    
    def evaluate_all_faithfulness(
        self,
        text: str,
        rationale_indices: List[int],
        token_importance_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Compute all faithfulness metrics for a single text.
        
        Args:
            text: Input text
            rationale_indices: Important token indices
            token_importance_scores: Importance score per token
        
        Returns:
            Dictionary with all faithfulness metrics
        """
        return {
            'comprehensiveness': self.comprehensiveness(text, rationale_indices),
            'sufficiency': self.sufficiency(text, rationale_indices),
            'monotonicity': self.monotonicity(text, token_importance_scores),
            'aopc': self.aopc(text, token_importance_scores)
        }
    
    def _predict(self, text: str) -> np.ndarray:
        """
        Get model predictions for text.
        
        Args:
            text: Input text
        
        Returns:
            Probability array (num_classes,)
        """
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return probs


def evaluate_faithfulness(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    rationale_indices_list: List[List[int]],
    token_importance_list: List[List[float]],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate faithfulness metrics across multiple samples.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        texts: List of texts
        rationale_indices_list: List of rationale indices per text
        token_importance_list: List of importance scores per text
        device: Computation device
    
    Returns:
        Aggregated faithfulness metrics
    """
    evaluator = FaithfulnessMetrics(model, tokenizer, device)
    
    # Aggregate metrics
    comp_scores = []
    suff_scores = []
    mono_scores = []
    aopc_scores = []
    
    for text, rationale_indices, token_scores in zip(texts, rationale_indices_list, token_importance_list):
        # Comprehensiveness
        comp = evaluator.comprehensiveness(text, rationale_indices)
        comp_scores.append(comp['comprehensiveness'])
        
        # Sufficiency
        suff = evaluator.sufficiency(text, rationale_indices)
        suff_scores.append(suff['sufficiency'])
        
        # Monotonicity
        mono = evaluator.monotonicity(text, token_scores)
        mono_scores.append(mono['monotonicity'])
        
        # AOPC
        aopc = evaluator.aopc(text, token_scores)
        aopc_scores.append(aopc['aopc'])
    
    # Decision flip rate
    flip_results = evaluator.decision_flip_rate(texts, rationale_indices_list)
    
    return {
        'comprehensiveness': {
            'mean': float(np.mean(comp_scores)),
            'std': float(np.std(comp_scores)),
            'values': comp_scores
        },
        'sufficiency': {
            'mean': float(np.mean(suff_scores)),
            'std': float(np.std(suff_scores)),
            'values': suff_scores
        },
        'monotonicity': {
            'mean': float(np.mean(mono_scores)),
            'std': float(np.std(mono_scores)),
            'values': mono_scores
        },
        'aopc': {
            'mean': float(np.mean(aopc_scores)),
            'std': float(np.std(aopc_scores)),
            'values': aopc_scores
        },
        'decision_flip_rate': flip_results['decision_flip_rate'],
        'num_samples': len(texts)
    }
